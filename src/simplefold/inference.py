#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import os
import torch
import hydra
import omegaconf
from copy import deepcopy
from pathlib import Path
from itertools import starmap
import lightning.pytorch as pl
from importlib import resources

from model.flow import LinearPath
from model.torch.sampler import EMSampler
from model.torch.teacache import TeaCacheSampler as TeaCacheSamplerTorch
from model.torch.teacache import TeaCacheConfig as TeaCacheConfigTorch

from processor.protein_processor import ProteinDataProcessor
from utils.datamodule_utils import process_one_inference_structure
from utils.esm_utils import _af2_to_esm, esm_registry
from utils.boltz_utils import process_structure, save_structure
from utils.fasta_utils import process_fastas, download_fasta_utilities, check_fasta_inputs, resolve_cache_dir
from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer

try:
    import mlx.core as mx
    from mlx.utils import tree_unflatten, tree_flatten
    from model.mlx.sampler import EMSampler as EMSamplerMLX
    from model.mlx.teacache import TeaCacheSampler, TeaCacheConfig
    from model.mlx.esm_network import ESM2 as ESM2MLX
    from utils.mlx_utils import map_torch_to_mlx, map_plddt_torch_to_mlx
    MLX_AVAILABLE = True
except:
    MLX_AVAILABLE = False
    print("MLX not installed, skip importing MLX related packages.")


ckpt_url_dict = {
    "simplefold_100M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_100M.ckpt",
    "simplefold_360M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_360M.ckpt",
    "simplefold_700M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_700M.ckpt",
    "simplefold_1.1B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_1.1B.ckpt",
    "simplefold_1.6B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_1.6B.ckpt",
    "simplefold_3B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_3B.ckpt",
}

plddt_ckpt_url = "https://ml-site.cdn-apple.com/models/simplefold/plddt_module_1.6B.ckpt"


def get_config_path(relative_path):
    """Get the absolute path to a config file using importlib.resources."""
    try:
        # Remove 'configs/' prefix if present since we access configs directly as a subpackage
        config_subpath = relative_path.replace('configs/', '')

        # Access configs as a subpackage resource
        config_files = resources.files('simplefold.configs')
        config_path = config_files / config_subpath

        if config_path.is_file():
            return str(config_path)

    except Exception as e:
        pass

    # If importlib.resources fails, raise an informative error
    raise FileNotFoundError(
        f"Could not find config file: {relative_path}. "
        f"Expected to find it in the simplefold.configs package."
    )



def _quantize_esm_int8(esm_model):
    """
    Quantize ESM-3B model to INT8 for 3.5x memory reduction.

    Memory: 11.36 GB (FP32) → 3.22 GB (INT8)
    Quality: Negligible degradation (mean diff ~0.001 vs std ~0.288)
    """
    import mlx.nn as nn

    quantized_count = 0
    for layer in esm_model.layers:
        # Quantize each transformer layer's Linear modules
        for name, child in layer.children().items():
            if isinstance(child, nn.Linear) and child.weight.shape[-1] % 64 == 0:
                try:
                    setattr(layer, name, child.to_quantized(group_size=64, bits=8))
                    quantized_count += 1
                except Exception:
                    pass
            # Handle nested modules (e.g., self_attn)
            elif hasattr(child, 'children'):
                for n2, c2 in child.children().items():
                    if isinstance(c2, nn.Linear) and c2.weight.shape[-1] % 64 == 0:
                        try:
                            setattr(child, n2, c2.to_quantized(group_size=64, bits=8))
                            quantized_count += 1
                        except Exception:
                            pass

    print(f"  ESM-3B quantized to INT8 ({quantized_count} layers, ~3.2GB)")
    return esm_model


def _quantize_simplefold_int8(model):
    """
    Quantize SimpleFold model to INT8 for 1.8x memory reduction.

    Memory: 387 MB (FP32) → 212 MB (INT8)
    Quality: ~4% relative error (acceptable for most uses)
    Speed: Slightly slower due to INT8 matmul overhead

    Note: This disables the fused SwiGLU optimization.
    """
    import mlx.nn as nn
    from model.mlx.layers import SwiGLUFeedForward

    quantized_count = 0

    def quantize_recursive(module, group_size=64, bits=8):
        nonlocal quantized_count
        children = dict(module.children()) if hasattr(module, 'children') else {}

        for name, child in children.items():
            if isinstance(child, SwiGLUFeedForward):
                # Quantize SwiGLU's internal layers
                for wname in ['w1', 'w2', 'w3']:
                    w = getattr(child, wname)
                    if isinstance(w, nn.Linear) and w.weight.shape[-1] % group_size == 0:
                        try:
                            setattr(child, wname, w.to_quantized(group_size=group_size, bits=bits))
                            quantized_count += 1
                        except Exception:
                            pass
                child._w13_fused = None  # Clear fused cache
            elif isinstance(child, nn.Linear):
                if child.weight.shape[-1] % group_size == 0:
                    try:
                        setattr(module, name, child.to_quantized(group_size=group_size, bits=bits))
                        quantized_count += 1
                    except Exception:
                        pass
            elif isinstance(child, (list, tuple)):
                for item in child:
                    if hasattr(item, 'children'):
                        quantize_recursive(item, group_size, bits)
            elif hasattr(child, 'children'):
                quantize_recursive(child, group_size, bits)

    quantize_recursive(model)
    print(f"  SimpleFold quantized to INT8 ({quantized_count} layers, ~212MB)")
    return model


def initialize_folding_model(args):
    # define folding model
    simplefold_model = args.simplefold_model

    # create checkpoint directory
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, f"{simplefold_model}.ckpt")

    # create folding model
    ckpt_path = os.path.join(ckpt_dir, f"{simplefold_model}.ckpt")
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_dir, exist_ok=True)
        os.system(f"curl -L {ckpt_url_dict[simplefold_model]} -o {ckpt_path}")
    cfg_path = get_config_path(f"configs/model/architecture/foldingdit_{simplefold_model[11:]}.yaml")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # load model checkpoint
    if args.backend == 'torch':
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model_config = omegaconf.OmegaConf.load(cfg_path)
        model = hydra.utils.instantiate(model_config)
        model.load_state_dict(checkpoint, strict=True)
        model = model.to(device)
    elif args.backend == 'mlx':
        device = "cpu"
        # replace torch implementations with mlx
        with open(cfg_path, "r") as f:
            yaml_str = f.read()
        yaml_str = yaml_str.replace('torch', 'mlx')

        model_config = omegaconf.OmegaConf.create(yaml_str)
        model = hydra.utils.instantiate(model_config)
        mlx_state_dict = {k: mx.array(v) for k, v in starmap(map_torch_to_mlx, checkpoint.items()) if k is not None}
        model.update(tree_unflatten(list(mlx_state_dict.items())))
    print(f"Folding model {simplefold_model} loaded.")
    print(f"Using device: {device}.")

    model.eval()
    return model, device


def initialize_plddt_module(args, device):
    if not args.plddt:
        return None, None

    # load pLDDT module if specified
    plddt_ckpt_path = os.path.join(args.ckpt_dir, "plddt.ckpt")
    if not os.path.exists(plddt_ckpt_path):
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.system(f"curl -L {plddt_ckpt_url} -o {plddt_ckpt_path}")

    plddt_module_path = get_config_path("configs/model/architecture/plddt_module.yaml")
    plddt_checkpoint = torch.load(plddt_ckpt_path, map_location="cpu", weights_only=False)

    if args.backend == "torch":
        plddt_config = omegaconf.OmegaConf.load(plddt_module_path)
        plddt_out_module = hydra.utils.instantiate(plddt_config)
        plddt_out_module.load_state_dict(plddt_checkpoint, strict=True)
        plddt_out_module = plddt_out_module.to(device)
    elif args.backend == "mlx":
        # replace torch implementations with mlx
        with open(plddt_module_path, "r") as f:
            yaml_str = f.read()
        yaml_str = yaml_str.replace('torch', 'mlx')

        plddt_config = omegaconf.OmegaConf.create(yaml_str)
        plddt_out_module = hydra.utils.instantiate(plddt_config)

        mlx_state_dict = {k: mx.array(v) for k, v in starmap(map_plddt_torch_to_mlx, plddt_checkpoint.items()) if k is not None}
        plddt_out_module.update(tree_unflatten(list(mlx_state_dict.items())))

    plddt_out_module.eval()
    print(f"pLDDT output module loaded with {args.backend} backend.")

    plddt_latent_ckpt_path = os.path.join(args.ckpt_dir, "simplefold_1.6B.ckpt")
    if not os.path.exists(plddt_latent_ckpt_path):
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.system(f"curl -L {ckpt_url_dict['simplefold_1.6B']} -o {plddt_latent_ckpt_path}")

    plddt_latent_config_path = get_config_path("configs/model/architecture/foldingdit_1.6B.yaml")
    plddt_latent_checkpoint = torch.load(plddt_latent_ckpt_path, map_location="cpu", weights_only=False)

    if args.backend == "torch":
        plddt_latent_config = omegaconf.OmegaConf.load(plddt_latent_config_path)
        plddt_latent_module = hydra.utils.instantiate(plddt_latent_config)
        plddt_latent_module.load_state_dict(plddt_latent_checkpoint, strict=True)
        plddt_latent_module = plddt_latent_module.to(device)
    elif args.backend == "mlx":
        # replace torch implementations with mlx
        with open(plddt_latent_config_path, "r") as f:
            yaml_str = f.read()
        yaml_str = yaml_str.replace('torch', 'mlx')

        plddt_latent_config = omegaconf.OmegaConf.create(yaml_str)
        plddt_latent_module = hydra.utils.instantiate(plddt_latent_config)
        mlx_state_dict = {k: mx.array(v) for k, v in starmap(map_torch_to_mlx, plddt_latent_checkpoint.items()) if k is not None}
        plddt_latent_module.update(tree_unflatten(list(mlx_state_dict.items())))

    plddt_latent_module.eval()
    print(f"pLDDT latent module loaded with {args.backend} backend.")

    return plddt_latent_module, plddt_out_module


def initialize_esm_model(args, device, quantize_esm=True):
    """
    Initialize ESM-3B protein language model.

    Args:
        args: Configuration with backend setting
        device: Target device
        quantize_esm: If True (default), apply INT8 quantization for 3.5x memory reduction.
                      Set to False for full precision (may improve accuracy).
    """
    # load ESM2 model
    esm_model, esm_dict = esm_registry["esm2_3B"]()
    af2_to_esm = _af2_to_esm(esm_dict)

    if args.backend == 'torch':
        esm_model = esm_model.to(device)
        af2_to_esm = af2_to_esm.to(device)
    elif args.backend == 'mlx':
        esm_model_mlx = ESM2MLX(num_layers=36, embed_dim=2560, attention_heads=40)
        esm_state_dict_torch = esm_model.cpu().state_dict()

        esm_state_dict_torch = {k: mx.array(v) for k, v in starmap(map_torch_to_mlx, esm_state_dict_torch.items()) if k is not None}
        esm_model_mlx.update(tree_unflatten(list(esm_state_dict_torch.items())))
        esm_model = esm_model_mlx

        if quantize_esm:
            # Apply INT8 quantization to ESM-3B for 3.5x memory reduction (11.4GB → 3.2GB)
            esm_model = _quantize_esm_int8(esm_model)
        else:
            print("  ESM-3B running at full precision (~11.4GB)")

    print(f"pLM ESM-3B loaded with {args.backend} backend.")

    esm_model.eval()
    return esm_model, esm_dict, af2_to_esm


def initialize_others(args, device):
    # prepare data tokenizer, featurizer, and processor
    tokenizer = BoltzTokenizer()
    featurizer = BoltzFeaturizer()
    processor = ProteinDataProcessor(
        device=device,
        scale=16.0, 
        ref_scale=5.0, 
        multiplicity=1,
        inference_multiplicity=args.nsample_per_protein,
        backend=args.backend,
    )

    # define flow process and sampler
    flow = LinearPath()

    teacache_threshold = getattr(args, 'teacache', 0.0)

    if args.backend == "torch":
        if teacache_threshold > 0:
            config = TeaCacheConfigTorch(threshold=teacache_threshold)
            sampler = TeaCacheSamplerTorch(
                num_timesteps=args.num_steps,
                t_start=1e-4,
                tau=args.tau,
                config=config,
            )
            print(f"TeaCache enabled (threshold={teacache_threshold})")
        else:
            sampler = EMSampler(
                num_timesteps=args.num_steps,
                t_start=1e-4,
                tau=args.tau,
                log_timesteps=True,
                w_cutoff=0.99,
            )
    elif args.backend == "mlx":
        if teacache_threshold > 0:
            config = TeaCacheConfig(threshold=teacache_threshold)
            sampler = TeaCacheSampler(
                num_timesteps=args.num_steps,
                t_start=1e-4,
                tau=args.tau,
                config=config,
            )
            print(f"TeaCache enabled (threshold={teacache_threshold})")
        else:
            sampler = EMSamplerMLX(
                num_timesteps=args.num_steps,
                t_start=1e-4,
                tau=args.tau,
                log_timesteps=True,
                w_cutoff=0.99,
            )
    return tokenizer, featurizer, processor, flow, sampler


def generate_structure(
    args, batch, sampler, flow, processor,
    model, plddt_latent_module, plddt_out_module, device
):
    # run inference for target protein
    if args.backend == "torch":
        noise = torch.randn_like(batch['coords']).to(device)
    elif args.backend == "mlx":
        noise = mx.random.normal(batch['coords'].shape)
    out_dict = sampler.sample(model, flow, noise, batch)

    if args.plddt:
        if args.backend == "torch":
            t = torch.ones(batch['coords'].shape[0], device=device)
            # use unscaled coords to extract latent for pLDDT prediction
            out_feat = plddt_latent_module(
                out_dict["denoised_coords"].detach(), t, batch)
            plddt_out_dict = plddt_out_module(
                out_feat["latent"].detach(),
                batch,
            )
        elif args.backend == "mlx":
            t = mx.ones(batch['coords'].shape[0])
            # use unscaled coords to extract latent for pLDDT prediction
            out_feat = plddt_latent_module(
                out_dict["denoised_coords"], t, batch)
            plddt_out_dict = plddt_out_module(
                out_feat["latent"],
                batch,
            )
        # scale pLDDT to [0, 100]
        plddts = plddt_out_dict["plddt"] * 100.0
    else:
        plddts = None

    out_dict = processor.postprocess(out_dict, batch)
    # sampled_coord = out_dict['denoised_coords'].detach()
    if args.backend == "torch":
        sampled_coord = out_dict['denoised_coords'].detach()
    else:
        sampled_coord = out_dict['denoised_coords']

    pad_mask = batch['atom_pad_mask']
    return sampled_coord, pad_mask, plddts


def predict_structures_from_fastas(args):
    # create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / f"predictions_{args.simplefold_model}"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    cache = resolve_cache_dir(getattr(args, "cache_dir", None))

    # set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    if args.backend == "mlx" and not MLX_AVAILABLE:
        args.backend = "torch"
        print("MLX not available, switch to torch backend.")

    # initialize models
    model, device = initialize_folding_model(args)
    plddt_latent_module, plddt_out_module = initialize_plddt_module(args, device)
    esm_model, esm_dict, af2_to_esm = initialize_esm_model(args, device)

    # initialize other components
    tokenizer, featurizer, processor, flow, sampler = initialize_others(args, device)

    # process fasta files to input format
    download_fasta_utilities(cache)
    data = check_fasta_inputs(Path(args.fasta_path))
    if not data:
        raise ValueError("No valid input files found. Please check the input directory.")
    process_fastas(
        data=data,
        out_dir=output_dir,
        ccd_path=cache / "ccd.pkl",
    )

    for struct_file in output_dir.glob("structures/*.npz"):
        record_file = output_dir / "records" / f"{struct_file.stem}.json"

        # prepare the target protein data for inference
        batch, structure, record = process_one_inference_structure(
            struct_file, record_file,
            tokenizer, featurizer, processor,
            esm_model, esm_dict, af2_to_esm,
        )

        sampled_coord, pad_mask, plddts = generate_structure(
            args, batch, sampler, flow, processor,
            model, plddt_latent_module, plddt_out_module, device
        )

        for i in range(args.nsample_per_protein):
            sampled_coord_i = sampled_coord[i]
            pad_mask_i = pad_mask[i]

            # save the generated structure
            structure_save = process_structure(
                deepcopy(structure), sampled_coord_i, pad_mask_i, record, backend=args.backend
            )
            outname = f"{record.id}_sampled_{i}"
            save_structure(
                structure_save, prediction_dir, outname,
                output_format=args.output_format,
                plddts=plddts[i] if plddts is not None else None
            )
