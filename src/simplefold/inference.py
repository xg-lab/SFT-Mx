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

from processor.protein_processor import ProteinDataProcessor
from utils.datamodule_utils import process_one_inference_structure
from utils.esm_utils import _af2_to_esm, esm_registry
from utils.boltz_utils import process_structure, save_structure
from utils.fasta_utils import process_fastas, download_fasta_utilities, check_fasta_inputs, resolve_cache_dir
from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer

import mlx.core as mx
from mlx.utils import tree_unflatten, tree_flatten
from model.mlx.sampler import EMSampler as EMSamplerMLX
from model.mlx.teacache import TeaCacheSampler, TeaCacheConfig
from model.mlx.esm_network import ESM2 as ESM2MLX
from utils.mlx_utils import map_torch_to_mlx, map_plddt_torch_to_mlx
MLX_AVAILABLE = True


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
    safetensors_path = os.path.join(ckpt_dir, f"{simplefold_model}.safetensors")

    if not os.path.exists(safetensors_path):
        # Fallback to download and convert on the fly
        ckpt_path = os.path.join(ckpt_dir, f"{simplefold_model}.ckpt")
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_dir, exist_ok=True)
            print(f"Downloading folding checkpoint {simplefold_model}...")
            os.system(f"curl -L {ckpt_url_dict[simplefold_model]} -o {ckpt_path}")
        
        print(f"Converting {ckpt_path} to MLX safetensors format...")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        mlx_state_dict = {}
        for k, v in checkpoint.items():
            k_mlx, v_np = map_torch_to_mlx(k, v)
            if k_mlx is not None:
                if v_np.dtype in (np.float32, np.float16):
                    mlx_state_dict[k_mlx] = mx.array(v_np).astype(mx.float16)
                else:
                    mlx_state_dict[k_mlx] = mx.array(v_np)
        mx.save_safetensors(safetensors_path, mlx_state_dict)
        del checkpoint, mlx_state_dict
        import gc
        gc.collect()

    cfg_path = get_config_path(f"configs/model/architecture/foldingdit_{simplefold_model[11:]}.yaml")

    # replace torch implementations with mlx
    with open(cfg_path, "r") as f:
        yaml_str = f.read()
    yaml_str = yaml_str.replace('torch', 'mlx')

    model_config = omegaconf.OmegaConf.create(yaml_str)
    model = hydra.utils.instantiate(model_config)
    
    # Load native safetensors
    weights = mx.load(safetensors_path)
    model.update(tree_unflatten(list(weights.items())))
    del weights
    mx.clear_cache()
    
    import gc
    gc.collect()
    print(f"Folding model {simplefold_model} loaded natively on MLX.")

    model.eval()
    return model, "cpu"


def initialize_plddt_module(args, device):
    if not args.plddt:
        return None, None

    ckpt_dir = Path(args.ckpt_dir)
    plddt_safetensors = os.path.join(ckpt_dir, "plddt.safetensors")
    if not os.path.exists(plddt_safetensors):
        plddt_ckpt_path = os.path.join(ckpt_dir, "plddt.ckpt")
        if not os.path.exists(plddt_ckpt_path):
            os.makedirs(ckpt_dir, exist_ok=True)
            print("Downloading pLDDT checkpoint...")
            os.system(f"curl -L {plddt_ckpt_url} -o {plddt_ckpt_path}")
        
        print(f"Converting {plddt_ckpt_path} to MLX safetensors format...")
        plddt_checkpoint = torch.load(plddt_ckpt_path, map_location="cpu", weights_only=False)
        mlx_state_dict = {}
        for k, v in plddt_checkpoint.items():
            k_mlx, v_np = map_plddt_torch_to_mlx(k, v)
            if k_mlx is not None:
                if v_np.dtype in (np.float32, np.float16):
                    mlx_state_dict[k_mlx] = mx.array(v_np).astype(mx.float16)
                else:
                    mlx_state_dict[k_mlx] = mx.array(v_np)
        mx.save_safetensors(plddt_safetensors, mlx_state_dict)
        del plddt_checkpoint, mlx_state_dict
        import gc
        gc.collect()

    plddt_module_path = get_config_path("configs/model/architecture/plddt_module.yaml")
    with open(plddt_module_path, "r") as f:
        yaml_str = f.read()
    yaml_str = yaml_str.replace('torch', 'mlx')

    plddt_config = omegaconf.OmegaConf.create(yaml_str)
    plddt_out_module = hydra.utils.instantiate(plddt_config)
    
    weights = mx.load(plddt_safetensors)
    plddt_out_module.update(tree_unflatten(list(weights.items())))
    del weights
    mx.clear_cache()

    import gc
    gc.collect()
    plddt_out_module.eval()
    print("pLDDT output module loaded natively on MLX.")

    # Latent module
    plddt_latent_safetensors = os.path.join(ckpt_dir, "simplefold_1.6B.safetensors")
    if not os.path.exists(plddt_latent_safetensors):
        plddt_latent_ckpt_path = os.path.join(ckpt_dir, "simplefold_1.6B.ckpt")
        if not os.path.exists(plddt_latent_ckpt_path):
            os.makedirs(ckpt_dir, exist_ok=True)
            print("Downloading simplefold_1.6B checkpoint for pLDDT...")
            os.system(f"curl -L {ckpt_url_dict['simplefold_1.6B']} -o {plddt_latent_ckpt_path}")
            
        print(f"Converting {plddt_latent_ckpt_path} to MLX safetensors format...")
        plddt_latent_checkpoint = torch.load(plddt_latent_ckpt_path, map_location="cpu", weights_only=False)
        mlx_state_dict = {}
        for k, v in plddt_latent_checkpoint.items():
            k_mlx, v_np = map_torch_to_mlx(k, v)
            if k_mlx is not None:
                if v_np.dtype in (np.float32, np.float16):
                    mlx_state_dict[k_mlx] = mx.array(v_np).astype(mx.float16)
                else:
                    mlx_state_dict[k_mlx] = mx.array(v_np)
        mx.save_safetensors(plddt_latent_safetensors, mlx_state_dict)
        del plddt_latent_checkpoint, mlx_state_dict
        import gc
        gc.collect()

    plddt_latent_config_path = get_config_path("configs/model/architecture/foldingdit_1.6B.yaml")
    with open(plddt_latent_config_path, "r") as f:
        yaml_str = f.read()
    yaml_str = yaml_str.replace('torch', 'mlx')

    plddt_config = omegaconf.OmegaConf.create(yaml_str)
    plddt_latent_module = hydra.utils.instantiate(plddt_config)
    
    weights = mx.load(plddt_latent_safetensors)
    plddt_latent_module.update(tree_unflatten(list(weights.items())))
    del weights
    mx.clear_cache()

    import gc
    gc.collect()
    plddt_latent_module.eval()
    print("pLDDT latent module loaded natively on MLX.")

    return plddt_latent_module, plddt_out_module


def initialize_esm_model(args, device, quantize_esm=True):
    """
    Initialize ESM-3B protein language model.
    """
    import gc
    
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    quantized_weights_path = ckpt_dir / "esm2_t36_3B_UR50D_quantized_8bit.safetensors"
    
    # Load small 8M model to extract alphabet dictionary without loading 3B model weights
    _, esm_dict = esm_registry["esm2_8M"]()
    af2_to_esm = _af2_to_esm(esm_dict)

    # Initialize the MLX architecture directly
    esm_model_mlx = ESM2MLX(num_layers=36, embed_dim=2560, attention_heads=40)

    if quantize_esm and quantized_weights_path.exists():
        print(f"[MLX Memory Optimizer] Loading pre-quantized 8-bit ESM-3B weights from: {quantized_weights_path.name}")
        # Quantize the empty model structure first
        esm_model_mlx = _quantize_esm_int8(esm_model_mlx)
        
        # Load quantized weights directly into the model
        weights = mx.load(str(quantized_weights_path))
        esm_model_mlx.update(tree_unflatten(list(weights.items())))
        del weights
        gc.collect()
        esm_model = esm_model_mlx
    else:
        # Find PyTorch Hub checkpoint path
        hub_dir = torch.hub.get_dir()
        checkpoint_path = Path(hub_dir) / "checkpoints" / "esm2_t36_3B_UR50D.pt"
        if not checkpoint_path.exists():
            print("ESM-3B checkpoint not found locally, downloading via torch.hub...")
            # This triggers download and saves it, but we don't keep the PyTorch model
            temp_model, _ = esm_registry["esm2_3B"]()
            del temp_model
            gc.collect()

        # Load weights directly via memory-mapping (mmap=True) to avoid loading the full 11.6GB into CPU memory
        print(f"[MLX Memory Optimizer] Memory-mapping ESM-3B checkpoint: {checkpoint_path.name}")
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True, weights_only=False)
        state_dict = checkpoint["model"]

        # Map state dict keys to MLX arrays on-demand and delete the mapped tensors from CPU memory
        mlx_state_dict = {}
        for k in list(state_dict.keys()):
            v = state_dict[k]
            
            # Strip PyTorch Hub model checkpoint prefixes to match state_dict format
            k_mapped = k
            if k_mapped.startswith("encoder.sentence_encoder."):
                k_mapped = k_mapped.replace("encoder.sentence_encoder.", "")
            elif k_mapped.startswith("encoder."):
                k_mapped = k_mapped.replace("encoder.", "")
            
            k_mlx, v_mlx = map_torch_to_mlx(k_mapped, v)
            if k_mlx is not None:
                mlx_state_dict[k_mlx] = mx.array(v_mlx)
            # Free CPU memory reference immediately
            del state_dict[k]

        del checkpoint, state_dict
        esm_model_mlx.update(tree_unflatten(list(mlx_state_dict.items())))
        del mlx_state_dict
        gc.collect()

        esm_model = esm_model_mlx

        if quantize_esm:
            # Apply INT8 quantization to ESM-3B for 3.5x memory reduction (11.4GB → 3.2GB)
            esm_model = _quantize_esm_int8(esm_model)
            
            # Save the quantized weights for future runs
            try:
                print(f"[MLX Memory Optimizer] Saving 8-bit quantized weights to: {quantized_weights_path.absolute()}")
                weights_to_save = dict(tree_flatten(esm_model.parameters()))
                mx.save_safetensors(str(quantized_weights_path), weights_to_save)
                del weights_to_save
                gc.collect()
            except Exception as e:
                print(f"Warning: Could not save quantized weights: {e}")
        else:
            print("  ESM-3B running at full precision (~11.4GB)")

    print("pLM ESM-3B loaded with mlx backend.")
    esm_model.eval()
    return esm_model, esm_dict, af2_to_esm


def initialize_others(args, device):
    # prepare data tokenizer, featurizer, and processor
    tokenizer = BoltzTokenizer()
    featurizer = BoltzFeaturizer()
    processor = ProteinDataProcessor(
        device="cpu",
        scale=16.0, 
        ref_scale=5.0, 
        multiplicity=1,
        inference_multiplicity=args.nsample_per_protein,
        backend="mlx",
    )

    # define flow process and sampler
    flow = LinearPath()

    teacache_threshold = getattr(args, 'teacache', 0.0)

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
    noise = mx.random.normal(batch['coords'].shape)
    out_dict = sampler.sample(model, flow, noise, batch)

    if args.plddt:
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

    # MLX backend ONLY
    device = "cpu"

    # -------------------------------------------------------------
    # STEP 1: Feature Extraction using ESM
    # -------------------------------------------------------------
    print("\n--- Step 1: Loading ESM model for Feature Extraction ---")
    esm_model, esm_dict, af2_to_esm = initialize_esm_model(args, device)
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

    # Pre-extract ESM features for all structures
    preprocessed_jobs = []
    for struct_file in output_dir.glob("structures/*.npz"):
        record_file = output_dir / "records" / f"{struct_file.stem}.json"

        print(f"Extracting ESM features for: {struct_file.name}")
        batch, structure, record = process_one_inference_structure(
            struct_file, record_file,
            tokenizer, featurizer, processor,
            esm_model, esm_dict, af2_to_esm,
        )
        preprocessed_jobs.append((batch, structure, record))

    # -------------------------------------------------------------
    # RELEASE RAM: Unload ESM immediately
    # -------------------------------------------------------------
    print("\n--- Releasing ESM model RAM/GPU memory ---")
    del esm_model, af2_to_esm
    import gc
    gc.collect()
    
    import mlx.core as mx
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()

    # -------------------------------------------------------------
    # STEP 2a: Coordinate Generation using Folding Model
    # -------------------------------------------------------------
    print("\n--- Step 2: Loading Folding Model ---")
    model, device = initialize_folding_model(args)

    raw_coords_jobs = []
    for batch, structure, record in preprocessed_jobs:
        print(f"Generating structure coordinates for: {record.id}")
        
        noise = mx.random.normal(batch['coords'].shape)
        out_dict = sampler.sample(model, flow, noise, batch)
        denoised_coords = out_dict["denoised_coords"]
            
        # Run coordinate post-processing
        post_out = processor.postprocess(out_dict, batch)
        sampled_coord = post_out['denoised_coords']
            
        pad_mask = batch['atom_pad_mask']
        raw_coords_jobs.append((batch, structure, record, denoised_coords, sampled_coord, pad_mask))

    # Free folding model and clear device memory caches immediately
    print("\n--- Releasing Folding Model RAM/GPU memory ---")
    del model
    gc.collect()
    
    import mlx.core as mx
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()

    # -------------------------------------------------------------
    # STEP 2b: pLDDT Confidence Score Prediction
    # -------------------------------------------------------------
    plddt_jobs = {}
    if args.plddt:
        print("\n--- Step 3: Loading pLDDT Model ---")
        plddt_latent_module, plddt_out_module = initialize_plddt_module(args, device)
        
        for batch, structure, record, denoised_coords, _, _ in raw_coords_jobs:
            print(f"Predicting confidence (pLDDT) for: {record.id}")
            t = mx.ones(batch['coords'].shape[0])
            out_feat = plddt_latent_module(denoised_coords, t, batch)
            plddt_out_dict = plddt_out_module(out_feat["latent"], batch)
                
            plddts = plddt_out_dict["plddt"] * 100.0
            plddt_jobs[record.id] = plddts

        # Free pLDDT models and clear device memory caches immediately
        print("\n--- Releasing pLDDT Model RAM/GPU memory ---")
        del plddt_latent_module, plddt_out_module
        gc.collect()
        
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()

    # -------------------------------------------------------------
    # STEP 2c: Post-Processing & Output Generation
    # -------------------------------------------------------------
    for batch, structure, record, _, sampled_coord, pad_mask in raw_coords_jobs:
        print(f"Saving final structure for: {record.id}")
        plddts = plddt_jobs.get(record.id, None)
        
        for i in range(args.nsample_per_protein):
            sampled_coord_i = sampled_coord[i]
            pad_mask_i = pad_mask[i]

            # save the generated structure
            structure_save = process_structure(
                deepcopy(structure), sampled_coord_i, pad_mask_i, record, backend="mlx"
            )
            outname = f"{record.id}_sampled_{i}"
            save_structure(
                structure_save, prediction_dir, outname,
                output_format=args.output_format,
                plddts=plddts[i] if plddts is not None else None
            )
