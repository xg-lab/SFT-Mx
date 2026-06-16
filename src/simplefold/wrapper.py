#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import os
import torch
import hydra
import omegaconf
from pathlib import Path
from itertools import starmap

from model.flow import LinearPath
from model.torch.sampler import EMSampler

from processor.protein_processor import ProteinDataProcessor
from utils.datamodule_utils import process_one_inference_structure
from utils.esm_utils import _af2_to_esm, esm_registry
from utils.boltz_utils import process_structure, save_structure
from utils.fasta_utils import process_fastas, download_fasta_utilities, resolve_cache_dir
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

plddt_ckpt_url = (
    "https://ml-site.cdn-apple.com/models/simplefold/plddt_module_1.6B.ckpt"
)


class ModelWrapper:
    def __init__(
        self,
        simplefold_model,
        plddt=False,
        ckpt_dir="./artifacts",
        backend="mlx",
    ):
        self.simplefold_model = simplefold_model
        self.plddt = plddt
        self.ckpt_dir = Path(ckpt_dir)
        self.backend = "mlx"
        self.folding_model = None
        self.plddt_out_module = None
        self.plddt_latent_module = None
        self.device = "cpu"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def get_device(self):
        return "cpu"

    def from_pretrained_folding_model(self):
        # define folding model
        simplefold_model = self.simplefold_model

        safetensors_path = os.path.join(self.ckpt_dir, f"{simplefold_model}.safetensors")
        if not os.path.exists(safetensors_path):
            # Convert on-demand if needed
            ckpt_path = os.path.join(self.ckpt_dir, f"{simplefold_model}.ckpt")
            if not os.path.exists(ckpt_path):
                os.makedirs(self.ckpt_dir, exist_ok=True)
                os.system(f"curl -L -o {ckpt_path} {ckpt_url_dict[simplefold_model]}")
            
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

        cfg_path = os.path.join(
            "configs/model/architecture", f"foldingdit_{simplefold_model[11:]}.yaml"
        )
        with open(cfg_path, "r") as f:
            yaml_str = f.read()
        yaml_str = yaml_str.replace("torch", "mlx")

        model_config = omegaconf.OmegaConf.create(yaml_str)
        model = hydra.utils.instantiate(model_config)
        
        # Natively load
        weights = mx.load(safetensors_path)
        model.update(tree_unflatten(list(weights.items())))
        del weights
        mx.clear_cache()
            
        import gc
        gc.collect()
        print(f"Folding model {simplefold_model} loaded natively on MLX.")

        model.eval()
        return model

    def from_pretrained_plddt_model(self):
        if not self.plddt:
            return {
                "plddt_out_module": None,
                "plddt_latent_module": None,
            }

        plddt_safetensors = os.path.join(self.ckpt_dir, "plddt.safetensors")
        if not os.path.exists(plddt_safetensors):
            # load pLDDT module if specified
            plddt_ckpt_path = os.path.join(self.ckpt_dir, "plddt.ckpt")
            if not os.path.exists(plddt_ckpt_path):
                os.system(f"curl -L -o {plddt_ckpt_path} {plddt_ckpt_url}")

            print(f"Converting {plddt_ckpt_path} to MLX safetensors format...")
            plddt_checkpoint = torch.load(
                plddt_ckpt_path, map_location="cpu", weights_only=False
            )
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

        plddt_module_path = "configs/model/architecture/plddt_module.yaml"
        with open(plddt_module_path, "r") as f:
            yaml_str = f.read()
        yaml_str = yaml_str.replace("torch", "mlx")

        plddt_config = omegaconf.OmegaConf.create(yaml_str)
        plddt_out_module = hydra.utils.instantiate(plddt_config)

        weights = mx.load(plddt_safetensors)
        plddt_out_module.update(tree_unflatten(list(weights.items())))
        del weights
        mx.clear_cache()
        plddt_out_module.eval()
        print("pLDDT output module loaded natively on MLX.")

        # Latent module
        plddt_latent_safetensors = os.path.join(self.ckpt_dir, "simplefold_1.6B.safetensors")
        if not os.path.exists(plddt_latent_safetensors):
            plddt_latent_ckpt_path = os.path.join(self.ckpt_dir, "simplefold_1.6B.ckpt")
            if not os.path.exists(plddt_latent_ckpt_path):
                os.makedirs(self.ckpt_dir, exist_ok=True)
                os.system(
                    f"curl -L -o {plddt_latent_ckpt_path} {ckpt_url_dict['simplefold_1.6B']}"
                )

            print(f"Converting {plddt_latent_ckpt_path} to MLX safetensors format...")
            plddt_latent_checkpoint = torch.load(
                plddt_latent_ckpt_path, map_location="cpu", weights_only=False
            )
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

        plddt_latent_config_path = "configs/model/architecture/foldingdit_1.6B.yaml"
        with open(plddt_latent_config_path, "r") as f:
            yaml_str = f.read()
        yaml_str = yaml_str.replace("torch", "mlx")

        plddt_latent_config = omegaconf.OmegaConf.create(yaml_str)
        plddt_latent_module = hydra.utils.instantiate(plddt_latent_config)

        weights = mx.load(plddt_latent_safetensors)
        plddt_latent_module.update(tree_unflatten(list(weights.items())))
        del weights
        mx.clear_cache()
        plddt_latent_module.eval()
        print("pLDDT latent module loaded natively on MLX.")

        return {
            "plddt_out_module": plddt_out_module,
            "plddt_latent_module": plddt_latent_module,
        }


class InferenceWrapper:
    def __init__(
        self,
        output_dir,
        prediction_dir,
        num_steps,
        nsample_per_protein,
        tau,
        device,
        backend,
        teacache: bool = False,
        teacache_threshold: float = 0.15,
        cache_dir: str = None,
    ):
        """
        Initialize inference wrapper.

        Args:
            teacache: Enable TeaCache acceleration (MLX only). Provides ~10x speedup
                      with minimal quality loss.
            teacache_threshold: Cache threshold (0.1=quality, 0.2=speed). Default 0.15.
            cache_dir: Shared cache directory for ccd.pkl / boltz1_conf.ckpt.
                       Defaults to ``artifacts/cache`` so these heavy downloads are
                       fetched once and reused across runs.
        """
        self.num_steps = num_steps
        self.nsample_per_protein = nsample_per_protein
        self.tau = tau
        self.device = "cpu"
        self.backend = "mlx"
        self.teacache = teacache
        self.teacache_threshold = teacache_threshold

        # create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # shared cache for ccd.pkl / boltz1_conf.ckpt (default: artifacts/cache)
        cache = resolve_cache_dir(cache_dir)

        # per-run working dir for input.fasta, structures/, records/
        work_dir = output_dir / "tmp"
        work_dir.mkdir(parents=True, exist_ok=True)

        # create prediction directory
        prediction_dir = output_dir / prediction_dir
        prediction_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = output_dir
        self.cache = cache
        self.work_dir = work_dir
        self.prediction_dir = prediction_dir

        self.esm_model = None
        self.esm_dict = None
        self.af2_to_esm = None
        self.initialize_esm_model()
        self.initialize_others()

    def initialize_esm_model(self):
        import gc
        from inference import _quantize_esm_int8

        quantized_weights_path = self.cache / "esm2_t36_3B_UR50D_quantized_8bit.safetensors"
        
        # Optimized load for MLX: load small 8M model to extract alphabet dictionary without loading 3B model weights
        _, esm_dict = esm_registry["esm2_8M"]()
        af2_to_esm = _af2_to_esm(esm_dict)

        # Initialize the MLX architecture directly
        esm_model_mlx = ESM2MLX(num_layers=36, embed_dim=2560, attention_heads=40)

        if quantized_weights_path.exists():
            print(f"[MLX Memory Optimizer] Loading pre-quantized 8-bit ESM-3B weights from: {quantized_weights_path.name}")
            esm_model_mlx = _quantize_esm_int8(esm_model_mlx)
            
            weights = mx.load(str(quantized_weights_path))
            esm_model_mlx.update(tree_unflatten(list(weights.items())))
            del weights
            gc.collect()
            esm_model = esm_model_mlx
        else:
            hub_dir = torch.hub.get_dir()
            checkpoint_path = Path(hub_dir) / "checkpoints" / "esm2_t36_3B_UR50D.pt"
            if not checkpoint_path.exists():
                print("ESM-3B checkpoint not found locally, downloading via torch.hub...")
                temp_model, _ = esm_registry["esm2_3B"]()
                del temp_model
                gc.collect()

            print(f"[MLX Memory Optimizer] Memory-mapping ESM-3B checkpoint: {checkpoint_path.name}")
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True, weights_only=False)
            state_dict = checkpoint["model"]

            mlx_state_dict = {}
            for k in list(state_dict.keys()):
                v = state_dict[k]
                
                k_mapped = k
                if k_mapped.startswith("encoder.sentence_encoder."):
                    k_mapped = k_mapped.replace("encoder.sentence_encoder.", "")
                elif k_mapped.startswith("encoder."):
                    k_mapped = k_mapped.replace("encoder.", "")
                
                k_mlx, v_mlx = map_torch_to_mlx(k_mapped, v)
                if k_mlx is not None:
                    mlx_state_dict[k_mlx] = mx.array(v_mlx)
                del state_dict[k]

            del checkpoint, state_dict
            esm_model_mlx.update(tree_unflatten(list(mlx_state_dict.items())))
            del mlx_state_dict
            gc.collect()

            # Quantize ESM-3B for MLX in wrapper too
            esm_model = _quantize_esm_int8(esm_model_mlx)
            
            try:
                print(f"[MLX Memory Optimizer] Saving 8-bit quantized weights to: {quantized_weights_path.absolute()}")
                weights_to_save = dict(tree_flatten(esm_model.parameters()))
                mx.save_safetensors(str(quantized_weights_path), weights_to_save)
                del weights_to_save
                gc.collect()
            except Exception as e:
                print(f"Warning: Could not save quantized weights: {e}")

        print("pLM ESM-3B loaded with mlx backend.")
        self.esm_model = esm_model.eval()
        self.esm_dict = esm_dict
        self.af2_to_esm = af2_to_esm

    def initialize_others(self):
        # prepare data tokenizer, featurizer, and processor
        self.tokenizer = BoltzTokenizer()
        self.featurizer = BoltzFeaturizer()
        self.processor = ProteinDataProcessor(
            device="cpu",
            scale=16.0,
            ref_scale=5.0,
            multiplicity=1,
            inference_multiplicity=self.nsample_per_protein,
            backend="mlx",
        )

        # define flow process and sampler
        self.flow = LinearPath()

        if self.teacache:
            config = TeaCacheConfig(threshold=self.teacache_threshold)
            self.sampler = TeaCacheSampler(
                num_timesteps=self.num_steps,
                t_start=1e-4,
                tau=self.tau,
                config=config,
            )
            print(f"TeaCache enabled (threshold={self.teacache_threshold})")
        else:
            self.sampler = EMSamplerMLX(
                num_timesteps=self.num_steps,
                t_start=1e-4,
                tau=self.tau,
                log_timesteps=True,
                w_cutoff=0.99,
            )

    def release_esm_model(self):
        """Release ESM model from memory/GPU to free up RAM."""
        if hasattr(self, "esm_model") and self.esm_model is not None:
            print("[InferenceWrapper] Releasing ESM model from RAM/GPU...")
            del self.esm_model
            self.esm_model = None
        if hasattr(self, "af2_to_esm") and self.af2_to_esm is not None:
            del self.af2_to_esm
            self.af2_to_esm = None
        import gc
        gc.collect()
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()

    def process_input(self, aa_seq, release_esm=False):
        if self.esm_model is None:
            print("[InferenceWrapper] Loading ESM model for feature extraction...")
            self.initialize_esm_model()

        # download shared utilities into the cache dir if missing
        download_fasta_utilities(self.cache)
        # save the input sequence to a fasta file in the per-run work dir
        with open(self.work_dir / "input.fasta", "w") as f:
            f.write(f">A|Protein\n{aa_seq}\n")
        data = [self.work_dir / "input.fasta"]
        process_fastas(
            data=data,
            out_dir=self.work_dir,
            ccd_path=self.cache / "ccd.pkl",
        )

        # prepare the target protein data for inference
        struct_file = self.work_dir / "structures" / "input.npz"
        record_file = self.work_dir / "records" / "input.json"
        batch, structure, record = process_one_inference_structure(
            struct_file,
            record_file,
            self.tokenizer,
            self.featurizer,
            self.processor,
            self.esm_model,
            self.esm_dict,
            self.af2_to_esm,
        )

        if release_esm:
            self.release_esm_model()

        return batch, structure, record

    def run_inference(self, batch, model, plddt_model, device):
        noise = mx.random.normal(batch["coords"].shape)
        out_dict = self.sampler.sample(model, self.flow, noise, batch)

        plddt_out_module = plddt_model["plddt_out_module"]
        plddt_latent_module = plddt_model["plddt_latent_module"]

        if plddt_latent_module is None or plddt_out_module is None:
            plddts = None
        else:
            t = mx.ones(batch["coords"].shape[0])
            out_feat = plddt_latent_module(out_dict["denoised_coords"], t, batch)
            plddt_out_dict = plddt_out_module(
                out_feat["latent"],
                batch,
            )
            plddts = plddt_out_dict["plddt"] * 100.0

        out_dict = self.processor.postprocess(out_dict, batch)
        sampled_coord = out_dict["denoised_coords"]

        return {
            "sampled_coord": sampled_coord,
            "pad_mask": batch["atom_pad_mask"],
            "plddts": plddts,
        }

    def save_result(self, structure, record, results, out_name):
        sampled_coord = results["sampled_coord"]
        pad_mask = results["pad_mask"]
        plddt = results["plddts"]

        save_paths = []
        for i in range(sampled_coord.shape[0]):
            sampled_coord_i = sampled_coord[i]
            pad_mask_i = pad_mask[i]
            plddt_i = plddt[i] if plddt is not None else None
            out_name_i = f"{out_name}_sampled_{i}"
            # save the generated structure
            structure_save = process_structure(
                structure, sampled_coord_i, pad_mask_i, record, backend="mlx"
            )
            save_structure(
                structure_save,
                self.prediction_dir,
                out_name_i,
                output_format="mmcif",
                plddts=plddt_i,
            )
            save_paths.append(self.prediction_dir / f"{out_name_i}.cif")
        return save_paths
