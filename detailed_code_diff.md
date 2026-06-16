# Detailed Codebase Diff: Optimized vs Official Repository

This document contains the exact code differences (git diffs) for all modified files in the codebase compared to the official upstream repository.

---

## File: .gitignore

```diff
diff --git a/.gitignore b/.gitignore
index 818fa72..d1e348b 100644
--- a/.gitignore
+++ b/.gitignore
@@ -9,4 +9,6 @@ artifacts/
 
 # Large data (hosted on Zenodo/HuggingFace)
 publication/data/structures/
-publication/data/structures.zip
\ No newline at end of file
+publication/data/structures.zip# pixi environments
+.pixi/*
+!.pixi/config.toml
```

---

## File: server.py

```diff
diff --git a/server.py b/server.py
index 00bac56..ff31cf7 100644
--- a/server.py
+++ b/server.py
@@ -182,8 +182,7 @@ class WorkerModel:
             # Unload current model
             if self.model is not None:
                 del self.model
-                import mlx.core as mx
-                mx.metal.clear_cache() if hasattr(mx.metal, 'clear_cache') else None
+                mx.clear_cache() if hasattr(mx, 'clear_cache') else (mx.metal.clear_cache() if hasattr(mx.metal, 'clear_cache') else None)
 
             # Load new model
             import argparse
@@ -213,8 +212,7 @@ class WorkerModel:
                 del self.model
                 self.model = None
                 self.model_name = None
-                import mlx.core as mx
-                mx.metal.clear_cache() if hasattr(mx.metal, 'clear_cache') else None
+                mx.clear_cache() if hasattr(mx, 'clear_cache') else (mx.metal.clear_cache() if hasattr(mx.metal, 'clear_cache') else None)
                 print(f"[Worker {self.worker_id}] Model unloaded after {self.config.warmup_timeout}s idle")
 
     def is_loaded(self) -> bool:
@@ -486,6 +484,14 @@ def inference_worker_thread(queue: JobQueue, engine: InferenceEngine, worker_mod
             job.error = str(e)
             queue.complete(job, success=False)
             print(f"[InferenceWorker] Failed {job.id}: {e}")
+        finally:
+            import gc
+            import mlx.core as mx
+            gc.collect()
+            if hasattr(mx, 'clear_cache'):
+                mx.clear_cache()
+            elif hasattr(mx.metal, 'clear_cache'):
+                mx.metal.clear_cache()
```

---

## File: src/simplefold/cli.py

```diff
diff --git a/src/simplefold/cli.py b/src/simplefold/cli.py
index 4f4064e..197e3fd 100644
--- a/src/simplefold/cli.py
+++ b/src/simplefold/cli.py
@@ -10,6 +10,8 @@ import argparse
 from simplefold import __version__
 from simplefold.inference import predict_structures_from_fastas
 
+MLX_DEFAULT = 'mlx'
+
 
 def main():
     parser = argparse.ArgumentParser(
@@ -27,7 +29,7 @@ def main():
     parser.add_argument("--nsample_per_protein", type=int, default=1, help="Number of samples to generate per protein.")
     parser.add_argument("--plddt", action="store_true", help="Enable pLDDT prediction.")
     parser.add_argument("--output_format", type=str, default="mmcif", choices=["pdb", "mmcif"], help="Output file format.")
-    parser.add_argument("--backend", type=str, default='torch', choices=['torch', 'mlx'], help="Backend to run inference either torch or mlx")
+    parser.add_argument("--backend", type=str, default="mlx", choices=['mlx'], help="Backend to run inference (strictly mlx)")
     parser.add_argument("--teacache", type=float, default=0.1, help="Enable TeaCache with threshold (0.0 to 1.0)")
     parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
     parser.add_argument(
```

---

## File: src/simplefold/inference.py

```diff
diff --git a/src/simplefold/inference.py b/src/simplefold/inference.py
index fef3a04..42e3567 100644
--- a/src/simplefold/inference.py
+++ b/src/simplefold/inference.py
@@ -7,6 +7,7 @@ import os
 import torch
 import hydra
 import omegaconf
+import numpy as np
 from copy import deepcopy
 from pathlib import Path
 from itertools import starmap
@@ -14,10 +15,6 @@ import lightning.pytorch as pl
 from importlib import resources
 
 from model.flow import LinearPath
-from model.torch.sampler import EMSampler
-from model.torch.teacache import TeaCacheSampler as TeaCacheSamplerTorch
-from model.torch.teacache import TeaCacheConfig as TeaCacheConfigTorch
-
 from processor.protein_processor import ProteinDataProcessor
 from utils.datamodule_utils import process_one_inference_structure
 from utils.esm_utils import _af2_to_esm, esm_registry
@@ -26,17 +23,13 @@ from utils.fasta_utils import process_fastas, download_fasta_utilities, check_fa
 from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
 from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
 
-try:
-    import mlx.core as mx
-    from mlx.utils import tree_unflatten, tree_flatten
-    from model.mlx.sampler import EMSampler as EMSamplerMLX
-    from model.mlx.teacache import TeaCacheSampler, TeaCacheConfig
-    from model.mlx.esm_network import ESM2 as ESM2MLX
-    from utils.mlx_utils import map_torch_to_mlx, map_plddt_torch_to_mlx
-    MLX_AVAILABLE = True
-except:
-    MLX_AVAILABLE = False
-    print("MLX not installed, skip importing MLX related packages.")
+import mlx.core as mx
+from mlx.utils import tree_unflatten, tree_flatten
+from model.mlx.sampler import EMSampler as EMSamplerMLX
+from model.mlx.teacache import TeaCacheSampler, TeaCacheConfig
+from model.mlx.esm_network import ESM2 as ESM2MLX
+from utils.mlx_utils import map_torch_to_mlx, map_plddt_torch_to_mlx
+MLX_AVAILABLE = True
 
 
 ckpt_url_dict = {
@@ -164,106 +157,142 @@ def initialize_folding_model(args):
 
     # create checkpoint directory
     ckpt_dir = Path(args.ckpt_dir)
-    ckpt_path = os.path.join(ckpt_dir, f"{simplefold_model}.ckpt")
+    safetensors_path = os.path.join(ckpt_dir, f"{simplefold_model}.safetensors")
+
+    if not os.path.exists(safetensors_path):
+        # Fallback to download and convert on the fly
+        ckpt_path = os.path.join(ckpt_dir, f"{simplefold_model}.ckpt")
+        if not os.path.exists(ckpt_path):
+            os.makedirs(ckpt_dir, exist_ok=True)
+            print(f"Downloading folding checkpoint {simplefold_model}...")
+            os.system(f"curl -L {ckpt_url_dict[simplefold_model]} -o {ckpt_path}")
+        
+        print(f"Converting {ckpt_path} to MLX safetensors format...")
+        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
+        mlx_state_dict = {}
+        for k, v in checkpoint.items():
+            k_mlx, v_np = map_torch_to_mlx(k, v)
+            if k_mlx is not None:
+                if v_np.dtype in (np.float32, np.float16):
+                    mlx_state_dict[k_mlx] = mx.array(v_np).astype(mx.float16)
+                else:
+                    mlx_state_dict[k_mlx] = mx.array(v_np)
+        mx.save_safetensors(safetensors_path, mlx_state_dict)
+        del checkpoint, mlx_state_dict
+        import gc
+        gc.collect()
 
-    # create folding model
-    ckpt_path = os.path.join(ckpt_dir, f"{simplefold_model}.ckpt")
-    if not os.path.exists(ckpt_path):
-        os.makedirs(ckpt_dir, exist_ok=True)
-        os.system(f"curl -L {ckpt_url_dict[simplefold_model]} -o {ckpt_path}")
     cfg_path = get_config_path(f"configs/model/architecture/foldingdit_{simplefold_model[11:]}.yaml")
 
-    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
-
-    # load model checkpoint
-    if args.backend == 'torch':
-        if torch.cuda.is_available():
-            device = torch.device("cuda")
-        elif torch.backends.mps.is_available():
-            device = torch.device("mps")
-        else:
-            device = torch.device("cpu")
-        model_config = omegaconf.OmegaConf.load(cfg_path)
-        model = hydra.utils.instantiate(model_config)
-        model.load_state_dict(checkpoint, strict=True)
-        model = model.to(device)
-    elif args.backend == 'mlx':
-        device = "cpu"
-        # replace torch implementations with mlx
-        with open(cfg_path, "r") as f:
-            yaml_str = f.read()
-        yaml_str = yaml_str.replace('torch', 'mlx')
-
-        model_config = omegaconf.OmegaConf.create(yaml_str)
-        model = hydra.utils.instantiate(model_config)
-        mlx_state_dict = {k: mx.array(v) for k, v in starmap(map_torch_to_mlx, checkpoint.items()) if k is not None}
-        model.update(tree_unflatten(list(mlx_state_dict.items())))
-    print(f"Folding model {simplefold_model} loaded.")
-    print(f"Using device: {device}.")
+    # replace torch implementations with mlx
+    with open(cfg_path, "r") as f:
+        yaml_str = f.read()
+    yaml_str = yaml_str.replace('torch', 'mlx')
+
+    model_config = omegaconf.OmegaConf.create(yaml_str)
+    model = hydra.utils.instantiate(model_config)
+    
+    # Load native safetensors
+    weights = mx.load(safetensors_path)
+    model.update(tree_unflatten(list(weights.items())))
+    del weights
+    mx.clear_cache()
+    
+    import gc
+    gc.collect()
+    print(f"Folding model {simplefold_model} loaded natively on MLX.")
 
     model.eval()
-    return model, device
+    return model, "cpu"
 
 
 def initialize_plddt_module(args, device):
     if not args.plddt:
         return None, None
 
-    # load pLDDT module if specified
-    plddt_ckpt_path = os.path.join(args.ckpt_dir, "plddt.ckpt")
-    if not os.path.exists(plddt_ckpt_path):
-        os.makedirs(args.ckpt_dir, exist_ok=True)
-        os.system(f"curl -L {plddt_ckpt_url} -o {plddt_ckpt_path}")
+    ckpt_dir = Path(args.ckpt_dir)
+    plddt_safetensors = os.path.join(ckpt_dir, "plddt.safetensors")
+    if not os.path.exists(plddt_safetensors):
+        plddt_ckpt_path = os.path.join(ckpt_dir, "plddt.ckpt")
+        if not os.path.exists(plddt_ckpt_path):
+            os.makedirs(ckpt_dir, exist_ok=True)
+            print("Downloading pLDDT checkpoint...")
+            os.system(f"curl -L {plddt_ckpt_url} -o {plddt_ckpt_path}")
+        
+        print(f"Converting {plddt_ckpt_path} to MLX safetensors format...")
+        plddt_checkpoint = torch.load(plddt_ckpt_path, map_location="cpu", weights_only=False)
+        mlx_state_dict = {}
+        for k, v in plddt_checkpoint.items():
+            k_mlx, v_np = map_plddt_torch_to_mlx(k, v)
+            if k_mlx is not None:
+                if v_np.dtype in (np.float32, np.float16):
+                    mlx_state_dict[k_mlx] = mx.array(v_np).astype(mx.float16)
+                else:
+                    mlx_state_dict[k_mlx] = mx.array(v_np)
+        mx.save_safetensors(plddt_safetensors, mlx_state_dict)
+        del plddt_checkpoint, mlx_state_dict
+        import gc
+        gc.collect()
 
     plddt_module_path = get_config_path("configs/model/architecture/plddt_module.yaml")
-    plddt_checkpoint = torch.load(plddt_ckpt_path, map_location="cpu", weights_only=False)
-
-    if args.backend == "torch":
-        plddt_config = omegaconf.OmegaConf.load(plddt_module_path)
-        plddt_out_module = hydra.utils.instantiate(plddt_config)
-        plddt_out_module.load_state_dict(plddt_checkpoint, strict=True)
-        plddt_out_module = plddt_out_module.to(device)
-    elif args.backend == "mlx":
-        # replace torch implementations with mlx
-        with open(plddt_module_path, "r") as f:
-            yaml_str = f.read()
-        yaml_str = yaml_str.replace('torch', 'mlx')
-
-        plddt_config = omegaconf.OmegaConf.create(yaml_str)
-        plddt_out_module = hydra.utils.instantiate(plddt_config)
-
-        mlx_state_dict = {k: mx.array(v) for k, v in starmap(map_plddt_torch_to_mlx, plddt_checkpoint.items()) if k is not None}
-        plddt_out_module.update(tree_unflatten(list(mlx_state_dict.items())))
-
+    with open(plddt_module_path, "r") as f:
+        yaml_str = f.read()
+    yaml_str = yaml_str.replace('torch', 'mlx')
+
+    plddt_config = omegaconf.OmegaConf.create(yaml_str)
+    plddt_out_module = hydra.utils.instantiate(plddt_config)
+    
+    weights = mx.load(plddt_safetensors)
+    plddt_out_module.update(tree_unflatten(list(weights.items())))
+    del weights
+    mx.clear_cache()
+
+    import gc
+    gc.collect()
     plddt_out_module.eval()
-    print(f"pLDDT output module loaded with {args.backend} backend.")
-
-    plddt_latent_ckpt_path = os.path.join(args.ckpt_dir, "simplefold_1.6B.ckpt")
-    if not os.path.exists(plddt_latent_ckpt_path):
-        os.makedirs(args.ckpt_dir, exist_ok=True)
-        os.system(f"curl -L {ckpt_url_dict['simplefold_1.6B']} -o {plddt_latent_ckpt_path}")
+    print("pLDDT output module loaded natively on MLX.")
+
+    # Latent module
+    plddt_latent_safetensors = os.path.join(ckpt_dir, "simplefold_1.6B.safetensors")
+    if not os.path.exists(plddt_latent_safetensors):
+        plddt_latent_ckpt_path = os.path.join(ckpt_dir, "simplefold_1.6B.ckpt")
+        if not os.path.exists(plddt_latent_ckpt_path):
+            os.makedirs(ckpt_dir, exist_ok=True)
+            print("Downloading simplefold_1.6B checkpoint for pLDDT...")
+            os.system(f"curl -L {ckpt_url_dict['simplefold_1.6B']} -o {plddt_latent_ckpt_path}")
+            
+        print(f"Converting {plddt_latent_ckpt_path} to MLX safetensors format...")
+        plddt_latent_checkpoint = torch.load(plddt_latent_ckpt_path, map_location="cpu", weights_only=False)
+        mlx_state_dict = {}
+        for k, v in plddt_latent_checkpoint.items():
+            k_mlx, v_np = map_torch_to_mlx(k, v)
+            if k_mlx is not None:
+                if v_np.dtype in (np.float32, np.float16):
+                    mlx_state_dict[k_mlx] = mx.array(v_np).astype(mx.float16)
+                else:
+                    mlx_state_dict[k_mlx] = mx.array(v_np)
+        mx.save_safetensors(plddt_latent_safetensors, mlx_state_dict)
+        del plddt_latent_checkpoint, mlx_state_dict
+        import gc
+        gc.collect()
 
     plddt_latent_config_path = get_config_path("configs/model/architecture/foldingdit_1.6B.yaml")
-    plddt_latent_checkpoint = torch.load(plddt_latent_ckpt_path, map_location="cpu", weights_only=False)
-
-    if args.backend == "torch":
-        plddt_latent_config = omegaconf.OmegaConf.load(plddt_latent_config_path)
-        plddt_latent_module = hydra.utils.instantiate(plddt_latent_config)
-        plddt_latent_module.load_state_dict(plddt_latent_checkpoint, strict=True)
-        plddt_latent_module = plddt_latent_module.to(device)
-    elif args.backend == "mlx":
-        # replace torch implementations with mlx
-        with open(plddt_latent_config_path, "r") as f:
-            yaml_str = f.read()
-        yaml_str = yaml_str.replace('torch', 'mlx')
-
-        plddt_latent_config = omegaconf.OmegaConf.create(yaml_str)
-        plddt_latent_module = hydra.utils.instantiate(plddt_latent_config)
-        mlx_state_dict = {k: mx.array(v) for k, v in starmap(map_torch_to_mlx, plddt_latent_checkpoint.items()) if k is not None}
-        plddt_latent_module.update(tree_unflatten(list(mlx_state_dict.items())))
-
+    with open(plddt_latent_config_path, "r") as f:
+        yaml_str = f.read()
+    yaml_str = yaml_str.replace('torch', 'mlx')
+
+    plddt_latent_config = omegaconf.OmegaConf.create(yaml_str)
+    plddt_latent_module = hydra.utils.instantiate(plddt_latent_config)
+    
+    weights = mx.load(plddt_latent_safetensors)
+    plddt_latent_module.update(tree_unflatten(list(weights.items())))
+    del weights
+    mx.clear_cache()
+
+    import gc
+    gc.collect()
     plddt_latent_module.eval()
-    print(f"pLDDT latent module loaded with {args.backend} backend.")
+    print("pLDDT latent module loaded natively on MLX.")
 
     return plddt_latent_module, plddt_out_module
 
@@ -310,36 +392,89 @@ def initialize_plddt_module(args, device):
 def initialize_esm_model(args, device, quantize_esm=True):
     """
     Initialize ESM-3B protein language model.
     """
-    # load ESM2 model
-    esm_model, esm_dict = esm_registry["esm2_3B"]()
+    import gc
+    
+    ckpt_dir = Path(args.ckpt_dir)
+    ckpt_dir.mkdir(parents=True, exist_ok=True)
+    quantized_weights_path = ckpt_dir / "esm2_t36_3B_UR50D_quantized_8bit.safetensors"
+    
+    # Load small 8M model to extract alphabet dictionary without loading 3B model weights
+    _, esm_dict = esm_registry["esm2_8M"]()
     af2_to_esm = _af2_to_esm(esm_dict)
 
-    if args.backend == 'torch':
-        esm_model = esm_model.to(device)
-        af2_to_esm = af2_to_esm.to(device)
-    elif args.backend == 'mlx':
-        esm_model_mlx = ESM2MLX(num_layers=36, embed_dim=2560, attention_heads=40)
-        esm_state_dict_torch = esm_model.cpu().state_dict()
+    # Initialize the MLX architecture directly
+    esm_model_mlx = ESM2MLX(num_layers=36, embed_dim=2560, attention_heads=40)
+
+    if quantize_esm and quantized_weights_path.exists():
+        print(f"[MLX Memory Optimizer] Loading pre-quantized 8-bit ESM-3B weights from: {quantized_weights_path.name}")
+        # Quantize the empty model structure first
+        esm_model_mlx = _quantize_esm_int8(esm_model_mlx)
+        
+        # Load quantized weights directly into the model
+        weights = mx.load(str(quantized_weights_path))
+        esm_model_mlx.update(tree_unflatten(list(weights.items())))
+        del weights
+        gc.collect()
+        esm_model = esm_model_mlx
+    else:
+        # Find PyTorch Hub checkpoint path
+        hub_dir = torch.hub.get_dir()
+        checkpoint_path = Path(hub_dir) / "checkpoints" / "esm2_t36_3B_UR50D.pt"
+        if not checkpoint_path.exists():
+            print("ESM-3B checkpoint not found locally, downloading via torch.hub...")
+            # This triggers download and saves it, but we don't keep the PyTorch model
+            temp_model, _ = esm_registry["esm2_3B"]()
+            del temp_model
+            gc.collect()
+
+        # Load weights directly via memory-mapping (mmap=True) to avoid loading the full 11.6GB into CPU memory
+        print(f"[MLX Memory Optimizer] Memory-mapping ESM-3B checkpoint: {checkpoint_path.name}")
+        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True, weights_only=False)
+        state_dict = checkpoint["model"]
+
+        # Map state dict keys to MLX arrays on-demand and delete the mapped tensors from CPU memory
+        mlx_state_dict = {}
+        for k in list(state_dict.keys()):
+            v = state_dict[k]
+            
+            # Strip PyTorch Hub model checkpoint prefixes to match state_dict format
+            k_mapped = k
+            if k_mapped.startswith("encoder.sentence_encoder."):
+                k_mapped = k_mapped.replace("encoder.sentence_encoder.", "")
+            elif k_mapped.startswith("encoder."):
+                k_mapped = k_mapped.replace("encoder.", "")
+            
+            k_mlx, v_mlx = map_torch_to_mlx(k_mapped, v)
+            if k_mlx is not None:
+                mlx_state_dict[k_mlx] = mx.array(v_mlx)
+            # Free CPU memory reference immediately
+            del state_dict[k]
+
+        del checkpoint, state_dict
+        esm_model_mlx.update(tree_unflatten(list(mlx_state_dict.items())))
+        del mlx_state_dict
+        gc.collect()
 
-        esm_state_dict_torch = {k: mx.array(v) for k, v in starmap(map_torch_to_mlx, esm_state_dict_torch.items()) if k is not None}
-        esm_model_mlx.update(tree_unflatten(list(esm_state_dict_torch.items())))
         esm_model = esm_model_mlx
 
         if quantize_esm:
             # Apply INT8 quantization to ESM-3B for 3.5x memory reduction (11.4GB → 3.2GB)
             esm_model = _quantize_esm_int8(esm_model)
+            
+            # Save the quantized weights for future runs
+            try:
+                print(f"[MLX Memory Optimizer] Saving 8-bit quantized weights to: {quantized_weights_path.absolute()}")
+                weights_to_save = dict(tree_flatten(esm_model.parameters()))
+                mx.save_safetensors(str(quantized_weights_path), weights_to_save)
+                del weights_to_save
+                gc.collect()
+            except Exception as e:
+                print(f"Warning: Could not save quantized weights: {e}")
         else:
             print("  ESM-3B running at full precision (~11.4GB)")
 
-    print(f"pLM ESM-3B loaded with {args.backend} backend.")
-
+    print("pLM ESM-3B loaded with mlx backend.")
     esm_model.eval()
     return esm_model, esm_dict, af2_to_esm
 
@@ -310,12 +392,12 @@ def initialize_others(args, device):
     tokenizer = BoltzTokenizer()
     featurizer = BoltzFeaturizer()
     processor = ProteinDataProcessor(
-        device=device,
+        device="cpu",
         scale=16.0, 
         ref_scale=5.0, 
         multiplicity=1,
         inference_multiplicity=args.nsample_per_protein,
-        backend=args.backend,
+        backend="mlx",
     )
 
     # define flow process and sampler
@@ -323,42 +405,23 @@ def initialize_others(args, device):
 
     teacache_threshold = getattr(args, 'teacache', 0.0)
 
-    if args.backend == "torch":
-        if teacache_threshold > 0:
-            config = TeaCacheConfigTorch(threshold=teacache_threshold)
-            sampler = TeaCacheSamplerTorch(
-                num_timesteps=args.num_steps,
-                t_start=1e-4,
-                tau=args.tau,
-                config=config,
-            )
-            print(f"TeaCache enabled (threshold={teacache_threshold})")
-        else:
-            sampler = EMSampler(
-                num_timesteps=args.num_steps,
-                t_start=1e-4,
-                tau=args.tau,
-                log_timesteps=True,
-                w_cutoff=0.99,
-            )
-    elif args.backend == "mlx":
-        if teacache_threshold > 0:
-            config = TeaCacheConfig(threshold=teacache_threshold)
-            sampler = TeaCacheSampler(
-                num_timesteps=args.num_steps,
-                t_start=1e-4,
-                tau=args.tau,
-                config=config,
-            )
-            print(f"TeaCache enabled (threshold={teacache_threshold})")
-        else:
-            sampler = EMSamplerMLX(
-                num_timesteps=args.num_steps,
-                t_start=1e-4,
-                tau=args.tau,
-                log_timesteps=True,
-                w_cutoff=0.99,
-            )
+    if teacache_threshold > 0:
+        config = TeaCacheConfig(threshold=teacache_threshold)
+        sampler = TeaCacheSampler(
+            num_timesteps=args.num_steps,
+            t_start=1e-4,
+            tau=args.tau,
+            config=config,
+        )
+        print(f"TeaCache enabled (threshold={teacache_threshold})")
+    else:
+        sampler = EMSamplerMLX(
+            num_timesteps=args.num_steps,
+            t_start=1e-4,
+            tau=args.tau,
+            log_timesteps=True,
+            w_cutoff=0.99,
+        )
     return tokenizer, featurizer, processor, flow, sampler
 
 
@@ -370,33 +433,16 @@ def generate_structure(
     args, batch, sampler, flow, processor,
     model, plddt_latent_module, plddt_out_module, device
 ):
-    # run inference for target protein
-    if args.backend == "torch":
-        noise = torch.randn_like(batch['coords']).to(device)
-    elif args.backend == "mlx":
-        noise = mx.random.normal(batch['coords'].shape)
+    noise = mx.random.normal(batch['coords'].shape)
     out_dict = sampler.sample(model, flow, noise, batch)
 
     if args.plddt:
-        if args.backend == "torch":
-            t = torch.ones(batch['coords'].shape[0], device=device)
-            # use unscaled coords to extract latent for pLDDT prediction
-            out_feat = plddt_latent_module(
-                out_dict["denoised_coords"].detach(), t, batch)
-            plddt_out_dict = plddt_out_module(
-                out_feat["latent"].detach(),
-                batch,
-            )
-        elif args.backend == "mlx":
-            t = mx.ones(batch['coords'].shape[0])
-            # use unscaled coords to extract latent for pLDDT prediction
-            out_feat = plddt_latent_module(
-                out_dict["denoised_coords"], t, batch)
-            plddt_out_dict = plddt_out_module(
-                out_feat["latent"],
-                batch,
-            )
+        t = mx.ones(batch['coords'].shape[0])
+        # use unscaled coords to extract latent for pLDDT prediction
+        out_feat = plddt_latent_module(
+            out_dict["denoised_coords"], t, batch)
+        plddt_out_dict = plddt_out_module(
+            out_feat["latent"],
+            batch,
+        )
         # scale pLDDT to [0, 100]
         plddts = plddt_out_dict["plddt"] * 100.0
     else:
@@ -404,13 +450,7 @@ def generate_structure(
         plddts = None
 
     out_dict = processor.postprocess(out_dict, batch)
-    # sampled_coord = out_dict['denoised_coords'].detach()
-    if args.backend == "torch":
-        sampled_coord = out_dict['denoised_coords'].detach()
-    else:
-        sampled_coord = out_dict['denoised_coords']
-
+    sampled_coord = out_dict['denoised_coords']
     pad_mask = batch['atom_pad_mask']
     return sampled_coord, pad_mask, plddts
 
@@ -419,16 +459,14 @@ def predict_structures_from_fastas(args):
     # set random seed for reproducibility
     pl.seed_everything(args.seed, workers=True)
 
-    if args.backend == "mlx" and not MLX_AVAILABLE:
-        args.backend = "torch"
-        print("MLX not available, switch to torch backend.")
+    # MLX backend ONLY
+    device = "cpu"
 
-    # initialize models
-    model, device = initialize_folding_model(args)
-    plddt_latent_module, plddt_out_module = initialize_plddt_module(args, device)
+    # -------------------------------------------------------------
+    # STEP 1: Feature Extraction using ESM
+    # -------------------------------------------------------------
+    print("\n--- Step 1: Loading ESM model for Feature Extraction ---")
     esm_model, esm_dict, af2_to_esm = initialize_esm_model(args, device)
-
-    # initialize other components
     tokenizer, featurizer, processor, flow, sampler = initialize_others(args, device)
 
     # process fasta files to input format
@@ -442,28 +480,107 @@ def predict_structures_from_fastas(args):
         ccd_path=cache / "ccd.pkl",
     )
 
+    # Pre-extract ESM features for all structures
+    preprocessed_jobs = []
     for struct_file in output_dir.glob("structures/*.npz"):
         record_file = output_dir / "records" / f"{struct_file.stem}.json"
 
-        # prepare the target protein data for inference
+        print(f"Extracting ESM features for: {struct_file.name}")
         batch, structure, record = process_one_inference_structure(
             struct_file, record_file,
             tokenizer, featurizer, processor,
             esm_model, esm_dict, af2_to_esm,
         )
+        preprocessed_jobs.append((batch, structure, record))
+
+    # -------------------------------------------------------------
+    # RELEASE RAM: Unload ESM immediately
+    # -------------------------------------------------------------
+    print("\n--- Releasing ESM model RAM/GPU memory ---")
+    del esm_model, af2_to_esm
+    import gc
+    gc.collect()
+    
+    import mlx.core as mx
+    if hasattr(mx, "clear_cache"):
+        mx.clear_cache()
+    elif hasattr(mx.metal, "clear_cache"):
+        mx.metal.clear_cache()
+
+    # -------------------------------------------------------------
+    # STEP 2a: Coordinate Generation using Folding Model
+    # -------------------------------------------------------------
+    print("\n--- Step 2: Loading Folding Model ---")
+    model, device = initialize_folding_model(args)
 
-        sampled_coord, pad_mask, plddts = generate_structure(
-            args, batch, sampler, flow, processor,
-            model, plddt_latent_module, plddt_out_module, device
-        )
-
+    raw_coords_jobs = []
+    for batch, structure, record in preprocessed_jobs:
+        print(f"Generating structure coordinates for: {record.id}")
+        
+        noise = mx.random.normal(batch['coords'].shape)
+        out_dict = sampler.sample(model, flow, noise, batch)
+        denoised_coords = out_dict["denoised_coords"]
+            
+        # Run coordinate post-processing
+        post_out = processor.postprocess(out_dict, batch)
+        sampled_coord = post_out['denoised_coords']
+            
+        pad_mask = batch['atom_pad_mask']
+        raw_coords_jobs.append((batch, structure, record, denoised_coords, sampled_coord, pad_mask))
+
+    # Free folding model and clear device memory caches immediately
+    print("\n--- Releasing Folding Model RAM/GPU memory ---")
+    del model
+    gc.collect()
+    
+    import mlx.core as mx
+    if hasattr(mx, "clear_cache"):
+        mx.clear_cache()
+    elif hasattr(mx.metal, "clear_cache"):
+        mx.metal.clear_cache()
+
+    # -------------------------------------------------------------
+    # STEP 2b: pLDDT Confidence Score Prediction
+    # -------------------------------------------------------------
+    plddt_jobs = {}
+    if args.plddt:
+        print("\n--- Step 3: Loading pLDDT Model ---")
+        plddt_latent_module, plddt_out_module = initialize_plddt_module(args, device)
+        
+        for batch, structure, record, denoised_coords, _, _ in raw_coords_jobs:
+            print(f"Predicting confidence (pLDDT) for: {record.id}")
+            t = mx.ones(batch['coords'].shape[0])
+            out_feat = plddt_latent_module(denoised_coords, t, batch)
+            plddt_out_dict = plddt_out_module(out_feat["latent"], batch)
+                
+            plddts = plddt_out_dict["plddt"] * 100.0
+            plddt_jobs[record.id] = plddts
+
+        # Free pLDDT models and clear device memory caches immediately
+        print("\n--- Releasing pLDDT Model RAM/GPU memory ---")
+        del plddt_latent_module, plddt_out_module
+        gc.collect()
+        
+        import mlx.core as mx
+        if hasattr(mx, "clear_cache"):
+            mx.clear_cache()
+        elif hasattr(mx.metal, "clear_cache"):
+            mx.metal.clear_cache()
+
+    # -------------------------------------------------------------
+    # STEP 2c: Post-Processing & Output Generation
+    # -------------------------------------------------------------
+    for batch, structure, record, _, sampled_coord, pad_mask in raw_coords_jobs:
+        print(f"Saving final structure for: {record.id}")
+        plddts = plddt_jobs.get(record.id, None)
+        
         for i in range(args.nsample_per_protein):
             sampled_coord_i = sampled_coord[i]
             pad_mask_i = pad_mask[i]
 
             # save the generated structure
             structure_save = process_structure(
-                deepcopy(structure), sampled_coord_i, pad_mask_i, record, backend=args.backend
+                deepcopy(structure), sampled_coord_i, pad_mask_i, record, backend="mlx"
             )
             outname = f"{record.id}_sampled_{i}"
             save_structure(
@@ -471,3 +588,4 @@ def predict_structures_from_fastas(args):
                 output_format=args.output_format,
                 plddts=plddts[i] if plddts is not None else None
             )
```

---

## File: src/simplefold/model/simplefold.py

```diff
diff --git a/src/simplefold/model/simplefold.py b/src/simplefold/model/simplefold.py
index 49ce757..4833406 100644
--- a/src/simplefold/model/simplefold.py
+++ b/src/simplefold/model/simplefold.py
@@ -402,7 +402,8 @@ class SimpleFold(pl.LightningModule):
         align_weights = y_t.new_ones(y_t.shape[:2])
 
         if self.use_rigid_align:
-            with torch.no_grad(), torch.autocast("cuda", enabled=False):
+            device_type = self.device.type if self.device.type in ["cuda", "cpu", "mps"] else "cpu"
+            with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=False):
                 v_t = out_dict['predict_velocity'].detach().float()
                 denoised_coords = y_t + v_t * (1.0 - t[:, None, None])
                 coords = batch["coords"].detach().float()
@@ -513,7 +514,8 @@ class SimpleFold(pl.LightningModule):
 
     @torch.no_grad()
     def predict_step(self, batch, batch_idx):
-        with torch.autocast(device_type='cuda', dtype=torch.float32):
+        device_type = self.device.type if self.device.type in ["cuda", "cpu", "mps"] else "cpu"
+        with torch.amp.autocast(device_type=device_type, dtype=torch.float32):
             batch = self.processor.preprocess_inference(
                 batch,
                 esm_model=self.esm_model,
@@ -671,7 +673,7 @@ class SimpleFold(pl.LightningModule):
                 self.af2_to_esm,
             )
             y = batch["coords"]
-            t = torch.zeros((y.shape[0])).cuda()
+            t = torch.zeros((y.shape[0]), device=self.device)
 
     def on_train_batch_end(self, outputs, batch, batch_idx):
         optimizer = self.optimizers()
```

---

## File: src/simplefold/processor/protein_processor.py

```diff
diff --git a/src/simplefold/processor/protein_processor.py b/src/simplefold/processor/protein_processor.py
index 2de580a..2b7ec0f 100644
--- a/src/simplefold/processor/protein_processor.py
+++ b/src/simplefold/processor/protein_processor.py
@@ -105,7 +105,6 @@ class ProteinDataProcessor:
 
     def batch_to_device(self, batch, multiplicity=1):
         for k, v in batch.items():
-            # if isinstance(v, torch.Tensor) and k in key2cuda:
             if isinstance(v, torch.Tensor):
                 if multiplicity > 1:
                     v = v.repeat_interleave(multiplicity, dim=0)
```

---

## File: src/simplefold/train.py

```diff
diff --git a/src/simplefold/train.py b/src/simplefold/train.py
index 782399e..b600114 100644
--- a/src/simplefold/train.py
+++ b/src/simplefold/train.py
@@ -25,8 +25,6 @@ from utils.pylogger import RankedLogger
 log = RankedLogger(__name__, rank_zero_only=True)
 
 torch.set_float32_matmul_precision("medium")
-torch.backends.cuda.matmul.allow_tf32 = True # This flag defaults to False
-torch.backends.cudnn.allow_tf32 = True       # This flag defaults to True
 
 
 @task_wrapper
```

---

## File: src/simplefold/utils/boltz_utils.py

```diff
diff --git a/src/simplefold/utils/boltz_utils.py b/src/simplefold/utils/boltz_utils.py
index 53d6628..89e28c2 100644
--- a/src/simplefold/utils/boltz_utils.py
+++ b/src/simplefold/utils/boltz_utils.py
@@ -261,9 +261,7 @@ def weighted_rigid_align(
     # Compute the SVD of the covariance matrix, required float32 for svd and determinant
     original_dtype = cov_matrix.dtype
     cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
-    U, S, V = torch.linalg.svd(
-        cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None
-    )
+    U, S, V = torch.linalg.svd(cov_matrix_32)
     V = V.mH
 
     # Catch ambiguous rotation by checking the magnitude of singular values
```

---

## File: src/simplefold/wrapper.py

```diff
diff --git a/src/simplefold/wrapper.py b/src/simplefold/wrapper.py
index afca678..c76b2ff 100644
--- a/src/simplefold/wrapper.py
+++ b/src/simplefold/wrapper.py
@@ -7,12 +7,11 @@ import os
 import torch
 import hydra
 import omegaconf
+import numpy as np
 from pathlib import Path
 from itertools import starmap
 
 from model.flow import LinearPath
-from model.torch.sampler import EMSampler
-
 from processor.protein_processor import ProteinDataProcessor
 from utils.datamodule_utils import process_one_inference_structure
 from utils.esm_utils import _af2_to_esm, esm_registry
@@ -21,17 +20,13 @@ from utils.fasta_utils import process_fastas, download_fasta_utilities, resolve_
 from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
 from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
 
-try:
-    import mlx.core as mx
-    from mlx.utils import tree_unflatten, tree_flatten
-    from model.mlx.sampler import EMSampler as EMSamplerMLX
-    from model.mlx.teacache import TeaCacheSampler, TeaCacheConfig
-    from model.mlx.esm_network import ESM2 as ESM2MLX
-    from utils.mlx_utils import map_torch_to_mlx, map_plddt_torch_to_mlx
-    MLX_AVAILABLE = True
-except:
-    MLX_AVAILABLE = False
-    print("MLX not installed, skip importing MLX related packages.")
+import mlx.core as mx
+from mlx.utils import tree_unflatten, tree_flatten
+from model.mlx.sampler import EMSampler as EMSamplerMLX
+from model.mlx.teacache import TeaCacheSampler, TeaCacheConfig
+from model.mlx.esm_network import ESM2 as ESM2MLX
+from utils.mlx_utils import map_torch_to_mlx, map_plddt_torch_to_mlx
+MLX_AVAILABLE = True
 
 
 ckpt_url_dict = {
@@ -54,63 +49,67 @@ class ModelWrapper:
         simplefold_model,
         plddt=False,
         ckpt_dir="./artifacts",
-        backend="torch",
+        backend="mlx",
     ):
         self.simplefold_model = simplefold_model
         self.plddt = plddt
         self.ckpt_dir = Path(ckpt_dir)
-        self.backend = backend
-        if self.backend == "mlx" and not MLX_AVAILABLE:
-            self.backend = "torch"
-            print("MLX not installed, skip importing MLX related packages.")
+        self.backend = "mlx"
         self.folding_model = None
         self.plddt_out_module = None
         self.plddt_latent_module = None
-        self.device = self.get_device()
+        self.device = "cpu"
         self.ckpt_dir.mkdir(parents=True, exist_ok=True)
 
     def get_device(self):
-        if self.backend == "torch":
-            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
-        elif self.backend == "mlx":
-            device = "cpu"
-        return device
+        return "cpu"
 
     def from_pretrained_folding_model(self):
         # define folding model
         simplefold_model = self.simplefold_model
 
-        # create folding model
-        ckpt_path = os.path.join(self.ckpt_dir, f"{simplefold_model}.ckpt")
-        if not os.path.exists(ckpt_path):
-            os.system(f"curl -L -o {ckpt_path} {ckpt_url_dict[simplefold_model]}")
-
-        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
+        safetensors_path = os.path.join(self.ckpt_dir, f"{simplefold_model}.safetensors")
+        if not os.path.exists(safetensors_path):
+            # Convert on-demand if needed
+            ckpt_path = os.path.join(self.ckpt_dir, f"{simplefold_model}.ckpt")
+            if not os.path.exists(ckpt_path):
+                os.makedirs(self.ckpt_dir, exist_ok=True)
+                os.system(f"curl -L -o {ckpt_path} {ckpt_url_dict[simplefold_model]}")
+            
+            print(f"Converting {ckpt_path} to MLX safetensors format...")
+            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
+            mlx_state_dict = {}
+            for k, v in checkpoint.items():
+                k_mlx, v_np = map_torch_to_mlx(k, v)
+                if k_mlx is not None:
+                    if v_np.dtype in (np.float32, np.float16):
+                        mlx_state_dict[k_mlx] = mx.array(v_np).astype(mx.float16)
+                    else:
+                        mlx_state_dict[k_mlx] = mx.array(v_np)
+            mx.save_safetensors(safetensors_path, mlx_state_dict)
+            del checkpoint, mlx_state_dict
+            import gc
+            gc.collect()
 
-        # load model checkpoint
         cfg_path = os.path.join(
             "configs/model/architecture", f"foldingdit_{simplefold_model[11:]}.yaml"
         )
-        if self.backend == "torch":
-            model_config = omegaconf.OmegaConf.load(cfg_path)
-            model = hydra.utils.instantiate(model_config)
-            model.load_state_dict(checkpoint, strict=True)
-            model = model.to(self.device)
-        elif self.backend == "mlx":
-            # replace torch implementations with mlx
-            with open(cfg_path, "r") as f:
-                yaml_str = f.read()
-            yaml_str = yaml_str.replace("torch", "mlx")
-
-            model_config = omegaconf.OmegaConf.create(yaml_str)
-            model = hydra.utils.instantiate(model_config)
-            mlx_state_dict = {
-                k: mx.array(v)
-                for k, v in starmap(map_torch_to_mlx, checkpoint.items())
-                if k is not None
-            }
-            model.update(tree_unflatten(list(mlx_state_dict.items())))
-        print(f"Folding model {simplefold_model} loaded with {self.backend} backend.")
+        with open(cfg_path, "r") as f:
+            yaml_str = f.read()
+        yaml_str = yaml_str.replace("torch", "mlx")
+
+        model_config = omegaconf.OmegaConf.create(yaml_str)
+        model = hydra.utils.instantiate(model_config)
+        
+        # Natively load
+        weights = mx.load(safetensors_path)
+        model.update(tree_unflatten(list(weights.items())))
+        del weights
+        mx.clear_cache()
+            
+        import gc
+        gc.collect()
+        print(f"Folding model {simplefold_model} loaded natively on MLX.")
 
         model.eval()
         return model
@@ -197,30 +196,107 @@ class ModelWrapper:
         self.prediction_dir = prediction_dir
 
         self.esm_model = None
         self.esm_dict = None
         self.af2_to_esm = None
 
         if not self.lazy:
             self.initialize_esm_model()
         self.initialize_others()
 
     def initialize_esm_model(self):
-        # load ESM2 model
-        esm_model, esm_dict = esm_registry["esm2_3B"]()
-        af2_to_esm = _af2_to_esm(esm_dict)
-
-        if self.backend == "torch":
-            esm_model = esm_model.to(self.device)
-            af2_to_esm = af2_to_esm.to(self.device)
-        elif self.backend == "mlx":
+        if self.backend == "mlx":
             import gc
             from inference import _quantize_esm_int8
             
             self.ckpt_dir.mkdir(parents=True, exist_ok=True)
             quantized_weights_path = self.ckpt_dir / "esm2_t36_3B_UR50D_quantized_8bit.safetensors"
             
             # Optimized load for MLX: load small 8M model to extract alphabet dictionary without loading 3B model weights
             _, esm_dict = esm_registry["esm2_8M"]()
             af2_to_esm = _af2_to_esm(esm_dict)
 
             # Initialize the MLX architecture directly
             esm_model_mlx = ESM2MLX(num_layers=36, embed_dim=2560, attention_heads=40)
 
             if quantized_weights_path.exists():
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
 
                 # Quantize ESM-3B for MLX in wrapper too
                 esm_model = _quantize_esm_int8(esm_model_mlx)
                 
                 # Save the quantized weights for future runs
                 try:
                     from mlx.utils import tree_flatten
                     print(f"[MLX Memory Optimizer] Saving 8-bit quantized weights to: {quantized_weights_path.absolute()}")
                     weights_to_save = dict(tree_flatten(esm_model.parameters()))
                     mx.save_safetensors(str(quantized_weights_path), weights_to_save)
                     del weights_to_save
                     gc.collect()
                 except Exception as e:
                     print(f"Warning: Could not save quantized weights: {e}")
 
+        print("pLM ESM-3B loaded with mlx backend.")
         self.esm_model = esm_model.eval()
         self.esm_dict = esm_dict
         self.af2_to_esm = af2_to_esm
@@ -291,46 +338,58 @@ class InferenceWrapper:
         self.tokenizer = BoltzTokenizer()
         self.featurizer = BoltzFeaturizer()
         self.processor = ProteinDataProcessor(
-            device=self.device,
+            device="cpu",
             scale=16.0,
             ref_scale=5.0,
             multiplicity=1,
             inference_multiplicity=self.nsample_per_protein,
-            backend=self.backend,
+            backend="mlx",
         )
 
         # define flow process and sampler
         self.flow = LinearPath()
 
-        if self.backend == "torch":
-            self.sampler = EMSampler(
+        if self.teacache:
+            # Use TeaCache-accelerated sampler
+            config = TeaCacheConfig(threshold=self.teacache_threshold)
+            self.sampler = TeaCacheSampler(
+                num_timesteps=self.num_steps,
+                t_start=1e-4,
+                tau=self.tau,
+                config=config,
+            )
+            print(f"TeaCache enabled (threshold={self.teacache_threshold})")
+        else:
+            self.sampler = EMSamplerMLX(
                 num_timesteps=self.num_steps,
                 t_start=1e-4,
                 tau=self.tau,
                 log_timesteps=True,
                 w_cutoff=0.99,
             )
-        elif self.backend == "mlx":
-            if self.teacache:
-                # Use TeaCache-accelerated sampler
-                config = TeaCacheConfig(threshold=self.teacache_threshold)
-                self.sampler = TeaCacheSampler(
-                    num_timesteps=self.num_steps,
-                    t_start=1e-4,
-                    tau=self.tau,
-                    config=config,
-                )
-                print(f"TeaCache enabled (threshold={self.teacache_threshold})")
-            else:
-                self.sampler = EMSamplerMLX(
-                    num_timesteps=self.num_steps,
-                    t_start=1e-4,
-                    tau=self.tau,
-                    log_timesteps=True,
-                    w_cutoff=0.99,
-                )
-
-    def process_input(self, aa_seq):
+
+    def release_esm_model(self):
+        """Release ESM model from memory/GPU to free up RAM."""
+        if hasattr(self, "esm_model") and self.esm_model is not None:
+            print("[InferenceWrapper] Releasing ESM model from RAM/GPU...")
+            del self.esm_model
+            self.esm_model = None
+        if hasattr(self, "af2_to_esm") and self.af2_to_esm is not None:
+            del self.af2_to_esm
+            self.af2_to_esm = None
+        import gc
+        gc.collect()
+        import mlx.core as mx
+        if hasattr(mx, "clear_cache"):
+            mx.clear_cache()
+        elif hasattr(mx.metal, "clear_cache"):
+            mx.metal.clear_cache()
+
+    def process_input(self, aa_seq, release_esm=False):
+        if self.esm_model is None:
+            print("[InferenceWrapper] Loading ESM model for feature extraction...")
+            self.initialize_esm_model()
+
         # download shared utilities into the cache dir if missing
         download_fasta_utilities(self.cache)
         # save the input sequence to a fasta file in the per-run work dir
@@ -356,14 +415,14 @@ class InferenceWrapper:
             self.esm_dict,
             self.af2_to_esm,
         )
+
+        if release_esm:
+            self.release_esm_model()
+
         return batch, structure, record
 
     def run_inference(self, batch, model, plddt_model, device):
-        # run inference for target protein
-        if self.backend == "torch":
-            noise = torch.randn_like(batch["coords"]).to(device)
-        elif self.backend == "mlx":
-            noise = mx.random.normal(batch["coords"].shape)
+        noise = mx.random.normal(batch["coords"].shape)
         out_dict = self.sampler.sample(model, self.flow, noise, batch)
 
         plddt_out_module = plddt_model["plddt_out_module"]
@@ -372,33 +431,16 @@ class InferenceWrapper:
         if plddt_latent_module is None or plddt_out_module is None:
             plddts = None
         else:
-            if self.backend == "torch":
-                t = torch.ones(batch["coords"].shape[0], device=device)
-                # use unscaled coords to extract latent for pLDDT prediction
-                out_feat = plddt_latent_module(
-                    out_dict["denoised_coords"].detach(), t, batch
-                )
-                plddt_out_dict = plddt_out_module(
-                    out_feat["latent"].detach(),
-                    batch,
-                )
-            elif self.backend == "mlx":
-                t = mx.ones(batch["coords"].shape[0])
-                # use unscaled coords to extract latent for pLDDT prediction
-                out_feat = plddt_latent_module(out_dict["denoised_coords"], t, batch)
-                plddt_out_dict = plddt_out_module(
-                    out_feat["latent"],
-                    batch,
-                )
-            # scale pLDDT to [0, 100]
+            t = mx.ones(batch["coords"].shape[0])
+            out_feat = plddt_latent_module(out_dict["denoised_coords"], t, batch)
+            plddt_out_dict = plddt_out_module(
+                out_feat["latent"],
+                batch,
+            )
             plddts = plddt_out_dict["plddt"] * 100.0
 
         out_dict = self.processor.postprocess(out_dict, batch)
-        # sampled_coord = out_dict['denoised_coords'].detach()
-        if self.backend == "torch":
-            sampled_coord = out_dict["denoised_coords"].detach()
-        else:
-            sampled_coord = out_dict["denoised_coords"]
+        sampled_coord = out_dict["denoised_coords"]
 
         return {
             "sampled_coord": sampled_coord,
@@ -419,7 +461,7 @@ class InferenceWrapper:
             out_name_i = f"{out_name}_sampled_{i}"
             # save the generated structure
             structure_save = process_structure(
-                structure, sampled_coord_i, pad_mask_i, record, backend=self.backend
+                structure, sampled_coord_i, pad_mask_i, record, backend="mlx"
             )
             save_structure(
                 structure_save,
```

---

## File: webui.py

```diff
diff --git a/webui.py b/webui.py
index c9c841d..5022b07 100644
--- a/webui.py
+++ b/webui.py
@@ -31,6 +31,8 @@ sys.path.append(str(Path(__file__).resolve().parent / "src" / "simplefold"))
 
 import numpy as np
 import torch
+import mlx.core as mx
+DEFAULT_BACKEND = "mlx"
 import streamlit as st
 import streamlit.components.v1 as components
 
@@ -40,6 +42,9 @@ from simplefold.inference import predict_structures_from_fastas
 logging.basicConfig(level=logging.INFO)
 logger = logging.getLogger(__name__)
 
+# Global lock to prevent concurrent folding runs and optimize RAM
+FOLDING_LOCK = threading.Lock()
+
 # Page config
 st.set_page_config(
     page_title="SF-T 0.1",
@@ -49,11 +54,7 @@ st.set_page_config(
 
 def get_device() -> str:
     """Determine the best available compute device."""
-    if torch.cuda.is_available():
-        return "cuda"
-    elif torch.backends.mps.is_available():
-        return "mps"
-    return "cpu"
+    return "Apple Silicon (Metal)"
 
 def validate_protein_sequence(seq: str) -> tuple[bool, str]:
     """
@@ -111,63 +112,69 @@ def fold_sequence(
         List of PDB format strings, one per ensemble member.
     """
     # Create temporary directory for this folding job
-    with tempfile.TemporaryDirectory() as tmpdir:
-        tmpdir = Path(tmpdir)
-        output_dir = tmpdir / "output"
-        output_dir.mkdir(parents=True, exist_ok=True)
-
-        # Write sequence to FASTA file
-        fasta_path = tmpdir / "input.fasta"
-        with open(fasta_path, 'w') as f:
-            f.write(f">query|protein\n{sequence}\n")
-
-        # Set up args for predict_structures_from_fastas
-        args = Namespace(
-            simplefold_model=get_model_name(model_size),
-            ckpt_dir=str(ARTIFACTS_DIR),  # Use local artifacts for model checkpoints
-            cache_dir=str(CACHE_DIR),
-            output_dir=str(output_dir),
-            num_steps=500,
-            tau=0.1,
-            no_log_timesteps=False,
-            fasta_path=str(fasta_path),
-            nsample_per_protein=ensemble_size,
-            plddt=False,
-            output_format="pdb",
-            backend="torch",
-            teacache=threshold if use_teacache else 0.0,
-            seed=42,
-        )
+    info_placeholder = st.empty()
+    if FOLDING_LOCK.locked():
+        info_placeholder.info("⏳ Another folding job is running. Queueing your request to optimize RAM usage...")
+
+    with FOLDING_LOCK:
+        info_placeholder.empty()
+        with tempfile.TemporaryDirectory() as tmpdir:
+            tmpdir = Path(tmpdir)
+            output_dir = tmpdir / "output"
+            output_dir.mkdir(parents=True, exist_ok=True)
+
+            # Write sequence to FASTA file
+            fasta_path = tmpdir / "input.fasta"
+            with open(fasta_path, 'w') as f:
+                f.write(f">query|protein\n{sequence}\n")
+
+            # Set up args for predict_structures_from_fastas
+            args = Namespace(
+                simplefold_model=get_model_name(model_size),
+                ckpt_dir=str(ARTIFACTS_DIR),  # Use local artifacts for model checkpoints
+                cache_dir=str(CACHE_DIR),
+                output_dir=str(output_dir),
+                num_steps=500,
+                tau=0.1,
+                no_log_timesteps=False,
+                fasta_path=str(fasta_path),
+                nsample_per_protein=ensemble_size,
+                plddt=False,
+                output_format="pdb",
+                backend=DEFAULT_BACKEND,
+                teacache=threshold if use_teacache else 0.0,
+                seed=42,
+            )
 
-        # Run folding
-        if progress_callback:
-            progress_callback(0, ensemble_size)
+            # Run folding
+            if progress_callback:
+                progress_callback(0, ensemble_size)
 
-        predict_structures_from_fastas(args)
+            predict_structures_from_fastas(args)
 
-        # Collect output PDB files
-        pdb_strings = []
-        prediction_dir = output_dir / f"predictions_{args.simplefold_model}"
+            # Collect output PDB files
+            pdb_strings = []
+            prediction_dir = output_dir / f"predictions_{args.simplefold_model}"
 
-        # Find all PDB files
-        pdb_files = sorted(prediction_dir.glob("**/*.pdb"))
-        logger.info(f"Found {len(pdb_files)} PDB files in {prediction_dir}")
+            # Find all PDB files
+            pdb_files = sorted(prediction_dir.glob("**/*.pdb"))
+            logger.info(f"Found {len(pdb_files)} PDB files in {prediction_dir}")
 
-        if not pdb_files:
-            # Check what's in the output directory
-            all_files = list(output_dir.rglob("*"))
-            logger.warning(f"No PDB files found. Output dir contents: {[str(f) for f in all_files[:20]]}")
+            if not pdb_files:
+                # Check what's in the output directory
+                all_files = list(output_dir.rglob("*"))
+                logger.warning(f"No PDB files found. Output dir contents: {[str(f) for f in all_files[:20]]}")
 
-        for i, pdb_file in enumerate(pdb_files):
-            content = pdb_file.read_text()
-            if content.strip():
-                pdb_strings.append(content)
-            else:
-                logger.warning(f"Empty PDB file: {pdb_file}")
-            if progress_callback:
-                progress_callback(i + 1, len(pdb_files))
+            for i, pdb_file in enumerate(pdb_files):
+                content = pdb_file.read_text()
+                if content.strip():
+                    pdb_strings.append(content)
+                else:
+                    logger.warning(f"Empty PDB file: {pdb_file}")
+                if progress_callback:
+                    progress_callback(i + 1, len(pdb_files))
 
-        return pdb_strings
+            return pdb_strings
 
 
 def extract_ca_coords(pdb_string: str) -> np.ndarray:
@@ -465,102 +472,108 @@ def fold_batch(
     Returns: (results_dict, zip_bytes, elapsed_seconds)
         results_dict: {input_id: [pdb_string, ...]}
     """
-    with tempfile.TemporaryDirectory() as tmpdir:
-        tmpdir = Path(tmpdir)
-        input_dir = tmpdir / "fastas"
-        input_dir.mkdir(parents=True, exist_ok=True)
-        output_dir = tmpdir / "output"
-        output_dir.mkdir(parents=True, exist_ok=True)
-
-        # Write each sequence to its own FASTA so the inference loop treats them as
-        # independent targets (a multi-record FASTA would be parsed as one multimer).
-        stem_to_id = {}
-        seen_stems = set()
-        for sid, seq in sequences:
-            stem = sanitize_id(sid)
-            base, n = stem, 2
-            while stem in seen_stems:
-                stem = f"{base}_{n}"
-                n += 1
-            seen_stems.add(stem)
-            stem_to_id[stem] = sid
-            (input_dir / f"{stem}.fasta").write_text(f">{stem}|protein\n{seq}\n")
-
-        args = Namespace(
-            simplefold_model=get_model_name(model_size),
-            ckpt_dir=str(ARTIFACTS_DIR),
-            cache_dir=str(CACHE_DIR),
-            output_dir=str(output_dir),
-            num_steps=500,
-            tau=0.1,
-            no_log_timesteps=False,
-            fasta_path=str(input_dir),
-            nsample_per_protein=ensemble_size,
-            plddt=False,
-            output_format="pdb",
-            backend="torch",
-            teacache=threshold if use_teacache else 0.0,
-            seed=42,
-        )
-
-        prediction_dir = output_dir / f"predictions_{args.simplefold_model}"
-        total = len(sequences)
+    info_placeholder = st.empty()
+    if FOLDING_LOCK.locked():
+        info_placeholder.info("⏳ Another folding job is running. Queueing your request to optimize RAM usage...")
+
+    with FOLDING_LOCK:
+        info_placeholder.empty()
+        with tempfile.TemporaryDirectory() as tmpdir:
+            tmpdir = Path(tmpdir)
+            input_dir = tmpdir / "fastas"
+            input_dir.mkdir(parents=True, exist_ok=True)
+            output_dir = tmpdir / "output"
+            output_dir.mkdir(parents=True, exist_ok=True)
+
+            # Write each sequence to its own FASTA so the inference loop treats them as
+            # independent targets (a multi-record FASTA would be parsed as one multimer).
+            stem_to_id = {}
+            seen_stems = set()
+            for sid, seq in sequences:
+                stem = sanitize_id(sid)
+                base, n = stem, 2
+                while stem in seen_stems:
+                    stem = f"{base}_{n}"
+                    n += 1
+                seen_stems.add(stem)
+                stem_to_id[stem] = sid
+                (input_dir / f"{stem}.fasta").write_text(f">{stem}|protein\n{seq}\n")
+
+            args = Namespace(
+                simplefold_model=get_model_name(model_size),
+                ckpt_dir=str(ARTIFACTS_DIR),
+                cache_dir=str(CACHE_DIR),
+                output_dir=str(output_dir),
+                num_steps=500,
+                tau=0.1,
+                no_log_timesteps=False,
+                fasta_path=str(input_dir),
+                nsample_per_protein=ensemble_size,
+                plddt=False,
+                output_format="pdb",
+                backend=DEFAULT_BACKEND,
+                teacache=threshold if use_teacache else 0.0,
+                seed=42,
+            )
 
-        state = {"error": None, "done": False}
+            prediction_dir = output_dir / f"predictions_{args.simplefold_model}"
+            total = len(sequences)
 
-        def runner():
-            try:
-                predict_structures_from_fastas(args)
-            except Exception as e:
-                state["error"] = e
-            finally:
-                state["done"] = True
-
-        t = threading.Thread(target=runner, daemon=True)
-        start = time.time()
-        t.start()
-
-        last_n = -1
-        while not state["done"]:
-            if prediction_dir.exists():
-                completed = {
-                    p.name.rsplit("_sampled_", 1)[0]
-                    for p in prediction_dir.glob("*_sampled_0.pdb")
-                }
-                n = len(completed)
-            else:
-                n = 0
-            if n != last_n and progress_callback:
-                progress_callback(n, total)
-                last_n = n
-            time.sleep(0.5)
-
-        t.join(timeout=2.0)
-        if state["error"]:
-            raise state["error"]
-        if progress_callback:
-            progress_callback(total, total)
-
-        results = {}
-        for stem, sid in stem_to_id.items():
-            pdbs = []
-            for i in range(ensemble_size):
-                p = prediction_dir / f"{stem}_sampled_{i}.pdb"
-                if p.exists():
-                    content = p.read_text()
-                    if content.strip():
-                        pdbs.append(content)
-            results[sid] = pdbs
-
-        buf = io.BytesIO()
-        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
-            for sid, pdbs in results.items():
-                stem = sanitize_id(sid)
-                for i, pdb in enumerate(pdbs):
-                    zf.writestr(f"{stem}/sample_{i}.pdb", pdb)
+            state = {"error": None, "done": False}
 
-        elapsed = time.time() - start
-        return results, buf.getvalue(), elapsed
+            def runner():
+                try:
+                    predict_structures_from_fastas(args)
+                except Exception as e:
+                    state["error"] = e
+                finally:
+                    state["done"] = True
+
+            t = threading.Thread(target=runner, daemon=True)
+            start = time.time()
+            t.start()
+
+            last_n = -1
+            while not state["done"]:
+                if prediction_dir.exists():
+                    completed = {
+                        p.name.rsplit("_sampled_", 1)[0]
+                        for p in prediction_dir.glob("*_sampled_0.pdb")
+                    }
+                    n = len(completed)
+                else:
+                    n = 0
+                if n != last_n and progress_callback:
+                    progress_callback(n, total)
+                    last_n = n
+                time.sleep(0.5)
+
+            t.join(timeout=2.0)
+            if state["error"]:
+                raise state["error"]
+            if progress_callback:
+                progress_callback(total, total)
+
+            results = {}
+            for stem, sid in stem_to_id.items():
+                pdbs = []
+                for i in range(ensemble_size):
+                    p = prediction_dir / f"{stem}_sampled_{i}.pdb"
+                    if p.exists():
+                        content = p.read_text()
+                        if content.strip():
+                            pdbs.append(content)
+                results[sid] = pdbs
+
+            buf = io.BytesIO()
+            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
+                for sid, pdbs in results.items():
+                    stem = sanitize_id(sid)
+                    for i, pdb in enumerate(pdbs):
+                        zf.writestr(f"{stem}/sample_{i}.pdb", pdb)
+
+            elapsed = time.time() - start
+            return results, buf.getvalue(), elapsed
```
