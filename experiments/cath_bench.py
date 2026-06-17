#!/usr/bin/env python3
"""
CATH S40 Benchmark: SimpleFold + TeaCache vs CATH Ground Truth

Uses maximally diverse CATH S40 structures for rigorous benchmarking.
Includes TeaCache ablation: no cache (threshold=0) vs cache (threshold=0.1).

Creates publication-quality figures showing:
- Quality (TM-score, RMSD, lDDT) vs compute tradeoff
- Inference time vs sequence length
- TeaCache speedup analysis

Prerequisites:
    1. Ensure diverse_cath_300.json is placed in cath_benchmark/
"""

import os
import sys
import time
import json
import numpy as np
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional, List
from scipy.spatial.transform import Rotation
from copy import deepcopy

# Set working directory and append src paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src' / 'simplefold'))

# Import SimpleFold-Turbo local inference requirements
import mlx.core as mx
from simplefold.inference import (
    initialize_esm_model,
    initialize_others,
    initialize_folding_model,
    generate_structure,
)
from utils.fasta_utils import process_fastas
from utils.datamodule_utils import process_one_inference_structure
from model.mlx.teacache import TeaCacheSampler, TeaCacheConfig

# =============================================================================
# Configuration
# =============================================================================

CATH_BENCHMARK_FILE = Path("cath_benchmark/diverse_cath_300.json")
OUTPUT_DIR = Path("cath_benchmark")
OUTPUT_DIR.mkdir(exist_ok=True)

# Models to benchmark
MODELS = ['simplefold_100M', 'simplefold_360M', 'simplefold_700M',
          'simplefold_1.1B', 'simplefold_1.6B', 'simplefold_3B']

# TeaCache thresholds: 0 = no caching, 0.1 = aggressive caching
TEACACHE_THRESHOLDS = [0.0, 0.1]

# Sequential run is recommended when running MLX locally to prevent GPU memory resource collision
PARALLEL_JOBS = 1

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str
    model: str
    teacache_threshold: float
    length: int
    sequence: str
    tm_score: Optional[float] = None
    rmsd: Optional[float] = None
    lddt: Optional[float] = None
    inference_time: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    error: Optional[str] = None

# =============================================================================
# Structure Parsing
# =============================================================================

def parse_pdb_ca(pdb_path: Path) -> np.ndarray:
    """Extract CA coordinates from PDB file."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') and ' CA ' in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords)

def parse_cif_ca(cif_content: str) -> Optional[np.ndarray]:
    """Extract CA coordinates from CIF content."""
    coords = []
    for line in cif_content.split('\n'):
        if line.startswith('ATOM') and ' CA ' in line:
            parts = line.split()
            for i, p in enumerate(parts):
                try:
                    if '.' in p and -1000 < float(p) < 1000:
                        if i + 2 < len(parts):
                            x = float(parts[i])
                            y = float(parts[i+1])
                            z = float(parts[i+2])
                            coords.append([x, y, z])
                            break
                except ValueError:
                    continue
    return np.array(coords) if coords else None

# =============================================================================
# Structure Comparison Metrics
# =============================================================================

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Align P onto Q using Kabsch algorithm, return aligned P."""
    P_center = P.mean(axis=0)
    Q_center = Q.mean(axis=0)
    P_c = P - P_center
    Q_c = Q - Q_center
    R, _ = Rotation.align_vectors(Q_c, P_c)
    return P_c @ R.as_matrix().T + Q_center

def compute_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute RMSD after Kabsch alignment."""
    P_aligned = kabsch_align(P, Q)
    return np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1)))

def compute_tm_score(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute TM-score."""
    P_aligned = kabsch_align(P, Q)
    L = len(Q)
    d0 = max(1.24 * (L - 15) ** (1/3) - 1.8, 0.5)
    distances = np.sqrt(np.sum((P_aligned - Q) ** 2, axis=1))
    return np.sum(1 / (1 + (distances / d0) ** 2)) / L

def compute_lddt(P: np.ndarray, Q: np.ndarray, cutoff: float = 15.0) -> float:
    """Compute lDDT (local Distance Difference Test)."""
    n = len(P)
    if n < 2:
        return 0.0

    # Pairwise distances
    P_dist = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
    Q_dist = np.sqrt(np.sum((Q[:, None, :] - Q[None, :, :]) ** 2, axis=2))

    # Only consider pairs within cutoff in reference
    mask = (Q_dist < cutoff) & (np.arange(n)[:, None] != np.arange(n)[None, :])

    if mask.sum() == 0:
        return 0.0

    diff = np.abs(P_dist - Q_dist)

    # lDDT thresholds: 0.5, 1, 2, 4 Angstroms
    scores = []
    for t in [0.5, 1.0, 2.0, 4.0]:
        preserved = (diff < t) & mask
        scores.append(preserved.sum() / mask.sum())

    return np.mean(scores)

def decode_atom_name(name_field):
    """Decode atom name from numpy array or string."""
    if isinstance(name_field, np.ndarray):
        return ''.join(chr(c + 32) for c in name_field if c != 0)
    return str(name_field).strip()

def save_cif_helper(coords, residues, atoms_array, job_id, output_path: Path):
    """Save structure in mmCIF format (AF3-compatible)."""
    ELEMENT_MAP = {7: 'N', 6: 'C', 8: 'O', 16: 'S', 1: 'H'}

    lines = [
        "data_simplefold_prediction",
        "#",
        f"_entry.id {job_id}",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_formal_charge",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]

    atom_idx = 0
    for res in residues:
        res_name = decode_atom_name(res['name'])[:3]
        res_num = int(res['res_idx']) + 1
        start_idx = int(res['atom_idx'])
        n_atoms = int(res['atom_num'])

        for j in range(n_atoms):
            if start_idx + j < len(atoms_array) and atom_idx < len(coords):
                atom = atoms_array[start_idx + j]
                atom_name = decode_atom_name(atom['name'])
                element = ELEMENT_MAP.get(int(atom['element']), 'C')
                coord = coords[atom_idx]

                line = (
                    f"ATOM {atom_idx+1} {element} {atom_name} . {res_name} A 1 {res_num} ? "
                    f"{coord[0]:.3f} {coord[1]:.3f} {coord[2]:.3f} 1.00 0.00 ? "
                    f"{res_num} {res_name} A {atom_name} 1"
                )
                lines.append(line)
                atom_idx += 1

    lines.append("#")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


class LocalInferenceEngine:
    """Manages the in-memory loading and execution of ESM and SimpleFold models."""

    def __init__(self, backend: str = 'mlx'):
        self.backend = backend
        self.device = "cpu"
        self.esm_model = None
        self.esm_dict = None
        self.af2_to_esm = None
        self.tokenizer = None
        self.featurizer = None
        self.processor = None
        self.flow = None
        self.sampler = None
        self.ccd_path = Path("../artifacts/cache/ccd.pkl")
        self.folding_model = None
        self.loaded_model_name = None

    def load_esm(self, model_name: str):
        """Initialize the shared ESM representation model."""
        import argparse
        args = argparse.Namespace(
            simplefold_model=model_name,
            ckpt_dir='../artifacts',
            num_steps=500,
            tau=0.1, no_log_timesteps=False,
            fasta_path='', nsample_per_protein=1,
            plddt=False, output_format='mmcif', backend=self.backend, seed=42
        )
        print("[LocalInferenceEngine] Loading ESM model (~3GB)...")
        self.esm_model, self.esm_dict, self.af2_to_esm = initialize_esm_model(args, self.device)
        self.tokenizer, self.featurizer, self.processor, self.flow, self.sampler = initialize_others(args, self.device)

    def load_folding_model(self, model_name: str):
        """Initialize the simplefold model weights."""
        import argparse
        if self.esm_model is None:
            self.load_esm(model_name)

        if self.loaded_model_name == model_name:
            return

        # Unload previous model
        if self.folding_model is not None:
            del self.folding_model
            mx.metal.clear_cache() if hasattr(mx.metal, 'clear_cache') else None

        args = argparse.Namespace(
            simplefold_model=model_name,
            ckpt_dir='../artifacts',
            num_steps=500,
            tau=0.1, no_log_timesteps=False,
            fasta_path='', nsample_per_protein=1,
            plddt=False, output_format='mmcif', backend=self.backend, seed=42
        )
        print(f"[LocalInferenceEngine] Loading folding model: {model_name}...")
        self.folding_model, _ = initialize_folding_model(args)
        self.loaded_model_name = model_name

    def fold(self, name: str, sequence: str, model_name: str, teacache_threshold: float):
        """Fold a single sequence in-process and return the coordinates and metadata."""
        self.load_folding_model(model_name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Write temp FASTA
            fasta_path = tmp_path / "protein.fasta"
            with open(fasta_path, 'w') as f:
                f.write(f">{name}\n{sequence}\n")

            # Run data pipeline using ESM
            process_fastas([fasta_path], tmp_path, self.ccd_path)

            struct_file = tmp_path / "structures" / "protein.npz"
            record_file = tmp_path / "records" / "protein.json"

            batch, structure, record = process_one_inference_structure(
                struct_file, record_file,
                self.tokenizer,
                self.featurizer,
                self.processor,
                self.esm_model,
                self.esm_dict,
                self.af2_to_esm
            )

            # Random noise initialization
            mx.random.seed(42)
            noise = mx.random.normal(batch['coords'].shape)

            # Configure and execute TeaCache sampling
            config = TeaCacheConfig(threshold=teacache_threshold)
            tea_sampler = TeaCacheSampler(num_timesteps=500, config=config)

            t0 = time.perf_counter()
            result = tea_sampler.sample(
                self.folding_model,
                self.flow,
                noise, batch, verbose=False
            )
            mx.eval(result['denoised_coords'])
            inference_time = time.perf_counter() - t0

            cache_stats = result.get('cache_stats', {})
            cache_hit_rate = cache_stats.get('hit_rate', 0)

            # Convert MLX coords to PDB scale
            coords = np.array(result['denoised_coords'][0]) * 16.0

            # Read structural array metadata
            struct_data = np.load(struct_file, allow_pickle=True)
            residues = struct_data['residues']
            atoms_array = struct_data['atoms']

            # Write to a temp mmcif file to be read back by parse_cif_ca
            output_cif_path = tmp_path / "result.cif"
            save_cif_helper(coords, residues, atoms_array, name, output_cif_path)

            with open(output_cif_path) as f:
                cif_content = f.read()

            # Clean up intermediate variables and clear cache to free unified RAM
            del result, noise, tea_sampler, batch, coords, struct_data
            import gc
            gc.collect()
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()

            return {
                'status': 'completed',
                'results': [{
                    'inference_time_s': inference_time,
                    'cache_hit_rate': cache_hit_rate
                }],
                'cif_content': cif_content
            }


# Single global engine instance for benchmark execution
local_infer = LocalInferenceEngine(backend='mlx')

# =============================================================================
# Benchmark Runner
# =============================================================================

import urllib.request
import urllib.error

def ensure_pdb_downloaded(name: str) -> Path:
    """Ensure the PDB file for the CATH domain is downloaded and cached locally."""
    local_dir = Path("cath_benchmark/dompdb")
    local_dir.mkdir(exist_ok=True, parents=True)
    local_path = local_dir / f"{name}.pdb"
    if not local_path.exists():
        url = f"https://www.cathdb.info/version/v4_3_0/api/rest/id/{name}.pdb"
        print(f"Downloading ground truth PDB for {name} from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                content = response.read()
                if b"ATOM" in content or b"HEADER" in content:
                    local_path.write_bytes(content)
                else:
                    raise ValueError(f"Downloaded content for {name} does not look like a PDB file.")
        except Exception as e:
            print(f"Error downloading {name} from CATH v4.3.0 API: {e}. Trying latest...")
            try:
                alt_url = f"https://www.cathdb.info/version/latest/api/rest/id/{name}.pdb"
                with urllib.request.urlopen(alt_url, timeout=15) as response:
                    content = response.read()
                    if b"ATOM" in content or b"HEADER" in content:
                        local_path.write_bytes(content)
                    else:
                        raise ValueError(f"Downloaded content for {name} does not look like a PDB file.")
            except Exception as e2:
                print(f"Failed to download PDB for {name}: {e2}")
                raise e2
    return local_path

def benchmark_one(struct: dict, model: str, teacache_threshold: float) -> BenchmarkResult:
    """Benchmark a single structure with specific model and cache setting."""
    name = struct['name']
    sequence = struct['sequence']
    length = struct['length']

    try:
        # Ensure ground truth PDB is downloaded
        pdb_path = ensure_pdb_downloaded(name)

        # Fold
        cache_str = f"tc{teacache_threshold}"
        result = local_infer.fold(f"{name}_{model}_{cache_str}", sequence, model, teacache_threshold)

        if result.get('status') != 'completed':
            return BenchmarkResult(
                name=name, model=model, teacache_threshold=teacache_threshold,
                length=length, sequence=sequence,
                error=result.get('error', 'Unknown')
            )

        res = result['results'][0]
        inference_time = res['inference_time_s']
        cache_hit_rate = res.get('cache_hit_rate', 0)

        # Get prediction
        cif_content = result.get('cif_content', '')
        pred_coords = parse_cif_ca(cif_content)
        gt_coords = parse_pdb_ca(pdb_path)

        if pred_coords is None or len(pred_coords) < 10 or len(gt_coords) < 10:
            return BenchmarkResult(
                name=name, model=model, teacache_threshold=teacache_threshold,
                length=length, sequence=sequence,
                inference_time=inference_time, cache_hit_rate=cache_hit_rate,
                error="Failed to parse coordinates"
            )

        # Align lengths
        min_len = min(len(pred_coords), len(gt_coords))
        pred_coords = pred_coords[:min_len]
        gt_coords = gt_coords[:min_len]

        # Compute metrics
        tm_score = compute_tm_score(pred_coords, gt_coords)
        rmsd = compute_rmsd(pred_coords, gt_coords)
        lddt = compute_lddt(pred_coords, gt_coords)

        return BenchmarkResult(
            name=name, model=model, teacache_threshold=teacache_threshold,
            length=length, sequence=sequence,
            tm_score=tm_score, rmsd=rmsd, lddt=lddt,
            inference_time=inference_time, cache_hit_rate=cache_hit_rate
        )

    except Exception as e:
        return BenchmarkResult(
            name=name, model=model, teacache_threshold=teacache_threshold,
            length=length, sequence=sequence,
            error=str(e)
        )

def process_job(args):
    """Process a single benchmark job (for thread pool)."""
    struct, model, threshold = args
    return benchmark_one(struct, model, threshold)

# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("CATH S40 BENCHMARK: SimpleFold + TeaCache Ablation")
    print("=" * 70)

    # Load structures
    print(f"\nLoading diverse CATH set from {CATH_BENCHMARK_FILE}...")
    if not CATH_BENCHMARK_FILE.exists():
        print("ERROR: Run python select_diverse_cath.py first!")
        sys.exit(1)

    with open(CATH_BENCHMARK_FILE) as f:
        structures = json.load(f)

    # Allow limiting structures for testing/debugging
    limit = os.environ.get("BENCHMARK_LIMIT")
    if limit:
        structures = structures[:int(limit)]
        print(f"Limiting to first {len(structures)} structures for quick run")
    else:
        print(f"Loaded {len(structures)} structures")
    lengths = [s['length'] for s in structures]
    print(f"Length range: {min(lengths)}-{max(lengths)} aa")

    # Dynamically select only models that have checkpoints available locally
    ckpt_dir = Path("../artifacts")
    available_models = []
    for model in MODELS:
        ckpt_path = ckpt_dir / f"{model}.ckpt"
        sf_path = ckpt_dir / f"{model}.safetensors"
        if ckpt_path.exists() or sf_path.exists():
            available_models.append(model)
    
    if not available_models:
        print(f"WARNING: No local checkpoints found in {ckpt_dir}. Defaults to all models.")
        run_models = MODELS
    else:
        print(f"Found checkpoints for: {available_models}. Only benchmarking these.")
        run_models = available_models

    # ==========================================================================
    # Run Benchmark
    # ==========================================================================

    all_results: List[BenchmarkResult] = []

    for model in run_models:
        for threshold in TEACACHE_THRESHOLDS:
            cache_label = "NO_CACHE" if threshold == 0 else f"CACHE_{threshold}"

            print(f"\n{'='*70}")
            print(f"MODEL: {model} | TeaCache: {cache_label}")
            print('='*70)

            jobs = [(s, model, threshold) for s in structures]

            completed = 0
            with ThreadPoolExecutor(max_workers=PARALLEL_JOBS) as executor:
                futures = {executor.submit(process_job, job): job for job in jobs}

                for future in as_completed(futures):
                    completed += 1
                    result = future.result()

                    if result.error:
                        print(f"[{completed}/{len(jobs)}] {result.name[:20]}... ERROR: {result.error}")
                    else:
                        print(f"[{completed}/{len(jobs)}] {result.name[:20]}... "
                              f"TM={result.tm_score:.3f} | RMSD={result.rmsd:.1f}Å | "
                              f"lDDT={result.lddt:.3f} | Time={result.inference_time:.1f}s | "
                              f"Cache={result.cache_hit_rate:.0%}")
                        all_results.append(result)

    # ==========================================================================
    # Save Results
    # ==========================================================================

    if not all_results:
        print("\nERROR: No benchmark results were successfully generated (all predictions failed).")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # CSV
    csv_path = OUTPUT_DIR / "cath_benchmark_full.csv"
    with open(csv_path, 'w') as f:
        f.write("model,teacache_threshold,name,length,tm_score,rmsd,lddt,time_s,cache_hit\n")
        for r in all_results:
            f.write(f"{r.model},{r.teacache_threshold},{r.name},{r.length},"
                    f"{r.tm_score:.4f},{r.rmsd:.2f},{r.lddt:.4f},"
                    f"{r.inference_time:.2f},{r.cache_hit_rate:.3f}\n")
    print(f"Saved: {csv_path}")

    # JSON
    json_path = OUTPUT_DIR / "cath_benchmark_full.json"
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"Saved: {json_path}")

    # ==========================================================================
    # Generate Figures
    # ==========================================================================

    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import pandas as pd

    df = pd.DataFrame([asdict(r) for r in all_results])

    # Color scheme
    model_colors = {
        'simplefold_100M': '#4169E1',
        'simplefold_360M': '#32CD32',
        'simplefold_700M': "#B86720",
        'simplefold_1.1B': '#FF6347',
        'simplefold_1.6B': "#DE60D6",
        'simplefold_3B': "#9D47FF",
    }

    # ==========================================================================
    # Figure 1: Quality vs Speed (with cache comparison)
    # ==========================================================================

    fig1, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Quality vs Speed (bubble plot)
    ax = axes[0]
    sizes = {'simplefold_100M': 150, 'simplefold_360M': 400,
             'simplefold_700M': 800, 'simplefold_1.1B': 1200,
             'simplefold_1.6B': 1800, 'simplefold_3B': 3000}

    for model in MODELS:
        for threshold in TEACACHE_THRESHOLDS:
            mask = (df['model'] == model) & (df['teacache_threshold'] == threshold)
            model_df = df[mask]
            if len(model_df) == 0:
                continue

            avg_tm = model_df['tm_score'].mean()
            avg_time = model_df['time_s' if 'time_s' in df.columns else 'inference_time'].mean()

            marker = 'o' if threshold > 0 else 's'  # circle=cache, square=no cache
            alpha = 0.8 if threshold > 0 else 0.4
            label = f"{model.replace('simplefold_', '')} ({'cache' if threshold > 0 else 'no cache'})"

            ax.scatter(avg_time, avg_tm, s=sizes[model], c=model_colors[model],
                      alpha=alpha, marker=marker, edgecolors='black', linewidth=1.5,
                      label=label)

    ax.set_xlabel('Avg Inference Time (seconds)', fontsize=12)
    ax.set_ylabel('Avg TM-score vs CATH', fontsize=12)
    ax.set_title('Quality vs Speed (○=TeaCache, □=No Cache)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8, ncol=2)

    # Panel B: TeaCache Speedup
    ax = axes[1]
    speedups = []
    for model in MODELS:
        no_cache = df[(df['model'] == model) & (df['teacache_threshold'] == 0)]
        with_cache = df[(df['model'] == model) & (df['teacache_threshold'] == 0.1)]

        if len(no_cache) > 0 and len(with_cache) > 0:
            time_col = 'time_s' if 'time_s' in df.columns else 'inference_time'
            t_nocache = no_cache[time_col].mean()
            t_cache = with_cache[time_col].mean()
            speedup = t_nocache / t_cache if t_cache > 0 else 1.0
            speedups.append({
                'model': model.replace('simplefold_', ''),
                'speedup': speedup,
                't_nocache': t_nocache,
                't_cache': t_cache,
                'color': model_colors[model]
            })

    if speedups:
        x = range(len(speedups))
        bars = ax.bar(x, [s['speedup'] for s in speedups],
                      color=[s['color'] for s in speedups], alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([s['model'] for s in speedups], rotation=45, ha='right')
        ax.set_ylabel('Speedup (no cache / with cache)', fontsize=12)
        ax.set_title('TeaCache Speedup by Model', fontsize=13, fontweight='bold')
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='No speedup')
        ax.grid(True, alpha=0.3, axis='y')

        # Add speedup labels
        for i, s in enumerate(speedups):
            ax.annotate(f"{s['speedup']:.2f}x", (i, s['speedup']), ha='center',
                       va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig1_path = OUTPUT_DIR / "cath_quality_speed.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fig1_path}")

    # ==========================================================================
    # Figure 2: Time vs Length (with cache comparison)
    # ==========================================================================

    fig2, ax = plt.subplots(figsize=(14, 8))

    time_col = 'time_s' if 'time_s' in df.columns else 'inference_time'

    for model in MODELS:
        for threshold in TEACACHE_THRESHOLDS:
            mask = (df['model'] == model) & (df['teacache_threshold'] == threshold)
            model_df = df[mask].sort_values('length')
            if len(model_df) == 0:
                continue

            linestyle = '-' if threshold > 0 else '--'
            alpha = 0.8 if threshold > 0 else 0.4
            label = f"{model.replace('simplefold_', '')} ({'cache' if threshold > 0 else 'no cache'})"

            ax.scatter(model_df['length'], model_df[time_col],
                      s=40, c=model_colors[model], alpha=alpha, edgecolors='black', linewidth=0.3)

            # Trend line
            if len(model_df) > 5:
                window = max(3, len(model_df) // 10)
                rolling = model_df[time_col].rolling(window=window, center=True, min_periods=1).mean()
                ax.plot(model_df['length'], rolling, color=model_colors[model],
                       linewidth=2.5, alpha=alpha, linestyle=linestyle, label=label)

    ax.set_xlabel('Sequence Length (residues)', fontsize=12)
    ax.set_ylabel('Inference Time (seconds)', fontsize=12)
    ax.set_title('Inference Time vs Length (solid=TeaCache, dashed=No Cache)', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fig2_path = OUTPUT_DIR / "cath_time_vs_length.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fig2_path}")

    # ==========================================================================
    # Figure 3: Summary Statistics
    # ==========================================================================

    fig3, axes = plt.subplots(1, 3, figsize=(15, 5))

    # TM-score distribution
    ax = axes[0]
    for model in MODELS:
        model_df = df[(df['model'] == model) & (df['teacache_threshold'] == 0.1)]
        if len(model_df) > 0:
            ax.hist(model_df['tm_score'], bins=20, alpha=0.5, label=model.replace('simplefold_', ''),
                   color=model_colors[model], edgecolor='black')
    ax.set_xlabel('TM-score')
    ax.set_ylabel('Count')
    ax.set_title('TM-score Distribution (with TeaCache)')
    ax.legend()

    # RMSD distribution
    ax = axes[1]
    for model in MODELS:
        model_df = df[(df['model'] == model) & (df['teacache_threshold'] == 0.1)]
        if len(model_df) > 0:
            ax.hist(model_df['rmsd'], bins=20, alpha=0.5, label=model.replace('simplefold_', ''),
                   color=model_colors[model], edgecolor='black')
    ax.set_xlabel('RMSD (Å)')
    ax.set_ylabel('Count')
    ax.set_title('RMSD Distribution (with TeaCache)')

    # lDDT distribution
    ax = axes[2]
    for model in MODELS:
        model_df = df[(df['model'] == model) & (df['teacache_threshold'] == 0.1)]
        if len(model_df) > 0:
            ax.hist(model_df['lddt'], bins=20, alpha=0.5, label=model.replace('simplefold_', ''),
                   color=model_colors[model], edgecolor='black')
    ax.set_xlabel('lDDT')
    ax.set_ylabel('Count')
    ax.set_title('lDDT Distribution (with TeaCache)')

    plt.tight_layout()
    fig3_path = OUTPUT_DIR / "cath_distributions.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fig3_path}")

    # ==========================================================================
    # Print Summary
    # ==========================================================================

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    time_col = 'time_s' if 'time_s' in df.columns else 'inference_time'

    for model in MODELS:
        print(f"\n{model}:")
        for threshold in TEACACHE_THRESHOLDS:
            mask = (df['model'] == model) & (df['teacache_threshold'] == threshold)
            model_df = df[mask]
            if len(model_df) == 0:
                continue
            cache_str = "TeaCache" if threshold > 0 else "No Cache"
            print(f"  [{cache_str}]")
            print(f"    TM-score: {model_df['tm_score'].mean():.3f} ± {model_df['tm_score'].std():.3f}")
            print(f"    RMSD:     {model_df['rmsd'].mean():.2f} ± {model_df['rmsd'].std():.2f} Å")
            print(f"    lDDT:     {model_df['lddt'].mean():.3f} ± {model_df['lddt'].std():.3f}")
            print(f"    Time:     {model_df[time_col].mean():.1f} ± {model_df[time_col].std():.1f} s")
            if threshold > 0:
                print(f"    Cache:    {model_df['cache_hit_rate'].mean()*100:.1f}%")

    # TeaCache Impact Summary
    print(f"\n{'='*70}")
    print("TEACACHE IMPACT SUMMARY")
    print("="*70)

    for model in MODELS:
        no_cache = df[(df['model'] == model) & (df['teacache_threshold'] == 0)]
        with_cache = df[(df['model'] == model) & (df['teacache_threshold'] == 0.1)]

        if len(no_cache) > 0 and len(with_cache) > 0:
            t_nc = no_cache[time_col].mean()
            t_c = with_cache[time_col].mean()
            speedup = t_nc / t_c if t_c > 0 else 1.0
            tm_diff = with_cache['tm_score'].mean() - no_cache['tm_score'].mean()

            print(f"\n{model}:")
            print(f"  Speedup: {speedup:.2f}x ({t_nc:.1f}s → {t_c:.1f}s)")
            print(f"  TM-score change: {tm_diff:+.4f}")

    print("\n✅ CATH BENCHMARK COMPLETE!")


if __name__ == "__main__":
    main()
