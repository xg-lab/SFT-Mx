#!/usr/bin/env python3
"""
Dual Sweep: Uniform Step-Skipping + Adaptive Caching across All Model Sizes.

Runs both uniform step-skipping and SF-T adaptive caching for all 6 model
sizes on the CATH-300 benchmark. Saves full-atom PDB files for every prediction
so analyses and figures can be regenerated without re-running inference.

Output structure:
    publication/data/structures/{model}/uniform_{steps}/{name}.pdb
    publication/data/structures/{model}/adaptive_{threshold}/{name}.pdb
    publication/data/dual_sweep_{model}.json   (per-model metrics)
    publication/data/dual_sweep_summary.csv    (merged summary)

Usage:
    python dual_sweep.py                                    # full run
    python dual_sweep.py --models simplefold_100M --n_proteins 5  # dry-run
"""

import argparse
import gc
import json
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional, Dict

EXPERIMENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = EXPERIMENT_DIR.parent
os.chdir(EXPERIMENT_DIR)
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'simplefold'))

import mlx.core as mx


# =============================================================================
# Configuration
# =============================================================================

ALL_MODELS = [
    'simplefold_100M',
    'simplefold_360M',
    'simplefold_700M',
    'simplefold_1.1B',
    'simplefold_1.6B',
    'simplefold_3B',
]

# Uniform step counts (matching mean computed steps from threshold sweep)
STEP_COUNTS = [12, 15, 16, 24, 36, 59, 100, 171, 250]

# Adaptive thresholds (0.0 = no caching baseline)
THRESHOLDS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 1.0]

N_PROTEINS = 300

# Ground truth PDB directory (CATH domain PDBs)
GT_PDB_DIR = Path("/Users/gjt4/Documents/GitHub/newton/dompdb")

DATA_DIR = PROJECT_ROOT / 'publication' / 'data'
STRUCTURES_DIR = DATA_DIR / 'structures'


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SweepResult:
    """Results for one protein under one condition."""
    name: str
    model: str
    method: str           # 'uniform' or 'adaptive'
    condition: str        # step count (e.g. '36') or threshold (e.g. '0.1')
    num_residues: int
    inference_time: float
    # Adaptive-only fields
    cache_hit_rate: Optional[float] = None
    n_computed_steps: Optional[int] = None
    # Quality metrics
    rmsd_vs_gt: Optional[float] = None
    tm_score_vs_gt: Optional[float] = None
    rmsd_vs_baseline: Optional[float] = None


# =============================================================================
# Utility Functions
# =============================================================================

def superimpose(pred: np.ndarray, gt: np.ndarray):
    """Superimpose pred onto gt using BioPython SVDSuperimposer.

    Returns (aligned_pred, rmsd).
    """
    from Bio.SVDSuperimposer import SVDSuperimposer
    sup = SVDSuperimposer()
    sup.set(gt, pred)
    sup.run()
    aligned = sup.get_transformed()
    return aligned, sup.get_rms()


def compute_tm_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """TM-score (Zhang & Skolnick, 2004) after optimal superposition."""
    aligned, _ = superimpose(pred, gt)
    L = len(gt)
    d0 = max(1.24 * (L - 15) ** (1/3) - 1.8, 0.5)
    distances = np.sqrt(np.sum((aligned - gt) ** 2, axis=1))
    return float(np.sum(1 / (1 + (distances / d0) ** 2)) / L)


def extract_ca_from_pdb(pdb_path: Path) -> Optional[np.ndarray]:
    """Extract CA coordinates from a PDB file. Same as load_gt_coords."""
    return load_gt_coords(pdb_path)


def load_fasta_sequences(fasta_path: Path) -> List[Tuple[str, str]]:
    sequences = []
    current_name = None
    current_seq = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name:
                    sequences.append((current_name, ''.join(current_seq)))
                current_name = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_name:
            sequences.append((current_name, ''.join(current_seq)))
    return sequences


def load_gt_coords(pdb_path: Path) -> Optional[np.ndarray]:
    """Load CA coordinates from PDB, taking only the first altloc per residue."""
    ca_coords = []
    seen_residues = set()
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    chain = line[21]
                    resnum = line[22:27].strip()  # includes insertion code
                    key = f"{chain}:{resnum}"
                    if key in seen_residues:
                        continue  # skip alternate conformations
                    seen_residues.add(key)
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_coords.append([x, y, z])
        return np.array(ca_coords) if ca_coords else None
    except:
        return None


# =============================================================================
# TeaCache Sampler with Stats Tracking
# =============================================================================

class TeaCacheSamplerWithStats:
    """TeaCache sampler that tracks cache hit/miss statistics."""

    def __init__(self, num_timesteps: int, threshold: float = 0.1, tau: float = 0.1,
                 warmup_steps: int = 10, log_timesteps: bool = True):
        self.num_timesteps = num_timesteps
        self.threshold = threshold
        self.tau = tau
        self.warmup_steps = warmup_steps

        if log_timesteps:
            self.steps = mx.exp(mx.linspace(np.log(1e-4), np.log(1.0), num=num_timesteps + 1))
        else:
            self.steps = mx.linspace(1e-4, 1.0, num=num_timesteps + 1)

        self.n_computed = 0
        self.n_cached = 0

    def _compute_modulated_input(self, noised_pos, t):
        t_val = t.item() if hasattr(t, 'item') else float(t)
        pos_norm = mx.sqrt((noised_pos ** 2).sum(axis=-1).mean())
        modulated = pos_norm * (1.0 - t_val + 0.1)
        return modulated.item() if hasattr(modulated, 'item') else float(modulated)

    def diffusion_coefficient(self, t, eps=0.01, w_cutoff=0.99):
        w = (1.0 - t) / (t + eps)
        t_val = t.item() if hasattr(t, 'item') else float(t)
        if t_val >= w_cutoff:
            w = 0.0
        return w

    def sample(self, model_fn, flow, noise, batch):
        from einops.array_api import repeat
        from utils.mlx_utils import center_random_augmentation

        y = noise
        cached_output = None
        prev_modulated_input = None
        accumulated_diff = 0.0
        self.n_computed = 0
        self.n_cached = 0

        for i in range(self.num_timesteps):
            t = self.steps[i]
            t_next = self.steps[i + 1]
            dt = t_next - t
            eps = mx.random.normal(y.shape)

            y = center_random_augmentation(
                y, batch["atom_pad_mask"],
                augmentation=False, centering=True,
            )

            modulated_input = self._compute_modulated_input(y, t)

            recompute = False
            if self.threshold == 0.0:
                recompute = True
            elif i < self.warmup_steps or cached_output is None:
                recompute = True
            else:
                input_diff = abs(modulated_input - prev_modulated_input) / (abs(prev_modulated_input) + 1e-6)
                accumulated_diff += input_diff
                if accumulated_diff >= self.threshold:
                    accumulated_diff = 0.0
                    recompute = True

            if recompute:
                batched_t = repeat(t, " -> b", b=y.shape[0])
                output = model_fn(noised_pos=y, t=batched_t, feats=batch)
                velocity = output["predict_velocity"]
                cached_output = velocity
                self.n_computed += 1
            else:
                velocity = cached_output
                self.n_cached += 1

            prev_modulated_input = modulated_input

            score = flow.compute_score_from_velocity(velocity, y, t)
            diff_coeff = self.diffusion_coefficient(t)
            drift = velocity + diff_coeff * score
            mean_y = y + drift * dt
            y = mean_y + mx.sqrt(2.0 * dt * diff_coeff * self.tau) * eps
            mx.eval(y)

        return y

    @property
    def cache_hit_rate(self):
        total = self.n_computed + self.n_cached
        return self.n_cached / total if total > 0 else 0.0


# =============================================================================
# PDB Saving
# =============================================================================

def debug_superimpose(pred_ca: np.ndarray, gt_ca: np.ndarray, name: str, label: str,
                      rmsd: float, tm: float, save_dir: Path):
    """DEBUG: Save a 3D scatter of aligned pred (yellow) vs GT (green)."""
    import matplotlib.pyplot as plt

    aligned, _ = superimpose(pred_ca, gt_ca)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*gt_ca.T, c='green', s=10, alpha=0.7, label='GT')
    ax.scatter(*aligned.T, c='gold', s=10, alpha=0.7, label='Pred (aligned)')
    ax.set_title(f'{name} | {label}\nRMSD={rmsd:.2f}Å  TM={tm:.3f}')
    ax.legend()
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"debug_{name}_{label}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    DEBUG saved: {out}")


def save_prediction_pdb(denoised_coords, batch, structure, record, save_dir: Path, name: str):
    """Save a prediction as a full-atom PDB file."""
    from utils.boltz_utils import process_structure, save_structure

    save_dir.mkdir(parents=True, exist_ok=True)
    structure_out = process_structure(
        structure, denoised_coords[0], batch['atom_pad_mask'][0], record, backend="mlx"
    )
    save_structure(structure_out, save_dir, name, output_format="pdb")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Dual sweep: uniform + adaptive across all models")
    parser.add_argument('--models', nargs='+', default=ALL_MODELS,
                        help='Model(s) to run (default: all 6)')
    parser.add_argument('--n_proteins', type=int, default=N_PROTEINS,
                        help='Number of proteins (default: 300)')
    return parser.parse_args()


def main():
    args = parse_args()
    models = args.models
    n_proteins = args.n_proteins

    print("=" * 70)
    print("DUAL SWEEP: Uniform + Adaptive × All Models")
    print("=" * 70)
    print(f"Models:          {models}")
    print(f"Uniform steps:   {STEP_COUNTS}")
    print(f"Adaptive τ:      {THRESHOLDS}")
    print(f"Proteins:        {n_proteins}")
    total_conditions = len(STEP_COUNTS) + len(THRESHOLDS)
    total_runs = len(models) * n_proteins * total_conditions
    print(f"Total runs:      {len(models)} × {n_proteins} × {total_conditions} = {total_runs}")

    # Load sequences (once for all models)
    fasta_path = DATA_DIR / "diverse_cath_300.fasta"
    sequences = load_fasta_sequences(fasta_path)[:n_proteins]
    print(f"\nLoaded {len(sequences)} sequences from {fasta_path}")

    all_summary_results = []
    sweep_start = time.time()

    for model_idx, model_name in enumerate(models):
        print(f"\n{'#' * 70}")
        print(f"# MODEL {model_idx+1}/{len(models)}: {model_name}")
        print(f"{'#' * 70}")

        # Load model (must chdir for config paths)
        simplefold_dir = PROJECT_ROOT / 'src' / 'simplefold'
        os.chdir(simplefold_dir)

        from wrapper import ModelWrapper, InferenceWrapper
        from model.mlx.sampler import EMSampler as EMSamplerMLX
        from model.flow import LinearPath

        model_wrapper = ModelWrapper(
            simplefold_model=model_name,
            plddt=False,
            ckpt_dir=str(PROJECT_ROOT / 'artifacts'),
            backend='mlx',
        )

        inference_wrapper = InferenceWrapper(
            output_dir=str(EXPERIMENT_DIR / 'dual_sweep_cache'),
            prediction_dir='predictions',
            num_steps=500,
            nsample_per_protein=1,
            tau=0.1,
            device='cpu',
            backend='mlx',
        )

        model = model_wrapper.from_pretrained_folding_model()
        flow = LinearPath()
        print(f"{model_name} loaded!")

        os.chdir(EXPERIMENT_DIR)

        # Per-model results
        model_results: List[SweepResult] = []
        baseline_coords: Dict[str, np.ndarray] = {}  # τ=0.0 coords for RMSD-vs-baseline
        debug_plotted = {'uniform': False, 'adaptive': False}  # one debug plot per method
        model_start = time.time()

        for prot_idx, (name, sequence) in enumerate(sequences):
            print(f"\n[{model_name}] [{prot_idx+1:3d}/{len(sequences)}] {name} ({len(sequence)} res)")

            batch, structure, record = inference_wrapper.process_input(sequence)

            # Load GT
            gt_pdb = GT_PDB_DIR / f"{name}.pdb"
            gt_coords = load_gt_coords(gt_pdb) if gt_pdb.exists() else None

            # =================================================================
            # Uniform step-skipping
            # =================================================================
            for n_steps in STEP_COUNTS:
                condition_str = str(n_steps)
                pdb_dir = STRUCTURES_DIR / model_name / f"uniform_{n_steps}"
                pdb_path = pdb_dir / f"{name}.pdb"

                # Skip if PDB already exists (resume support)
                if pdb_path.exists():
                    print(f"  uniform {n_steps:4d}: SKIP (exists)")
                    continue

                mx.random.seed(42 + prot_idx)
                noise = mx.random.normal(batch['coords'].shape)

                sampler = EMSamplerMLX(
                    num_timesteps=n_steps,
                    t_start=1e-4,
                    tau=0.1,
                    log_timesteps=True,
                    w_cutoff=0.99,
                )

                t0 = time.time()
                out_dict = sampler.sample(model, flow, noise, batch)
                # Postprocess: center + scale back to Angstroms (×16)
                out_dict = inference_wrapper.processor.postprocess(out_dict, batch)
                mx.eval(out_dict['denoised_coords'])
                elapsed = time.time() - t0

                # Save PDB
                save_prediction_pdb(
                    out_dict['denoised_coords'], batch, structure, record,
                    pdb_dir, name
                )

                # Metrics — extract CAs from the saved PDB (correct atom indexing)
                ca_coords = extract_ca_from_pdb(pdb_path)
                rmsd_vs_gt = None
                tm_score_vs_gt = None
                if ca_coords is not None and gt_coords is not None and len(gt_coords) == len(ca_coords):
                    try:
                        _, rmsd_vs_gt = superimpose(ca_coords, gt_coords)
                        rmsd_vs_gt = float(rmsd_vs_gt)
                        tm_score_vs_gt = compute_tm_score(ca_coords, gt_coords)
                    except Exception:
                        pass

                # DEBUG: visualize first valid protein
                if not debug_plotted['uniform'] and rmsd_vs_gt is not None:
                    debug_plotted['uniform'] = True
                    debug_superimpose(ca_coords, gt_coords, name,
                                      f"uniform_{n_steps}", rmsd_vs_gt, tm_score_vs_gt,
                                      EXPERIMENT_DIR / "debug_plots")

                result = SweepResult(
                    name=name, model=model_name, method='uniform',
                    condition=condition_str, num_residues=len(sequence),
                    inference_time=elapsed, rmsd_vs_gt=rmsd_vs_gt,
                    tm_score_vs_gt=tm_score_vs_gt,
                )
                model_results.append(result)

                rmsd_str = f"{rmsd_vs_gt:.2f}" if rmsd_vs_gt is not None else "N/A"
                tm_str = f"{tm_score_vs_gt:.3f}" if tm_score_vs_gt is not None else "N/A"
                print(f"  uniform {n_steps:4d}: {elapsed:5.1f}s  RMSD={rmsd_str:>7}  TM={tm_str:>6}")

                # Free inner memory and clear cache
                del noise, sampler, out_dict
                import gc
                gc.collect()
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                elif hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()

            # =================================================================
            # Adaptive caching (threshold sweep)
            # =================================================================
            for threshold in THRESHOLDS:
                condition_str = str(threshold)
                pdb_dir = STRUCTURES_DIR / model_name / f"adaptive_{threshold}"
                pdb_path = pdb_dir / f"{name}.pdb"

                if pdb_path.exists():
                    print(f"  adaptive τ={threshold:.2f}: SKIP (exists)")
                    continue

                mx.random.seed(42 + prot_idx)
                noise = mx.random.normal(batch['coords'].shape)

                sampler = TeaCacheSamplerWithStats(
                    num_timesteps=500,
                    threshold=threshold,
                    tau=0.1,
                    warmup_steps=10,
                )

                t0 = time.time()
                final_coords = sampler.sample(model, flow, noise, batch)
                # Postprocess: center + scale back to Angstroms (×16)
                out_dict_adaptive = {'denoised_coords': final_coords}
                out_dict_adaptive = inference_wrapper.processor.postprocess(out_dict_adaptive, batch)
                final_coords = out_dict_adaptive['denoised_coords']
                mx.eval(final_coords)
                elapsed = time.time() - t0

                # Save PDB
                # final_coords is (1, n_atoms, 3) from the sampler — wrap if needed
                if final_coords.ndim == 2:
                    final_coords = final_coords[None, ...]
                save_prediction_pdb(
                    final_coords, batch, structure, record,
                    pdb_dir, name
                )

                # Metrics — extract CAs from the saved PDB (correct atom indexing)
                ca_coords = extract_ca_from_pdb(pdb_path)

                if ca_coords is not None and threshold == 0.0:
                    baseline_coords[name] = ca_coords.copy()

                rmsd_vs_baseline = None
                if ca_coords is not None and threshold > 0.0 and name in baseline_coords:
                    _, rmsd_vs_baseline = superimpose(ca_coords, baseline_coords[name])
                    rmsd_vs_baseline = float(rmsd_vs_baseline)

                rmsd_vs_gt = None
                tm_score_vs_gt = None
                if ca_coords is not None and gt_coords is not None and len(gt_coords) == len(ca_coords):
                    try:
                        _, rmsd_vs_gt = superimpose(ca_coords, gt_coords)
                        rmsd_vs_gt = float(rmsd_vs_gt)
                        tm_score_vs_gt = compute_tm_score(ca_coords, gt_coords)
                    except Exception:
                        pass

                # DEBUG: visualize first valid baseline
                if not debug_plotted['adaptive'] and threshold == 0.0 and rmsd_vs_gt is not None:
                    debug_plotted['adaptive'] = True
                    debug_superimpose(ca_coords, gt_coords, name,
                                      f"adaptive_tau0.0", rmsd_vs_gt, tm_score_vs_gt,
                                      EXPERIMENT_DIR / "debug_plots")

                result = SweepResult(
                    name=name, model=model_name, method='adaptive',
                    condition=condition_str, num_residues=len(sequence),
                    inference_time=elapsed,
                    cache_hit_rate=sampler.cache_hit_rate,
                    n_computed_steps=sampler.n_computed,
                    rmsd_vs_gt=rmsd_vs_gt, tm_score_vs_gt=tm_score_vs_gt,
                    rmsd_vs_baseline=rmsd_vs_baseline,
                )
                model_results.append(result)

                rmsd_str = f"{rmsd_vs_gt:.2f}" if rmsd_vs_gt is not None else "N/A"
                tm_str = f"{tm_score_vs_gt:.3f}" if tm_score_vs_gt is not None else "N/A"
                print(f"  adaptive τ={threshold:.2f}: {sampler.n_computed:3d} steps | "
                      f"hit={sampler.cache_hit_rate:.1%} | {elapsed:5.1f}s | "
                      f"RMSD={rmsd_str:>7}  TM={tm_str:>6}")

                # Free inner memory and clear cache
                del noise, sampler, final_coords, out_dict_adaptive
                import gc
                gc.collect()
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                elif hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()

            # Free outer protein loop variables
            del batch, structure, record, gt_coords
            import gc
            gc.collect()

        model_elapsed = time.time() - model_start
        print(f"\n{model_name} complete: {model_elapsed/60:.1f} min, {len(model_results)} results")

        # Save per-model JSON
        json_path = DATA_DIR / f"dual_sweep_{model_name}.json"

        def to_serializable(r):
            d = asdict(r)
            for k, v in d.items():
                if isinstance(v, (np.floating, np.integer)):
                    d[k] = float(v) if isinstance(v, np.floating) else int(v)
                elif hasattr(v, 'item'):
                    d[k] = v.item()
            return d

        with open(json_path, 'w') as f:
            json.dump([to_serializable(r) for r in model_results], f, indent=2)
        print(f"Saved: {json_path}")

        all_summary_results.extend(model_results)

        # Free model memory before loading next
        del model, model_wrapper, inference_wrapper, flow
        gc.collect()

    # =========================================================================
    # Merged summary CSV
    # =========================================================================
    total_elapsed = time.time() - sweep_start
    print(f"\n{'=' * 70}")
    print(f"ALL MODELS COMPLETE: {total_elapsed/60:.1f} min total")
    print("=" * 70)

    csv_path = DATA_DIR / "dual_sweep_summary.csv"
    df = pd.DataFrame([to_serializable(r) for r in all_summary_results])
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df)} rows)")

    print("\nDone!")


if __name__ == "__main__":
    main()
