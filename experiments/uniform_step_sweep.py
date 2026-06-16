#!/usr/bin/env python3
"""
Uniform Step-Skipping Sweep vs SF-T (Adaptive Caching).

Compares uniform step-skipping (naive baseline) against SF-T's adaptive
caching across multiple compute budgets, to demonstrate that adaptive caching
consistently outperforms naive step reduction.

Usage:
    python uniform_step_sweep.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

EXPERIMENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = EXPERIMENT_DIR.parent
os.chdir(EXPERIMENT_DIR)
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'simplefold'))

import mlx.core as mx


# =============================================================================
# Configuration
# =============================================================================

# Step counts chosen to match mean_computed steps from the threshold sweep:
#   τ=1.0 → 12, τ=0.5 → 15, τ=0.4 → 16, τ=0.2 → 24, τ=0.1 → 36,
#   τ=0.05 → 59, τ=0.01 → 171
# Plus additional reference points at 100 and 250
# 500-step baseline already exists from threshold sweep (τ=0.0)
STEP_COUNTS = [12, 15, 16, 24, 36, 59, 100, 171, 250]

N_PROTEINS = 300

# Ground truth PDB directory (CATH domain PDBs)
GT_PDB_DIR = Path("/Users/gjt4/Documents/GitHub/newton/dompdb")

OUTPUT_DIR = Path("uniform_step_sweep")
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class UniformResult:
    """Results for one protein at one step count."""
    name: str
    num_steps: int
    num_residues: int
    inference_time: float
    rmsd_vs_gt: Optional[float] = None
    tm_score_vs_gt: Optional[float] = None


# =============================================================================
# Utility Functions (same as threshold_sweep.py)
# =============================================================================

def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD between two coordinate sets."""
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=-1)))


def kabsch_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD after optimal superposition using Kabsch algorithm."""
    if hasattr(coords1, 'tolist'):
        coords1 = np.array(coords1)
    if hasattr(coords2, 'tolist'):
        coords2 = np.array(coords2)

    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)

    H = c1.T @ c2
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    c1_rotated = c1 @ R
    return compute_rmsd(c1_rotated, c2)


def extract_ca_coords(coords: mx.array, atom_pad_mask: mx.array, n_residues: int) -> np.ndarray:
    """Extract CA coordinates from full atom coords.

    Uses the known sequence length to avoid counting padded atoms.
    CA is at index 1 within each 5-atom residue block.
    """
    coords_np = np.array(coords[0])
    ca_indices = [i * 5 + 1 for i in range(n_residues)]
    return coords_np[ca_indices]


def load_fasta_sequences(fasta_path: Path) -> List[Tuple[str, str]]:
    """Load sequences from FASTA file."""
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


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Align P onto Q using Kabsch algorithm, return aligned P."""
    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)

    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return P_c @ R + Q.mean(axis=0)


def compute_tm_score(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute TM-score (Zhang & Skolnick, 2004)."""
    P_aligned = kabsch_align(P, Q)
    L = len(Q)
    d0 = max(1.24 * (L - 15) ** (1/3) - 1.8, 0.5)
    distances = np.sqrt(np.sum((P_aligned - Q) ** 2, axis=1))
    return float(np.sum(1 / (1 + (distances / d0) ** 2)) / L)


def load_gt_coords(pdb_path: Path) -> Optional[np.ndarray]:
    """Load CA coordinates from ground truth PDB."""
    ca_coords = []
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_coords.append([x, y, z])
        return np.array(ca_coords) if ca_coords else None
    except:
        return None


# =============================================================================
# Main Sweep
# =============================================================================

def main():
    print("=" * 70)
    print("UNIFORM STEP-SKIPPING SWEEP")
    print("Comparing uniform step reduction vs SF-T adaptive caching")
    print("=" * 70)
    print(f"\nStep counts: {STEP_COUNTS}")
    print(f"N proteins: {N_PROTEINS}")

    # Initialize model
    print("\nLoading model...")
    from wrapper import ModelWrapper, InferenceWrapper
    from model.mlx.sampler import EMSampler as EMSamplerMLX
    from model.flow import LinearPath

    # wrapper.py uses relative paths for configs/ — must be in the simplefold package dir
    simplefold_dir = PROJECT_ROOT / 'src' / 'simplefold'
    os.chdir(simplefold_dir)

    model_wrapper = ModelWrapper(
        simplefold_model='simplefold_100M',
        plddt=False,
        ckpt_dir=str(PROJECT_ROOT / 'artifacts'),
        backend='mlx',
    )

    inference_wrapper = InferenceWrapper(
        output_dir=str(EXPERIMENT_DIR / OUTPUT_DIR),
        prediction_dir='predictions',
        num_steps=500,
        nsample_per_protein=1,
        tau=0.1,
        device='cpu',
        backend='mlx',
    )

    model = model_wrapper.from_pretrained_folding_model()
    flow = LinearPath()
    print("Model loaded!")

    # Restore CWD to experiment dir
    os.chdir(EXPERIMENT_DIR)

    # Load sequences
    data_dir = PROJECT_ROOT / 'publication' / 'data'
    fasta_path = data_dir / "diverse_cath_300.fasta"
    print(f"\nLoading sequences from {fasta_path}...")
    sequences = load_fasta_sequences(fasta_path)[:N_PROTEINS]
    print(f"Loaded {len(sequences)} sequences")

    # Load GT names (PDBs live in GT_PDB_DIR as {name}.pdb)
    with open(data_dir / "diverse_cath_300.json") as f:
        gt_names = [p['name'] for p in json.load(f)]

    # Results storage
    all_results: List[UniformResult] = []
    all_ca_coords: Dict[str, np.ndarray] = {}  # Save all CA coords for reanalysis

    # Run sweep
    total_start = time.time()

    for prot_idx, (name, sequence) in enumerate(sequences):
        print(f"\n{'='*70}")
        print(f"[{prot_idx+1:3d}/{len(sequences)}] {name} ({len(sequence)} residues)")
        print("=" * 70)

        # Process input once
        batch, structure, record = inference_wrapper.process_input(sequence)

        # Load GT from dompdb directory
        gt_pdb = GT_PDB_DIR / f"{name}.pdb"
        gt_coords = load_gt_coords(gt_pdb) if gt_pdb.exists() else None

        for n_steps in STEP_COUNTS:
            # Use deterministic seed for reproducibility (same per protein across step counts)
            mx.random.seed(42 + prot_idx)
            noise = mx.random.normal(batch['coords'].shape)

            # Create sampler with this step count
            sampler = EMSamplerMLX(
                num_timesteps=n_steps,
                t_start=1e-4,
                tau=0.1,
                log_timesteps=True,
                w_cutoff=0.99,
            )

            # Run inference
            start_time = time.time()
            out_dict = sampler.sample(model, flow, noise, batch)
            # Postprocess: center + scale back to Angstroms (×16)
            out_dict = inference_wrapper.processor.postprocess(out_dict, batch)
            mx.eval(out_dict['denoised_coords'])
            elapsed = time.time() - start_time

            # Extract CA coords (use sequence length, not padded atom count)
            ca_coords = extract_ca_coords(
                out_dict['denoised_coords'],
                batch['atom_pad_mask'],
                n_residues=len(sequence)
            )

            # Save CA coords for reanalysis
            all_ca_coords[f"{name}_steps{n_steps}"] = ca_coords

            # Compute metrics vs ground truth
            rmsd_vs_gt = None
            tm_score_vs_gt = None
            if gt_coords is None:
                if n_steps == STEP_COUNTS[0]:  # Only print once per protein
                    print(f"    WARNING: No GT coords for {name} (pdb={gt_pdb}, exists={gt_pdb.exists()})")
            else:
                if len(ca_coords) != len(gt_coords):
                    if n_steps == STEP_COUNTS[0]:
                        print(f"    WARNING: CA count mismatch for {name}: pred={len(ca_coords)} gt={len(gt_coords)}")
                else:
                    try:
                        rmsd_vs_gt = float(kabsch_rmsd(ca_coords, gt_coords))
                        tm_score_vs_gt = compute_tm_score(ca_coords, gt_coords)
                    except Exception as e:
                        if n_steps == STEP_COUNTS[0]:
                            print(f"    WARNING: Metric computation failed for {name}: {e}")

            result = UniformResult(
                name=name,
                num_steps=n_steps,
                num_residues=len(sequence),
                inference_time=elapsed,
                rmsd_vs_gt=rmsd_vs_gt,
                tm_score_vs_gt=tm_score_vs_gt,
            )
            all_results.append(result)

            rmsd_str = f"{rmsd_vs_gt:.2f}" if rmsd_vs_gt is not None else "N/A"
            tm_str = f"{tm_score_vs_gt:.3f}" if tm_score_vs_gt is not None else "N/A"
            print(f"  {n_steps:4d} steps: {elapsed:5.2f}s  RMSD_gt={rmsd_str:>8}  TM={tm_str:>6}")

            # Free inner memory and clear cache
            del noise, sampler, out_dict
            import gc
            gc.collect()
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()

        # Free outer loop variables
        del batch, structure, record, gt_coords
        import gc
        gc.collect()

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"Total sweep time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    # =========================================================================
    # Save results
    # =========================================================================
    print("\n--- Saving results ---")

    # JSON
    json_path = data_dir / "uniform_step_sweep.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"  Saved {json_path}")

    # Save all CA coordinates for reanalysis without re-running inference
    coords_path = data_dir / "uniform_step_sweep_coords.npz"
    np.savez_compressed(coords_path, **all_ca_coords)
    print(f"  Saved {coords_path} ({len(all_ca_coords)} structures)")

    # =========================================================================
    # Build comparison table: uniform vs adaptive
    # =========================================================================
    print("\n--- Building comparison table ---")

    # Load SF-T (adaptive) results:
    #   - benchmark has GT metrics (rmsd, tm_score) for τ=0.0 and τ=0.1
    #   - threshold sweep has computed step counts for all thresholds
    with open(data_dir / "cath_benchmark_full.json") as f:
        bench_data = json.load(f)
    with open(data_dir / "threshold_sweep.json") as f:
        sweep_data = json.load(f)

    # Get mean computed steps per threshold from the sweep
    sweep_steps = {}
    for r in sweep_data:
        th = r['threshold']
        sweep_steps.setdefault(th, []).append(r['n_computed_steps'])
    mean_steps_by_thresh = {th: int(np.mean(steps)) for th, steps in sweep_steps.items()}

    # Aggregate adaptive results from the benchmark (simplefold_100M only)
    adaptive_by_thresh = {}
    for r in bench_data:
        if r['model'] != 'simplefold_100M':
            continue
        if r.get('error'):
            continue
        th = r['teacache_threshold']
        if th not in adaptive_by_thresh:
            computed = mean_steps_by_thresh.get(th, 500 if th == 0.0 else None)
            adaptive_by_thresh[th] = {'steps': computed, 'rmsds_gt': [], 'tm_scores': []}
        if r.get('rmsd') is not None:
            adaptive_by_thresh[th]['rmsds_gt'].append(r['rmsd'])
        if r.get('tm_score') is not None:
            adaptive_by_thresh[th]['tm_scores'].append(r['tm_score'])

    # Aggregate uniform results by step count
    uniform_by_steps = {}
    for r in all_results:
        ns = r.num_steps
        if ns not in uniform_by_steps:
            uniform_by_steps[ns] = {'rmsds_gt': [], 'tm_scores': [], 'times': []}
        if r.rmsd_vs_gt is not None:
            uniform_by_steps[ns]['rmsds_gt'].append(r.rmsd_vs_gt)
        if r.tm_score_vs_gt is not None:
            uniform_by_steps[ns]['tm_scores'].append(r.tm_score_vs_gt)
        uniform_by_steps[ns]['times'].append(r.inference_time)

    # Build comparison CSV
    comparison_rows = []

    # Uniform rows
    for ns in sorted(uniform_by_steps.keys()):
        d = uniform_by_steps[ns]
        comparison_rows.append({
            'method': 'uniform',
            'computed_steps': ns,
            'mean_rmsd_vs_gt': float(np.mean(d['rmsds_gt'])) if d['rmsds_gt'] else None,
            'std_rmsd_vs_gt': float(np.std(d['rmsds_gt'])) if d['rmsds_gt'] else None,
            'mean_tm_score': float(np.mean(d['tm_scores'])) if d['tm_scores'] else None,
            'std_tm_score': float(np.std(d['tm_scores'])) if d['tm_scores'] else None,
            'mean_time': float(np.mean(d['times'])),
        })

    # Adaptive rows
    for th in sorted(adaptive_by_thresh.keys()):
        d = adaptive_by_thresh[th]
        comparison_rows.append({
            'method': 'adaptive',
            'computed_steps': d['steps'],
            'mean_rmsd_vs_gt': float(np.mean(d['rmsds_gt'])) if d['rmsds_gt'] else None,
            'std_rmsd_vs_gt': float(np.std(d['rmsds_gt'])) if d['rmsds_gt'] else None,
            'mean_tm_score': float(np.mean(d['tm_scores'])) if d['tm_scores'] else None,
            'std_tm_score': float(np.std(d['tm_scores'])) if d['tm_scores'] else None,
            'threshold': th,
        })

    csv_path = data_dir / "uniform_vs_adaptive.csv"
    df = pd.DataFrame(comparison_rows)
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    # =========================================================================
    # Generate figure
    # =========================================================================
    print("\n--- Generating figure ---")
    generate_figure(uniform_by_steps, adaptive_by_thresh)

    print("\nDone!")


def generate_figure(uniform_by_steps: dict, adaptive_by_thresh: dict):
    """Generate FigS5: Uniform vs Adaptive comparison."""
    import matplotlib.pyplot as plt

    # Publication style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4

    COLORS = {
        'blue': '#332288',
        'cyan': '#88CCEE',
        'teal': '#44AA99',
        'rose': '#CC6677',
        'sand': '#DDCC77',
    }

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)

    # --- Panel A: RMSD vs ground truth ---
    ax = axes[0]

    u_gt_means, u_gt_stds, u_gt_steps = [], [], []
    for s in sorted(uniform_by_steps.keys()):
        d = uniform_by_steps[s]['rmsds_gt']
        if d:
            u_gt_steps.append(s)
            u_gt_means.append(np.mean(d))
            u_gt_stds.append(np.std(d))

    ax.errorbar(u_gt_steps, u_gt_means, yerr=u_gt_stds,
                fmt='s-', color=COLORS['blue'], markersize=5, linewidth=1.5,
                capsize=3, capthick=1, label='Uniform step-skipping')

    # Adaptive data points (from benchmark: τ=0.0 at 500 steps, τ=0.1 at ~36 steps)
    a_gt_means, a_gt_stds, a_gt_steps = [], [], []
    a_tm_means, a_tm_stds, a_tm_steps = [], [], []
    for t in sorted(adaptive_by_thresh.keys()):
        d = adaptive_by_thresh[t]
        steps = d['steps']
        if steps is None:
            continue
        if d['rmsds_gt']:
            a_gt_steps.append(steps)
            a_gt_means.append(np.mean(d['rmsds_gt']))
            a_gt_stds.append(np.std(d['rmsds_gt']))
        if d['tm_scores']:
            a_tm_steps.append(steps)
            a_tm_means.append(np.mean(d['tm_scores']))
            a_tm_stds.append(np.std(d['tm_scores']))

    ax.errorbar(a_gt_steps, a_gt_means, yerr=a_gt_stds,
                fmt='o-', color=COLORS['rose'], markersize=6, linewidth=1.5,
                capsize=3, capthick=1, label='SF-T (adaptive)')

    ax.set_xscale('log')
    ax.set_xlabel('Computed steps')
    ax.set_ylabel('RMSD vs. ground truth (Å)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_title('A', loc='left', fontweight='bold', fontsize=12)

    # --- Panel B: TM-score vs ground truth ---
    ax = axes[1]

    u_tm_means, u_tm_stds, u_tm_steps = [], [], []
    for s in sorted(uniform_by_steps.keys()):
        d = uniform_by_steps[s]['tm_scores']
        if d:
            u_tm_steps.append(s)
            u_tm_means.append(np.mean(d))
            u_tm_stds.append(np.std(d))

    ax.errorbar(u_tm_steps, u_tm_means, yerr=u_tm_stds,
                fmt='s-', color=COLORS['blue'], markersize=5, linewidth=1.5,
                capsize=3, capthick=1, label='Uniform step-skipping')

    ax.errorbar(a_tm_steps, a_tm_means, yerr=a_tm_stds,
                fmt='o-', color=COLORS['rose'], markersize=6, linewidth=1.5,
                capsize=3, capthick=1, label='SF-T (adaptive)')

    ax.set_xscale('log')
    ax.set_xlabel('Computed steps')
    ax.set_ylabel('TM-score vs. ground truth')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.9, loc='lower right')
    ax.set_title('B', loc='left', fontweight='bold', fontsize=12)

    # Save
    fig_path = PROJECT_ROOT / "publication" / "FigS5_uniform_vs_adaptive"
    plt.savefig(f"{fig_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{fig_path}.pdf", bbox_inches='tight', facecolor='white')
    print(f"  Saved {fig_path}.png and .pdf")
    plt.close()


if __name__ == "__main__":
    main()
