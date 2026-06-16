#!/usr/bin/env python3
"""
TeaCache Threshold Sweep for Figure 1 Validation Data.

Tests multiple thresholds to generate:
- Panel A: RMSD baseline vs TeaCache (quality preservation)
- Panel B: Speedup vs threshold
- Panel C: Cache hit rate vs threshold
- Panel D: Representative structures

Usage:
    python threshold_sweep.py
"""

import os
import sys
import json
import time
import numpy as np
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

# Thresholds to test: 0.0 = baseline (no caching)
THRESHOLDS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 1.0]

# Use subset for faster testing, or full 300 for publication
N_PROTEINS = 300  # Full set for publication

# Ground truth PDB directory (CATH domain PDBs)
GT_PDB_DIR = Path("/Users/gjt4/Documents/GitHub/newton/dompdb")

OUTPUT_DIR = Path("threshold_sweep")
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SweepResult:
    """Results for one protein at one threshold."""
    name: str
    threshold: float
    num_residues: int
    inference_time: float
    cache_hit_rate: float
    n_computed_steps: int
    rmsd_vs_baseline: Optional[float] = None
    rmsd_vs_gt: Optional[float] = None
    tm_score_vs_gt: Optional[float] = None


# =============================================================================
# Utility Functions
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


# =============================================================================
# TeaCache Sampler with Tracking
# =============================================================================

class TeaCacheSamplerWithStats:
    """TeaCache sampler that tracks statistics."""

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

        # Stats
        self.n_computed = 0
        self.n_cached = 0

    def _compute_modulated_input(self, noised_pos: mx.array, t: mx.array) -> float:
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

    def sample(self, model_fn, flow, noise: mx.array, batch: Dict, verbose: bool = False) -> mx.array:
        from tqdm import tqdm
        from einops.array_api import repeat
        from utils.mlx_utils import center_random_augmentation

        y = noise
        cached_output = None
        prev_modulated_input = None
        accumulated_diff = 0.0

        self.n_computed = 0
        self.n_cached = 0

        iterator = tqdm(range(self.num_timesteps), desc=f"τ={self.threshold}") if verbose else range(self.num_timesteps)

        for i in iterator:
            t = self.steps[i]
            t_next = self.steps[i + 1]
            dt = t_next - t
            eps = mx.random.normal(y.shape)

            y = center_random_augmentation(
                y, batch["atom_pad_mask"],
                augmentation=False, centering=True,
            )

            modulated_input = self._compute_modulated_input(y, t)

            # Cache decision
            recompute = False
            if self.threshold == 0.0:
                # No caching - always compute (baseline)
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
    def cache_hit_rate(self) -> float:
        total = self.n_computed + self.n_cached
        return self.n_cached / total if total > 0 else 0.0


# =============================================================================
# Main Sweep
# =============================================================================

def main():
    print("=" * 70)
    print("TEACACHE THRESHOLD SWEEP")
    print("Generating Figure 1 validation data")
    print("=" * 70)
    print(f"\nThresholds: {THRESHOLDS}")
    print(f"N proteins: {N_PROTEINS}")

    # Initialize model
    print("\nLoading model...")
    from wrapper import ModelWrapper, InferenceWrapper

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
    all_results: List[SweepResult] = []
    baseline_coords: Dict[str, np.ndarray] = {}  # Store baseline (τ=0) coords
    all_ca_coords: Dict[str, np.ndarray] = {}  # Save all CA coords for reanalysis

    # Run sweep
    start_time = time.time()

    for prot_idx, (name, sequence) in enumerate(sequences):
        print(f"\n{'='*70}")
        print(f"[{prot_idx+1:3d}/{len(sequences)}] {name} ({len(sequence)} residues)")
        print("=" * 70)

        # Process input once
        batch, structure, record = inference_wrapper.process_input(sequence)

        # Load GT from dompdb directory
        gt_pdb = GT_PDB_DIR / f"{name}.pdb"
        gt_coords = load_gt_coords(gt_pdb) if gt_pdb.exists() else None

        for threshold in THRESHOLDS:
            # Use deterministic seed for reproducibility
            mx.random.seed(42 + prot_idx)
            noise = mx.random.normal(batch['coords'].shape)

            # Create sampler
            sampler = TeaCacheSamplerWithStats(
                num_timesteps=500,
                threshold=threshold,
                tau=0.1,
                warmup_steps=10
            )

            # Run inference with timing
            t0 = time.time()
            final_coords = sampler.sample(
                model_fn=model,
                flow=inference_wrapper.flow,
                noise=noise,
                batch=batch,
                verbose=False
            )
            # Postprocess: center + scale back to Angstroms (×16)
            out_dict_tc = {'denoised_coords': final_coords}
            out_dict_tc = inference_wrapper.processor.postprocess(out_dict_tc, batch)
            final_coords = out_dict_tc['denoised_coords']
            mx.eval(final_coords)
            inference_time = time.time() - t0

            # Extract CA coords (use sequence length, not padded atom count)
            ca_coords = extract_ca_coords(final_coords, batch['atom_pad_mask'],
                                          n_residues=len(sequence))

            # Save CA coords for reanalysis
            all_ca_coords[f"{name}_tau{threshold}"] = ca_coords

            # Store baseline coords
            if threshold == 0.0:
                baseline_coords[name] = ca_coords.copy()

            # Compute RMSD vs baseline
            rmsd_vs_baseline = None
            if threshold > 0.0 and name in baseline_coords:
                rmsd_vs_baseline = kabsch_rmsd(ca_coords, baseline_coords[name])

            # Compute metrics vs ground truth
            rmsd_vs_gt = None
            tm_score_vs_gt = None
            if gt_coords is not None and len(gt_coords) == len(ca_coords):
                try:
                    rmsd_vs_gt = float(kabsch_rmsd(ca_coords, gt_coords))
                    tm_score_vs_gt = compute_tm_score(ca_coords, gt_coords)
                except Exception as e:
                    if threshold == 0.0:
                        print(f"    WARNING: Metric computation failed for {name}: {e}")

            result = SweepResult(
                name=name,
                threshold=threshold,
                num_residues=len(sequence),
                inference_time=inference_time,
                cache_hit_rate=sampler.cache_hit_rate,
                n_computed_steps=sampler.n_computed,
                rmsd_vs_baseline=rmsd_vs_baseline,
                rmsd_vs_gt=rmsd_vs_gt,
                tm_score_vs_gt=tm_score_vs_gt,
            )

            all_results.append(result)

            rmsd_gt_str = f"RMSD_gt={rmsd_vs_gt:.2f}" if rmsd_vs_gt is not None else "RMSD_gt=N/A"
            tm_str = f"TM={tm_score_vs_gt:.3f}" if tm_score_vs_gt is not None else "TM=N/A"
            print(f"  τ={threshold:.2f}: {sampler.n_computed:3d} steps | "
                  f"hit={sampler.cache_hit_rate:.1%} | time={inference_time:.1f}s | "
                  f"{rmsd_gt_str} | {tm_str}"
                  + (f" | ΔRMSD={rmsd_vs_baseline:.2f}Å" if rmsd_vs_baseline else ""))

            # Free inner memory and clear cache
            del noise, sampler, final_coords, out_dict_tc
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

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE in {total_time/60:.1f} minutes")
    print("=" * 70)

    # Save results (convert numpy types to native Python for JSON)
    json_path = data_dir / "threshold_sweep.json"

    def to_serializable(r):
        d = asdict(r)
        for k, v in d.items():
            if hasattr(v, 'item'):  # numpy scalar
                d[k] = v.item()
            elif isinstance(v, (np.floating, np.integer)):
                d[k] = float(v) if isinstance(v, np.floating) else int(v)
        return d

    with open(json_path, 'w') as f:
        json.dump([to_serializable(r) for r in all_results], f, indent=2)
    print(f"Saved: {json_path}")

    # Save all CA coordinates for reanalysis without re-running inference
    coords_path = data_dir / "threshold_sweep_coords.npz"
    np.savez_compressed(coords_path, **all_ca_coords)
    print(f"Saved: {coords_path} ({len(all_ca_coords)} structures)")

    # ==========================================================================
    # Generate Figure 1
    # ==========================================================================

    print("\nGenerating Figure 1...")
    generate_figure_1(all_results, OUTPUT_DIR)

    print("\n✅ THRESHOLD SWEEP COMPLETE!")


def generate_figure_1(results: List[SweepResult], output_dir: Path):
    """Generate publication Figure 1: TeaCache Validation."""
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Figure 1: TeaCache Validation on SimpleFold 100M\n(n={} CATH proteins, 500 diffusion steps)'.format(
        len(set(r.name for r in results))), fontsize=14, fontweight='bold')

    # Aggregate by threshold
    thresholds = sorted(set(r.threshold for r in results))

    # Exclude baseline for non-baseline stats
    teacache_thresholds = [t for t in thresholds if t > 0]

    # Stats per threshold
    stats = {}
    for t in thresholds:
        t_results = [r for r in results if r.threshold == t]
        stats[t] = {
            'mean_time': np.mean([r.inference_time for r in t_results]),
            'std_time': np.std([r.inference_time for r in t_results]),
            'mean_hit_rate': np.mean([r.cache_hit_rate for r in t_results]),
            'std_hit_rate': np.std([r.cache_hit_rate for r in t_results]),
            'mean_computed': np.mean([r.n_computed_steps for r in t_results]),
        }
        if t > 0:
            rmsds = [r.rmsd_vs_baseline for r in t_results if r.rmsd_vs_baseline is not None]
            stats[t]['mean_rmsd_vs_baseline'] = np.mean(rmsds) if rmsds else None
            stats[t]['std_rmsd_vs_baseline'] = np.std(rmsds) if rmsds else None

    # ==========================================================================
    # Panel A: RMSD vs Baseline
    # ==========================================================================
    ax = axes[0, 0]

    # Scatter plot for all proteins at threshold 0.1 (our default)
    t01_results = [r for r in results if r.threshold == 0.1 and r.rmsd_vs_baseline is not None]
    if t01_results:
        rmsd_values = [r.rmsd_vs_baseline for r in t01_results]
        sizes = [r.num_residues for r in t01_results]

        scatter = ax.scatter(sizes, rmsd_values, c='steelblue', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        ax.axhline(np.mean(rmsd_values), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(rmsd_values):.3f}Å')
        ax.fill_between([min(sizes), max(sizes)],
                        np.mean(rmsd_values) - np.std(rmsd_values),
                        np.mean(rmsd_values) + np.std(rmsd_values),
                        color='red', alpha=0.1)

    ax.set_xlabel('Protein Length (residues)', fontsize=11)
    ax.set_ylabel('RMSD vs Baseline (Å)', fontsize=11)
    ax.set_title('A) Quality Preservation: TeaCache vs No-Cache\n(threshold=0.1)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # ==========================================================================
    # Panel B: Speedup vs Threshold
    # ==========================================================================
    ax = axes[0, 1]

    baseline_time = stats[0.0]['mean_time']
    speedups = [baseline_time / stats[t]['mean_time'] for t in teacache_thresholds]

    ax.plot(teacache_thresholds, speedups, 'o-', color='darkgreen', linewidth=2, markersize=10)

    # Highlight optimal threshold
    opt_idx = np.argmax(speedups)
    ax.scatter([teacache_thresholds[opt_idx]], [speedups[opt_idx]], s=200, c='gold',
               edgecolors='black', linewidth=2, zorder=5, label=f'Best: τ={teacache_thresholds[opt_idx]} ({speedups[opt_idx]:.1f}x)')

    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('TeaCache Threshold (τ)', fontsize=11)
    ax.set_ylabel('Speedup vs Baseline', fontsize=11)
    ax.set_title('B) Speedup vs Threshold', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # ==========================================================================
    # Panel C: Cache Hit Rate vs Threshold
    # ==========================================================================
    ax = axes[1, 0]

    hit_rates = [stats[t]['mean_hit_rate'] * 100 for t in teacache_thresholds]
    hit_std = [stats[t]['std_hit_rate'] * 100 for t in teacache_thresholds]

    ax.errorbar(teacache_thresholds, hit_rates, yerr=hit_std, fmt='o-', color='purple',
                linewidth=2, markersize=10, capsize=5, capthick=2)

    # Highlight threshold 0.1
    if 0.1 in teacache_thresholds:
        idx = teacache_thresholds.index(0.1)
        ax.scatter([0.1], [hit_rates[idx]], s=200, c='orange', edgecolors='black',
                   linewidth=2, zorder=5, label=f'τ=0.1: {hit_rates[idx]:.1f}%')

    ax.set_xlabel('TeaCache Threshold (τ)', fontsize=11)
    ax.set_ylabel('Cache Hit Rate (%)', fontsize=11)
    ax.set_title('C) Cache Hit Rate vs Threshold', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_ylim(0, 100)

    # ==========================================================================
    # Panel D: Summary Statistics
    # ==========================================================================
    ax = axes[1, 1]
    ax.axis('off')

    # Compute key numbers
    t01_stats = stats.get(0.1, {})
    baseline_stats = stats.get(0.0, {})

    summary_text = f"""
    KEY FINDINGS
    ════════════════════════════════════════════

    TeaCache threshold = 0.1:
    • Cache hit rate: {t01_stats.get('mean_hit_rate', 0)*100:.1f}%
    • Computed steps: {t01_stats.get('mean_computed', 500):.0f} / 500
    • Mean time: {t01_stats.get('mean_time', 0):.1f}s vs {baseline_stats.get('mean_time', 0):.1f}s (baseline)
    • Speedup: {baseline_stats.get('mean_time', 1) / t01_stats.get('mean_time', 1):.1f}x

    Quality preservation:
    • RMSD vs baseline: {t01_stats.get('mean_rmsd_vs_baseline', 0):.3f} ± {t01_stats.get('std_rmsd_vs_baseline', 0):.3f} Å

    Comparison to video diffusion:
    • Video: ~50% cache hit at τ=0.4
    • Protein: ~{t01_stats.get('mean_hit_rate', 0)*100:.0f}% cache hit at τ=0.1

    ════════════════════════════════════════════
    Ramachandran constraints enable extreme caching!
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fig_path = output_dir / "figure1_teacache_validation.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fig_path}")

    # Also save summary table
    csv_path = output_dir / "threshold_summary.csv"
    with open(csv_path, 'w') as f:
        f.write("threshold,mean_time,std_time,mean_hit_rate,std_hit_rate,mean_computed,speedup,mean_rmsd_vs_baseline\n")
        for t in thresholds:
            s = stats[t]
            speedup = baseline_stats['mean_time'] / s['mean_time'] if t > 0 else 1.0
            rmsd = s.get('mean_rmsd_vs_baseline', '')
            f.write(f"{t},{s['mean_time']:.2f},{s['std_time']:.2f},{s['mean_hit_rate']:.4f},{s['std_hit_rate']:.4f},{s['mean_computed']:.0f},{speedup:.2f},{rmsd}\n")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
