#!/usr/bin/env python3
"""
Dimensionality Control Experiment for Length–Cacheability Correlation.

Tests whether the observed correlation between chain length and cache hit rate
(r = 0.78) can be explained by the curse of dimensionality in embedding space,
rather than by biophysical properties of proteins.

Simulates TeaCache on random linear trajectories in R^d at dimensionalities
matching real protein chain lengths, then compares with actual protein data.

Pure numpy — no model inference needed. Runs in seconds.

Usage:
    python dimensionality_control.py
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Configuration
# =============================================================================

# Dimensionality range: each residue has ~37 atom slots × 3 coordinates = 111 dims
DIMS_PER_RESIDUE = 111

# Chain lengths to simulate (covering the range in our dataset: 50–446 residues)
CHAIN_LENGTHS = [50, 75, 100, 150, 200, 250, 300, 350, 400, 446]
DIMENSIONALITIES = [l * DIMS_PER_RESIDUE for l in CHAIN_LENGTHS]

# TeaCache parameters (matching actual implementation)
N_STEPS = 500
THRESHOLD = 0.1
WARMUP_STEPS = 10
N_TRIALS = 100  # Repeats per dimensionality

OUTPUT_DIR = Path("dimensionality_control")
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SimulationResult:
    """Results for one dimensionality."""
    chain_length: int
    dimensionality: int
    mean_hit_rate: float
    std_hit_rate: float
    n_trials: int


# =============================================================================
# Simulation
# =============================================================================

def simulate_teacache_hits(d: int, n_steps: int = N_STEPS, threshold: float = THRESHOLD,
                           warmup: int = WARMUP_STEPS, n_trials: int = N_TRIALS) -> Tuple[float, float]:
    """
    Simulate TeaCache on a random linear trajectory in R^d.

    Mimics the flow-matching generative process:
    - Start from Gaussian noise x_0
    - Linear interpolation toward a random target x_target
    - Apply TeaCache decision metric at each step

    Returns (mean_hit_rate, std_hit_rate) across n_trials.
    """
    hit_rates = []

    for trial in range(n_trials):
        rng = np.random.RandomState(42 + trial)
        x0 = rng.randn(d)
        x_target = rng.randn(d)
        steps = np.linspace(1e-4, 1.0, n_steps + 1)

        prev_mod = None
        acc_diff = 0.0
        hits = 0
        misses = 0

        for i in range(n_steps):
            t = steps[i]
            x = (1 - t) * x0 + t * x_target

            # Matching TeaCache implementation: RMS norm × time modulation
            pos_norm = np.sqrt(np.mean(x ** 2))
            modulated = pos_norm * (1.0 - t + 0.1)

            if i < warmup or prev_mod is None:
                misses += 1
            else:
                diff = abs(modulated - prev_mod) / (abs(prev_mod) + 1e-6)
                acc_diff += diff
                if acc_diff >= threshold:
                    acc_diff = 0.0
                    misses += 1
                else:
                    hits += 1

            prev_mod = modulated

        hit_rates.append(hits / (hits + misses))

    return float(np.mean(hit_rates)), float(np.std(hit_rates))


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("DIMENSIONALITY CONTROL EXPERIMENT")
    print("Curse of dimensionality vs length–cacheability correlation")
    print("=" * 70)
    print(f"\nChain lengths: {CHAIN_LENGTHS}")
    print(f"Dimensionalities: {DIMENSIONALITIES}")
    print(f"Trials per dimensionality: {N_TRIALS}")
    print(f"TeaCache threshold: {THRESHOLD}")

    # =========================================================================
    # Part 1: Synthetic simulation
    # =========================================================================
    print("\n--- Part 1: Synthetic random walk simulation ---")

    sim_results: List[SimulationResult] = []

    for length, dim in zip(CHAIN_LENGTHS, DIMENSIONALITIES):
        print(f"  d={dim:6d} ({length:3d} residues) ... ", end="", flush=True)
        mean_hr, std_hr = simulate_teacache_hits(dim)
        result = SimulationResult(
            chain_length=length,
            dimensionality=dim,
            mean_hit_rate=mean_hr,
            std_hit_rate=std_hr,
            n_trials=N_TRIALS,
        )
        sim_results.append(result)
        print(f"hit rate = {mean_hr:.4f} ± {std_hr:.4f}")

    # =========================================================================
    # Part 2: Load actual protein data
    # =========================================================================
    print("\n--- Part 2: Loading actual protein data ---")

    skip_patterns_path = Path("../publication/data/skip_patterns.json")
    with open(skip_patterns_path) as f:
        skip_data = json.load(f)

    actual_lengths = []
    actual_hit_rates = []

    for pattern in skip_data["patterns"]:
        actual_lengths.append(pattern["num_residues"])
        actual_hit_rates.append(pattern["hit_rate"])

    actual_lengths = np.array(actual_lengths)
    actual_hit_rates = np.array(actual_hit_rates)

    # Compute Pearson r for actual data
    r_actual = np.corrcoef(actual_lengths, actual_hit_rates)[0, 1]
    print(f"  Actual proteins: n={len(actual_lengths)}, r={r_actual:.3f}")

    # Compute Pearson r for synthetic data
    sim_lengths = np.array([r.chain_length for r in sim_results])
    sim_hit_rates = np.array([r.mean_hit_rate for r in sim_results])
    r_synthetic = np.corrcoef(sim_lengths, sim_hit_rates)[0, 1]
    print(f"  Synthetic:        n={len(sim_lengths)}, r={r_synthetic:.3f}")

    # =========================================================================
    # Save results
    # =========================================================================
    print("\n--- Saving results ---")

    # JSON
    output = {
        "config": {
            "n_steps": N_STEPS,
            "threshold": THRESHOLD,
            "warmup_steps": WARMUP_STEPS,
            "n_trials": N_TRIALS,
            "dims_per_residue": DIMS_PER_RESIDUE,
        },
        "synthetic": [asdict(r) for r in sim_results],
        "actual": {
            "r_pearson": float(r_actual),
            "n_proteins": len(actual_lengths),
        },
        "synthetic_r_pearson": float(r_synthetic),
    }

    json_path = Path("../publication/data/dimensionality_control.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved {json_path}")

    # CSV summary
    csv_path = Path("../publication/data/dimensionality_control.csv")
    df = pd.DataFrame([asdict(r) for r in sim_results])
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    # =========================================================================
    # Generate figure
    # =========================================================================
    print("\n--- Generating figure ---")
    generate_figure(sim_results, actual_lengths, actual_hit_rates, r_actual, r_synthetic)

    print("\nDone!")


def generate_figure(sim_results: List[SimulationResult],
                    actual_lengths: np.ndarray,
                    actual_hit_rates: np.ndarray,
                    r_actual: float,
                    r_synthetic: float):
    """Generate FigS6: Dimensionality control panel."""
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
        'rose': '#CC6677',
        'grey': '#BBBBBB',
    }

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5), constrained_layout=True)

    # Actual proteins (gray scatter)
    ax.scatter(actual_lengths, actual_hit_rates * 100, c=COLORS['grey'], s=12,
               alpha=0.5, edgecolors='none', zorder=2,
               label=f'Actual proteins (r = {r_actual:.2f})')

    # Synthetic prediction (red line + shaded band)
    sim_lengths = np.array([r.chain_length for r in sim_results])
    sim_means = np.array([r.mean_hit_rate for r in sim_results]) * 100
    sim_stds = np.array([r.std_hit_rate for r in sim_results]) * 100

    ax.plot(sim_lengths, sim_means, 'o-', color=COLORS['rose'], markersize=6,
            linewidth=1.5, zorder=3,
            label=f'Synthetic prediction (r = {r_synthetic:.2f})')
    ax.fill_between(sim_lengths, sim_means - sim_stds, sim_means + sim_stds,
                    color=COLORS['rose'], alpha=0.2, linewidth=0)

    # Styling
    ax.set_xlabel('Chain length (residues)')
    ax.set_ylabel('Cache hit rate (%)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_ylim(85, 100)

    # Save
    fig_path = Path("../publication/FigS6_dimensionality_control")
    plt.savefig(f"{fig_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{fig_path}.pdf", bbox_inches='tight', facecolor='white')
    print(f"  Saved {fig_path}.png and .pdf")
    plt.close()


if __name__ == "__main__":
    main()
