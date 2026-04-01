#!/usr/bin/env python3
"""Regenerate FigS5 from dual sweep data (no model inference needed).

Reads from dual_sweep_summary.csv or falls back to individual per-model JSONs.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "publication" / "data"

# Ordered by parameter count
MODEL_ORDER = [
    'simplefold_100M', 'simplefold_360M', 'simplefold_700M',
    'simplefold_1.1B', 'simplefold_1.6B', 'simplefold_3B',
]
MODEL_LABELS = {
    'simplefold_100M': '100M', 'simplefold_360M': '360M',
    'simplefold_700M': '700M', 'simplefold_1.1B': '1.1B',
    'simplefold_1.6B': '1.6B', 'simplefold_3B': '3B',
}


def load_dual_sweep_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "dual_sweep_summary.csv"
    if csv_path.exists():
        print(f"Loading from {csv_path}")
        return pd.read_csv(csv_path)
    all_rows = []
    for json_path in sorted(DATA_DIR.glob("dual_sweep_simplefold_*.json")):
        print(f"Loading from {json_path}")
        with open(json_path) as f:
            all_rows.extend(json.load(f))
    if not all_rows:
        raise FileNotFoundError("No dual sweep data found. Run dual_sweep.py first.")
    return pd.DataFrame(all_rows)


def generate_figure(df: pd.DataFrame):
    """Generate FigS5: 3×2 grid of models, each with RMSD + TM-score."""

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['ytick.major.size'] = 3

    COLORS = {
        'uniform': '#332288',
        'adaptive': '#CC6677',
    }

    models = [m for m in MODEL_ORDER if m in df['model'].unique()]
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.2 * n_rows),
                             constrained_layout=True, squeeze=False)

    for idx, model_name in enumerate(models):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        label = MODEL_LABELS.get(model_name, model_name)
        mdf = df[df['model'] == model_name]

        # --- Uniform ---
        udf = mdf[mdf['method'] == 'uniform'].dropna(subset=['tm_score_vs_gt'])
        u_steps = udf['condition'].astype(int)
        u_agg = udf.groupby(u_steps).agg(
            tm_mean=('tm_score_vs_gt', 'mean'),
            tm_sd=('tm_score_vs_gt', 'std'),
        ).sort_index()

        # --- Adaptive ---
        adf = mdf[mdf['method'] == 'adaptive'].dropna(subset=['tm_score_vs_gt'])
        adf = adf.copy()
        adf['threshold'] = adf['condition'].astype(float)
        a_agg = adf.groupby('threshold').agg(
            steps_mean=('n_computed_steps', 'mean'),
            tm_mean=('tm_score_vs_gt', 'mean'),
            tm_sd=('tm_score_vs_gt', 'std'),
        ).sort_values('steps_mean')

        # Plot
        if not u_agg.empty:
            ax.errorbar(u_agg.index, u_agg['tm_mean'], yerr=u_agg['tm_sd'],
                        fmt='s-', color=COLORS['uniform'], markersize=4, linewidth=1.2,
                        capsize=2, capthick=0.8, label='Uniform')
        if not a_agg.empty:
            ax.errorbar(a_agg['steps_mean'], a_agg['tm_mean'], yerr=a_agg['tm_sd'],
                        fmt='o-', color=COLORS['adaptive'], markersize=5, linewidth=1.2,
                        capsize=2, capthick=0.8, label='SF-T')

        ax.set_xscale('log')
        ax.set_xlabel('Computed steps')
        ax.set_ylabel('TM-score vs. GT')
        ax.set_ylim(-0.05, 1.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(label, loc='left', fontweight='bold', fontsize=11)
        if idx == 0:
            ax.legend(fontsize=7, framealpha=0.9, loc='lower right')

    # Hide unused axes
    for idx in range(n_models, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig_path = PROJECT_ROOT / "publication" / "FigS5_uniform_vs_adaptive"
    plt.savefig(f"{fig_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{fig_path}.pdf", bbox_inches='tight', facecolor='white')
    print(f"\nSaved {fig_path}.png and .pdf")
    plt.close()

    # Also generate RMSD version
    generate_rmsd_figure(df, models)


def generate_rmsd_figure(df: pd.DataFrame, models: list):
    """Generate companion RMSD figure."""
    COLORS = {'uniform': '#332288', 'adaptive': '#CC6677'}
    n_cols = 3
    n_rows = (len(models) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.2 * n_rows),
                             constrained_layout=True, squeeze=False)

    for idx, model_name in enumerate(models):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        label = MODEL_LABELS.get(model_name, model_name)
        mdf = df[df['model'] == model_name]

        udf = mdf[mdf['method'] == 'uniform'].dropna(subset=['rmsd_vs_gt'])
        u_steps = udf['condition'].astype(int)
        u_agg = udf.groupby(u_steps).agg(
            rmsd_mean=('rmsd_vs_gt', 'mean'),
            rmsd_sd=('rmsd_vs_gt', 'std'),
        ).sort_index()

        adf = mdf[mdf['method'] == 'adaptive'].dropna(subset=['rmsd_vs_gt'])
        adf = adf.copy()
        adf['threshold'] = adf['condition'].astype(float)
        a_agg = adf.groupby('threshold').agg(
            steps_mean=('n_computed_steps', 'mean'),
            rmsd_mean=('rmsd_vs_gt', 'mean'),
            rmsd_sd=('rmsd_vs_gt', 'std'),
        ).sort_values('steps_mean')

        if not u_agg.empty:
            ax.errorbar(u_agg.index, u_agg['rmsd_mean'], yerr=u_agg['rmsd_sd'],
                        fmt='s-', color=COLORS['uniform'], markersize=4, linewidth=1.2,
                        capsize=2, capthick=0.8, label='Uniform')
        if not a_agg.empty:
            ax.errorbar(a_agg['steps_mean'], a_agg['rmsd_mean'], yerr=a_agg['rmsd_sd'],
                        fmt='o-', color=COLORS['adaptive'], markersize=5, linewidth=1.2,
                        capsize=2, capthick=0.8, label='SF-T')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Computed steps')
        ax.set_ylabel('RMSD vs. GT (Å)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(label, loc='left', fontweight='bold', fontsize=11)
        if idx == 0:
            ax.legend(fontsize=7, framealpha=0.9)

    for idx in range(len(models), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig_path = PROJECT_ROOT / "publication" / "FigS5_rmsd_uniform_vs_adaptive"
    plt.savefig(f"{fig_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{fig_path}.pdf", bbox_inches='tight', facecolor='white')
    print(f"Saved {fig_path}.png and .pdf")
    plt.close()


if __name__ == "__main__":
    df = load_dual_sweep_data()
    print(f"\nLoaded {len(df)} results across {df['model'].nunique()} models")
    print(f"Methods: {df['method'].value_counts().to_dict()}")
    generate_figure(df)
