import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import json

### Set up sans-serif font and publication-quality defaults
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4

# Colorblind-safe palette (Paul Tol muted)
COLORS = {
    'blue': '#332288',
    'cyan': '#88CCEE',
    'teal': '#44AA99',
    'green': '#117733',
    'olive': '#999933',
    'sand': '#DDCC77',
    'rose': '#CC6677',
    'wine': '#882255',
    'purple': '#AA4499',
    'grey': '#BBBBBB'
}

# Load data
benchmark_df = pd.read_csv('./data/cath_benchmark_full.csv')
threshold_df = pd.read_csv('./data/threshold_summary.csv')

with open('./data/skip_patterns.json') as f:
    skip_data = json.load(f)

with open('./data/threshold_sweep.json') as f:
    threshold_sweep_data = json.load(f)

# Calculate model scaling stats (all 6 models)
models = ['simplefold_100M', 'simplefold_360M', 'simplefold_700M', 'simplefold_1.1B', 'simplefold_1.6B', 'simplefold_3B']
model_labels = ['100M', '360M', '700M', '1.1B', '1.6B', '3B']
model_params = [100e6, 360e6, 700e6, 1.1e9, 1.6e9, 3e9]  # For log x-axis

baseline_times = {}
cached_stats = {}

for model in models:
    baseline = benchmark_df[(benchmark_df['model'] == model) & (benchmark_df['teacache_threshold'] == 0.0)]
    cached = benchmark_df[(benchmark_df['model'] == model) & (benchmark_df['teacache_threshold'] == 0.1)]

    baseline_times[model] = baseline['time_s'].mean()
    cached_stats[model] = {
        'time': cached['time_s'].mean(),
        'time_std': cached['time_s'].std(),
        'hit_rate': cached['cache_hit'].mean(),
        'hit_rate_std': cached['cache_hit'].std()
    }

speedups = [baseline_times[m] / cached_stats[m]['time'] for m in models]
speedup_errors = [cached_stats[m]['time_std'] / cached_stats[m]['time'] * speedups[i] for i, m in enumerate(models)]
hit_rates = [cached_stats[m]['hit_rate'] * 100 for m in models]
hit_rate_errors = [cached_stats[m]['hit_rate_std'] * 100 for m in models]

# Create figure with 4 subpanels
fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
fig.subplots_adjust(hspace=0.35, wspace=0.35)

### Panel A: Speedup vs model size (dot plot with log x-axis)
ax = axes[0, 0]
ax.errorbar(model_params, speedups, yerr=speedup_errors, fmt='o-', color=COLORS['blue'],
            markersize=7, linewidth=1.5, capsize=3, capthick=1, markeredgecolor='white', markeredgewidth=0.5)
ax.set_xscale('log')
ax.set_xlabel('Parameters')
ax.set_ylabel('Speedup (x)')
ax.set_ylim(0, 20)
ax.set_xlim(7e7, 5e9)
ax.axhline(y=1, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
# Custom x-tick labels
ax.set_xticks(model_params, model_labels, rotation=45)
ax.set_xticklabels(model_labels, fontsize=8)
# Add value labels
for px, v, e in zip(model_params, speedups, speedup_errors):
    ax.text(px, v + e + 0.6, f'{v:.1f}x', ha='center', va='bottom', fontsize=7)
ax.text(-0.15, 1.15, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: Cache hit rate vs model size (dot plot with log x-axis to match A)
ax = axes[0, 1]
ax.errorbar(model_params, hit_rates, yerr=hit_rate_errors, fmt='o-', color=COLORS['teal'],
            markersize=7, linewidth=1.5, capsize=3, capthick=1, markeredgecolor='white', markeredgewidth=0.5)
ax.set_xscale('log')
ax.set_xlabel('Parameters')
ax.set_ylabel('Cache hit rate (%)')
ax.set_ylim(90, 96)
ax.set_xlim(7e7, 5e9)
# Custom x-tick labels
ax.set_xticks(model_params, model_labels, rotation=45)
ax.set_xticklabels(model_labels, fontsize=8)
# Add value labels
"""
for px, v, e in zip(model_params, hit_rates, hit_rate_errors):
    ax.text(px, v + e + 0.15, f'{v:.1f}%', ha='center', va='bottom', fontsize=7)
"""
ax.text(-0.15, 1.15, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

### Panel C: Threshold sweep (speed-quality tradeoff)
ax = axes[1, 0]
thresh_data = threshold_df[threshold_df['threshold'] > 0].copy()

# Compute per-protein speedups and RMSD std from threshold_sweep_data
baseline_times_by_protein = {d['name']: d['inference_time'] for d in threshold_sweep_data if d['threshold'] == 0.0}
threshold_stats = {}
for thresh in thresh_data['threshold'].values:
    thresh_records = [d for d in threshold_sweep_data if d['threshold'] == thresh]
    speedups_per_protein = [baseline_times_by_protein[d['name']] / d['inference_time'] for d in thresh_records]
    rmsd_per_protein = [d['rmsd_vs_baseline'] for d in thresh_records if d['rmsd_vs_baseline'] is not None]
    threshold_stats[thresh] = {
        'speedup_std': np.std(speedups_per_protein),
        'rmsd_std': np.std(rmsd_per_protein) if rmsd_per_protein else 0
    }

thresh_data['speedup_std'] = thresh_data['threshold'].map(lambda t: threshold_stats[t]['speedup_std'])
thresh_data['rmsd_std'] = thresh_data['threshold'].map(lambda t: threshold_stats[t]['rmsd_std'])

# Create twin axis for RMSD
ax2 = ax.twinx()

# Plot speedup with shaded SD band
ax.fill_between(thresh_data['threshold'],
                thresh_data['speedup'] - thresh_data['speedup_std'],
                thresh_data['speedup'] + thresh_data['speedup_std'],
                color=COLORS['blue'], alpha=0.2, linewidth=0)
line1, = ax.plot(thresh_data['threshold'], thresh_data['speedup'], 'o-',
                 color=COLORS['blue'], linewidth=2, markersize=6, label='Speedup')
ax.set_xlabel('Cache Threshold (θ)')
ax.set_ylabel('Speedup (x)', color=COLORS['blue'])
ax.tick_params(axis='y', labelcolor=COLORS['blue'])

# Plot ΔRMSD with shaded SD band
ax2.fill_between(thresh_data['threshold'],
                 thresh_data['mean_rmsd_vs_baseline'] - thresh_data['rmsd_std'],
                 thresh_data['mean_rmsd_vs_baseline'] + thresh_data['rmsd_std'],
                 color=COLORS['rose'], alpha=0.2, linewidth=0)
line2, = ax2.plot(thresh_data['threshold'], thresh_data['mean_rmsd_vs_baseline'], 's-',
                  color=COLORS['rose'], linewidth=2, markersize=6, label='ΔRMSD')
ax2.set_ylabel('ΔRMSD (Å)', color=COLORS['rose'])
ax2.tick_params(axis='y', labelcolor=COLORS['rose'])

# Mark optimal threshold
opt_idx = thresh_data[thresh_data['threshold'] == 0.1].index[0]
ax.axvline(x=0.1, color='grey', linestyle='--', linewidth=1, alpha=0.7)
ax.annotate('θ = 0.1\n(optimal)', xy=(0.1, 9), xytext=(0.025, 9),
            fontsize=8, ha='left',
            arrowprops=dict(arrowstyle='->', color='grey', lw=0.8))

ax.set_xscale('log')
ax.set_xticks([0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
ax.set_xticklabels(['0.01', '0.05', '0.1', '0.2', '0.5', '1.0'])

# Legend
lines = [line1, line2]
labels = ['Speedup', 'ΔRMSD']
ax.legend(lines, labels, loc='lower right', fontsize=8, frameon=False)

ax.text(-0.15, 1.15, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
ax.spines['top'].set_visible(False)

### Panel D: Skip pattern heatmap
ax = axes[1, 1]

# Build skip pattern matrix (proteins x steps)
patterns = skip_data['patterns']
n_proteins = len(patterns)
n_steps = 500

# Sort proteins by length for better visualization
sorted_patterns = sorted(patterns, key=lambda p: p['num_residues'])
skip_matrix = np.array([p['skip_mask'] for p in sorted_patterns])

# Create heatmap - use imshow with custom colormap
cmap = mpl.colors.ListedColormap([COLORS['blue'], COLORS['sand']])
bounds = [-0.5, 0.5, 1.5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

im = ax.imshow(skip_matrix, aspect='auto', cmap=cmap, norm=norm,
               extent=[0, 500, 0, n_proteins], interpolation='nearest')

ax.set_xlabel('Diffusion step')
ax.set_ylabel('CATH domains (sorted by length)')
ax.tick_params(axis='y', left=False, labelleft=False)

# Add phase annotations
ax.axvline(x=10, color='white', linestyle='-', linewidth=1.5)
ax.axvline(x=480, color='white', linestyle='-', linewidth=1.5)

# Phase labels at top
ax.text(5, n_proteins + 12, 'Init.', fontsize=7, ha='center', va='bottom')
ax.text(245, n_proteins + 12, 'Sparse cruise (96% skip)', fontsize=7, ha='center', va='bottom')
ax.text(490, n_proteins + 12, 'Refine', fontsize=7, ha='center', va='bottom')

# Colorbar
cbar = plt.colorbar(im, ax=ax, ticks=[0, 1], shrink=0.6, aspect=15)
cbar.ax.set_yticklabels(['Compute', 'Skip'], fontsize=8, rotation=270, va='center')
cbar.ax.tick_params(axis='y', length=0)

ax.text(-0.15, 1.15, 'D', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

#plt.tight_layout()
plt.savefig('./Figure1_SF-T.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('./Figure1_SF-T.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved Figure1_SF-T.png and .pdf")
