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

# Model colors - gradient from blue to teal
MODEL_COLORS = ['#332288', '#4477AA', '#44AA99', '#88CCEE', '#DDCC77', '#CC6677']

# Load data
benchmark_df = pd.read_csv('./data/cath_benchmark_full.csv')
threshold_df = pd.read_csv('./data/threshold_summary.csv')

with open('./data/skip_patterns.json') as f:
    skip_data = json.load(f)

with open('./data/threshold_sweep.json') as f:
    threshold_sweep_data = json.load(f)

# Model configuration
models = ['simplefold_100M', 'simplefold_360M', 'simplefold_700M', 'simplefold_1.1B', 'simplefold_1.6B', 'simplefold_3B']
model_labels = ['100M', '360M', '700M', '1.1B', '1.6B', '3B']

# Compute per-protein speedups and hit rates for each model
model_speedups = {}
model_hit_rates = {}

for model in models:
    baseline = benchmark_df[(benchmark_df['model'] == model) & (benchmark_df['teacache_threshold'] == 0.0)]
    cached = benchmark_df[(benchmark_df['model'] == model) & (benchmark_df['teacache_threshold'] == 0.1)]

    # Merge on protein name to compute per-protein speedup
    merged = baseline[['name', 'time_s']].merge(
        cached[['name', 'time_s', 'cache_hit']],
        on='name',
        suffixes=('_baseline', '_cached')
    )

    merged['speedup'] = merged['time_s_baseline'] / merged['time_s_cached']
    merged['hit_rate'] = merged['cache_hit'] * 100

    model_speedups[model] = merged['speedup'].values
    model_hit_rates[model] = merged['hit_rate'].values

# Create figure with 4 subpanels
fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
fig.subplots_adjust(hspace=0.35, wspace=0.35)

### Panel A: Inference time comparison (baseline vs SF-T) - grouped bars
ax = axes[0, 0]

# Load dual sweep data for timing
dual_df = pd.read_csv('./data/dual_sweep_summary.csv')

bar_width = 0.35
positions = np.arange(len(models))

baseline_means = []
baseline_sds = []
cached_means = []
cached_sds = []
speedup_labels = []

for m in models:
    base_t = dual_df[(dual_df['model']==m) & (dual_df['method']=='adaptive') & (dual_df['condition']==0.0)]['inference_time'].dropna()
    cache_t = dual_df[(dual_df['model']==m) & (dual_df['method']=='adaptive') & (dual_df['condition']==0.1)]['inference_time'].dropna()
    baseline_means.append(base_t.mean())
    baseline_sds.append(base_t.std())
    cached_means.append(cache_t.mean())
    cached_sds.append(cache_t.std())
    speedup_labels.append(f'{base_t.mean()/cache_t.mean():.0f}\u00d7')

bars_base = ax.bar(positions - bar_width/2, baseline_means, bar_width,
                   yerr=baseline_sds, capsize=3, color=COLORS['grey'],
                   edgecolor='white', linewidth=0.5, label='SimpleFold (500 steps)',
                   error_kw={'linewidth': 0.8})
bars_cache = ax.bar(positions + bar_width/2, cached_means, bar_width,
                    yerr=cached_sds, capsize=3, color=COLORS['blue'],
                    edgecolor='white', linewidth=0.5, label='SF-T (\u03c4 = 0.1)',
                    error_kw={'linewidth': 0.8})

# Add speedup annotations above each pair
for i, (pos, label) in enumerate(zip(positions, speedup_labels)):
    y_top = baseline_means[i] + baseline_sds[i] + 2
    ax.text(pos, y_top, label, ha='center', va='bottom', fontsize=7.5, fontweight='bold',
            color=COLORS['blue'])

ax.set_xticks(positions)
ax.set_xticklabels(model_labels, fontsize=8)
ax.set_xlabel('Model size')
ax.set_ylabel('Inference time (s)')
ax.legend(fontsize=7, framealpha=0.9, loc='upper left')
ax.text(-0.15, 1.15, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

### Panel B: Cache hit rate distribution for all 6 models (violin + jitter)
ax = axes[0, 1]

hit_rate_data = [model_hit_rates[m] for m in models]

# Violin plot
parts = ax.violinplot(hit_rate_data, positions=positions, showmeans=True, showextrema=False, widths=0.7)
for i, body in enumerate(parts['bodies']):
    body.set_facecolor(MODEL_COLORS[i])
    body.set_alpha(0.4)
parts['cmeans'].set_color('black')
parts['cmeans'].set_linewidth(1.5)

# Add jittered points
np.random.seed(42)
for i, (pos, data) in enumerate(zip(positions, hit_rate_data)):
    jitter = np.random.normal(0, 0.08, len(data))
    ax.scatter(pos + jitter, data, c=MODEL_COLORS[i], alpha=0.4, s=12, edgecolors='none')

ax.set_xticks(positions)
ax.set_xticklabels(model_labels, fontsize=8)
ax.set_xlabel('Model Size')
ax.set_ylabel('Cache hit rate (%)')
ax.set_ylim(88, 98)
ax.text(-0.15, 1.15, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

### Panel C: Threshold sweep (speed-quality tradeoff) - 100M, from dual sweep data
ax = axes[1, 0]

# Compute per-threshold stats from dual sweep (100M, adaptive, τ > 0)
ds_100m = dual_df[(dual_df['model']=='simplefold_100M') & (dual_df['method']=='adaptive')]
baseline_times = ds_100m[ds_100m['condition']==0.0].set_index('name')['inference_time']

thresholds = sorted([t for t in ds_100m['condition'].unique() if t > 0])
c_speedup_mean, c_speedup_sd = [], []
c_rmsd_mean, c_rmsd_sd = [], []

for t in thresholds:
    tdf = ds_100m[ds_100m['condition']==t]
    # Per-protein speedup
    merged_t = tdf.set_index('name')[['inference_time']].join(baseline_times, rsuffix='_base')
    merged_t['speedup'] = merged_t['inference_time_base'] / merged_t['inference_time']
    c_speedup_mean.append(merged_t['speedup'].mean())
    c_speedup_sd.append(merged_t['speedup'].std())
    # ΔRMSD vs baseline
    rmsd_vals = tdf['rmsd_vs_baseline'].dropna()
    c_rmsd_mean.append(rmsd_vals.mean() if len(rmsd_vals) > 0 else 0)
    c_rmsd_sd.append(rmsd_vals.std() if len(rmsd_vals) > 0 else 0)

thresholds = np.array(thresholds)
c_speedup_mean = np.array(c_speedup_mean)
c_speedup_sd = np.array(c_speedup_sd)
c_rmsd_mean = np.array(c_rmsd_mean)
c_rmsd_sd = np.array(c_rmsd_sd)

# Create twin axis for RMSD
ax2 = ax.twinx()

# Plot speedup with error bars
line1 = ax.errorbar(thresholds, c_speedup_mean, yerr=c_speedup_sd,
                    fmt='o-', color=COLORS['blue'], linewidth=2, markersize=6,
                    capsize=3, capthick=1, elinewidth=0.8, label='Speedup')

ax.set_xlabel('Cache threshold (\u03c4)')
ax.set_ylabel('Speedup (\u00d7)', color=COLORS['blue'])
ax.tick_params(axis='y', labelcolor=COLORS['blue'])

# Plot ΔRMSD with error bars
line2 = ax2.errorbar(thresholds, c_rmsd_mean, yerr=c_rmsd_sd,
                     fmt='s-', color=COLORS['rose'], linewidth=2, markersize=6,
                     capsize=3, capthick=1, elinewidth=0.8, label='\u0394RMSD')
ax2.set_ylabel('\u0394RMSD (\u00c5)', color=COLORS['rose'])
ax2.tick_params(axis='y', labelcolor=COLORS['rose'])

# Mark optimal threshold
ax.axvline(x=0.1, color='grey', linestyle='--', linewidth=1, alpha=0.7)
ax.annotate('\u03c4 = 0.1\n(default)', xy=(0.1, c_speedup_mean[thresholds==0.1][0]),
            xytext=(0.025, c_speedup_mean[thresholds==0.1][0]),
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

### Panel D: Skip pattern heatmap (100M only)
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

plt.savefig('./Figure1_alt_SF-T.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('./Figure1_alt_SF-T.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved Figure1_alt_SF-T.png and .pdf")
