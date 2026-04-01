import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import json
from scipy import stats
from matplotlib.patches import Patch

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

# Per-model colors — Paul Tol muted, matching Figure 1
MODEL_COLORS = {
    'simplefold_100M':  '#332288',  # indigo
    'simplefold_360M':  '#4477AA',  # blue
    'simplefold_700M':  '#44AA99',  # teal
    'simplefold_1.1B':  '#88CCEE',  # cyan
    'simplefold_1.6B':  '#DDCC77',  # sand
    'simplefold_3B':    '#CC6677',  # rose
}
MODEL_COLORS_LIST = list(MODEL_COLORS.values())
MODEL_LABELS = {
    'simplefold_100M': '100M', 'simplefold_360M': '360M',
    'simplefold_700M': '700M', 'simplefold_1.1B': '1.1B',
    'simplefold_1.6B': '1.6B', 'simplefold_3B': '3B',
}
MODELS = list(MODEL_COLORS.keys())

# Load dual sweep data
df = pd.read_csv('./data/dual_sweep_summary.csv')
a01 = df[(df['method'] == 'adaptive') & (df['condition'] == 0.1)]
a00 = df[(df['method'] == 'adaptive') & (df['condition'] == 0.0)]

# Also load SS content for panel B (100M only)
with open('./data/ss_content.json') as f:
    ss_data = json.load(f)
with open('./data/skip_patterns.json') as f:
    skip_data = json.load(f)

# Create figure: 1 row, 3 columns
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)

# =========================================================================
# Panel A: Cache hit rate vs sequence length — all 6 models
# =========================================================================
ax = axes[0]

for model in MODELS:
    mdf = a01[a01['model'] == model].dropna(subset=['cache_hit_rate', 'num_residues'])
    color = MODEL_COLORS[model]
    label = MODEL_LABELS[model]
    ax.scatter(mdf['num_residues'], mdf['cache_hit_rate'] * 100,
               c=color, alpha=0.35, s=15, edgecolors='none', label=label)

# Regression line using all models pooled
all_lengths = a01['num_residues'].values
all_hits = a01['cache_hit_rate'].values * 100
slope, intercept, r_all, p_all, _ = stats.linregress(all_lengths, all_hits)
x_line = np.array([all_lengths.min(), all_lengths.max()])
ax.plot(x_line, slope * x_line + intercept, color='black', linewidth=1.5,
        linestyle='--', zorder=10)

# Per-model r values as compact annotation
r_vals = []
for model in MODELS:
    mdf = a01[a01['model'] == model].dropna(subset=['cache_hit_rate', 'num_residues'])
    r, _ = stats.pearsonr(mdf['num_residues'], mdf['cache_hit_rate'])
    r_vals.append(r)
r_range = f'r = {min(r_vals):.2f}\u2013{max(r_vals):.2f}'
ax.text(0.03, 0.12, r_range, transform=ax.transAxes, fontsize=8.5,
        ha='left', va='bottom', fontweight='medium')

ax.set_xlabel('Sequence length (residues)')
ax.set_ylabel('Cache hit rate (%)')
ax.set_ylim(88, 98)
ax.set_xlim(40, 460)
ax.legend(fontsize=6.5, ncol=2, loc='lower right', framealpha=0.9,
          handletextpad=0.3, columnspacing=0.6)
ax.text(-0.15, 1.15, 'A', transform=ax.transAxes, fontsize=14,
        fontweight='bold', va='top')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# =========================================================================
# Panel B: Skip rate vs secondary structure (100M, from skip_patterns)
# =========================================================================
ax = axes[1]

patterns = skip_data['patterns']
helix_fracs, sheet_fracs, skip_rates_ss = [], [], []
for p in patterns:
    name = p['name']
    if name in ss_data:
        helix_fracs.append(ss_data[name]['helix_frac'] * 100)
        sheet_fracs.append(ss_data[name]['sheet_frac'] * 100)
        skip_rates_ss.append(np.mean(p['skip_mask']) * 100)

helix_fracs = np.array(helix_fracs)
sheet_fracs = np.array(sheet_fracs)
skip_rates_ss = np.array(skip_rates_ss)

sc = ax.scatter(helix_fracs, sheet_fracs, c=skip_rates_ss, cmap='viridis',
                alpha=0.7, s=30, edgecolors='white', linewidths=0.3,
                vmin=90, vmax=96)

cbar = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=20)
cbar.set_label('Skip rate (%)', fontsize=8)
cbar.ax.tick_params(labelsize=7)

r_helix, _ = stats.pearsonr(helix_fracs, skip_rates_ss)
r_sheet, _ = stats.pearsonr(sheet_fracs, skip_rates_ss)

ax.text(0.03, 0.23, f'r(\u03b1) = {r_helix:.2f}', transform=ax.transAxes,
        fontsize=8, ha='left', va='bottom')
ax.text(0.03, 0.12, f'r(\u03b2) = {r_sheet:.2f}', transform=ax.transAxes,
        fontsize=8, ha='left', va='bottom')

ax.set_xlabel('\u03b1-helix content (%)')
ax.set_ylabel('\u03b2-sheet content (%)')
ax.set_xlim(0, 70)
ax.set_ylim(0, 100)
ax.text(-0.15, 1.15, 'B', transform=ax.transAxes, fontsize=14,
        fontweight='bold', va='top')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# =========================================================================
# Panel C: TM-score vs GT — violin plot, all 6 models
# =========================================================================
ax = axes[2]

# Also get uniform 36-step data for comparison
u36 = df[(df['method'] == 'uniform') & (df['condition'] == 36.0)]

# Build violin data: 3 conditions per model
violin_data = []
violin_positions = []
violin_colors = []
group_width = 0.28  # spacing between violins within a model group

for i, model in enumerate(MODELS):
    base_vals = a00[a00['model'] == model]['tm_score_vs_gt'].dropna().values
    cache_vals = a01[a01['model'] == model]['tm_score_vs_gt'].dropna().values
    u36_vals = u36[u36['model'] == model]['tm_score_vs_gt'].dropna().values

    color = MODEL_COLORS[model]
    for j, (vals, alpha) in enumerate([
        (base_vals, 0.35),
        (cache_vals, 0.7),
        (u36_vals, 0.4),
    ]):
        pos = i + (j - 1) * group_width
        if len(vals) > 1:
            parts = ax.violinplot(vals, positions=[pos], showmeans=True,
                                  showextrema=False, widths=0.2)
            for pc in parts['bodies']:
                if j == 2:  # uniform — grey
                    pc.set_facecolor(COLORS['grey'])
                else:
                    pc.set_facecolor(color)
                pc.set_alpha(alpha)
                pc.set_edgecolor('none')
            parts['cmeans'].set_color('black')
            parts['cmeans'].set_linewidth(1.0)

# Manual legend
legend_elements = [
    Patch(facecolor=COLORS['blue'], alpha=0.35, edgecolor='none', label='Baseline (500 steps)'),
    Patch(facecolor=COLORS['blue'], alpha=0.7, edgecolor='none', label='SF-T (\u03c4 = 0.1)'),
    Patch(facecolor=COLORS['grey'], alpha=0.4, edgecolor='none', label='Uniform (36 steps)'),
]
ax.legend(handles=legend_elements, fontsize=6.5, loc='upper center',
          bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)

ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=8)
ax.set_xlabel('Model size')
ax.set_ylabel('TM-score vs. ground truth')
ax.set_ylim(0, 1.0)
ax.text(-0.15, 1.15, 'C', transform=ax.transAxes, fontsize=14,
        fontweight='bold', va='top')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# =========================================================================
# Save
# =========================================================================
plt.savefig('./Figure2_SF-T.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('./Figure2_SF-T.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved Figure2_SF-T.png and .pdf")
print(f"\nCache hit rate vs length: r = {min(r_vals):.3f}–{max(r_vals):.3f} across models")
print(f"SS correlations (100M): r(helix)={r_helix:.3f}, r(sheet)={r_sheet:.3f}")
