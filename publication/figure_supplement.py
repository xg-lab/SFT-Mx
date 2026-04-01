#!/usr/bin/env python3
"""
Supplementary Figures for SF-T (SimpleFold Turbo) Manuscript
Generates Fig S1-S4 as separate PNG/PDF files
Ordered by appearance in manuscript
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

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

# ============================================================================
# Helper functions
# ============================================================================

def compute_sequence_features(sequence: str) -> dict:
    """Compute sequence-based features for skip driver analysis."""
    if not sequence:
        return {}

    n = len(sequence)
    aa_counts = {}
    for aa in sequence:
        aa_counts[aa] = aa_counts.get(aa, 0) + 1

    # Hydrophobicity (Kyte-Doolittle)
    hydro_scale = {
        'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
        'G': -0.4, 'T': -0.7, 'W': -0.9, 'S': -0.8, 'Y': -1.3, 'P': -1.6,
        'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
    }
    hydro = np.mean([hydro_scale.get(aa, 0) for aa in sequence])

    # Disorder propensity
    disorder_prone = set('PSEKQRGD')
    disorder_score = sum(1 for aa in sequence if aa in disorder_prone) / n

    # Sequence entropy
    probs = np.array(list(aa_counts.values())) / n
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    return {
        'length': n,
        'hydrophobicity': hydro,
        'disorder_score': disorder_score,
        'sequence_entropy': entropy
    }

def compute_skip_metrics(skip_mask: list) -> dict:
    """Extract metrics from skip pattern."""
    mask = np.array(skip_mask)
    skip_rate = np.mean(mask)

    # Transitions
    transitions = np.abs(np.diff(mask))
    n_transitions = np.sum(transitions)

    # Mean compute interval
    compute_steps = np.where(mask == 0)[0]
    if len(compute_steps) > 1:
        mean_compute_interval = np.mean(np.diff(compute_steps))
    else:
        mean_compute_interval = 0

    return {
        'skip_rate': skip_rate,
        'n_transitions': n_transitions,
        'mean_compute_interval': mean_compute_interval
    }

# ============================================================================
# Load data
# ============================================================================

print("Loading data...")

with open('./data/skip_patterns.json') as f:
    skip_data = json.load(f)

with open('./data/ss_content.json') as f:
    ss_data = json.load(f)

gt_df = pd.read_csv('./data/gt_comparison.csv')

with open('./data/crystallization_results.json') as f:
    crystallization_data = json.load(f)

patterns = skip_data['patterns']

# ============================================================================
# Figure S1: Clustering Analysis (first reference in manuscript)
# K-means clustering with k=3 for improved interpretability
# ============================================================================

print("\nGenerating Fig S1: Clustering Analysis...")

# Get skip masks
masks = np.array([p['skip_mask'] for p in patterns])

# Perform k-means clustering with k=3
N_CLUSTERS = 3
print(f"  Running k-means clustering (k={N_CLUSTERS})...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels = kmeans.fit_predict(masks)

# Cluster colors - distinct, colorblind-safe
CLUSTER_COLORS = [COLORS['teal'], COLORS['wine'], COLORS['blue']]
CLUSTER_NAMES = ['Cluster 1', 'Cluster 2', 'Cluster 3']

# SS data is keyed directly by protein name
ss_by_protein = ss_data

fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)

# A: t-SNE of skip patterns
ax = axes[0, 0]
print("  Computing t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
coords = tsne.fit_transform(masks)

for cluster_id in range(N_CLUSTERS):
    mask = labels == cluster_id
    ax.scatter(coords[mask, 0], coords[mask, 1], c=CLUSTER_COLORS[cluster_id],
               label=CLUSTER_NAMES[cluster_id], alpha=0.6, s=30, edgecolors='white', linewidths=0.3)

ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE of skip patterns (k-means, k=3)', fontsize=10)
ax.legend(loc='lower right', fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.08, 'A', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# B: SS composition by cluster
ax = axes[0, 1]
x = np.arange(N_CLUSTERS)
width = 0.25

# Compute SS stats for each cluster
cluster_ss_stats = []
for cluster_id in range(N_CLUSTERS):
    cluster_proteins = [patterns[i]['name'] for i in range(len(patterns)) if labels[i] == cluster_id]
    helix_vals = []
    sheet_vals = []
    coil_vals = []
    for pname in cluster_proteins:
        if pname in ss_by_protein:
            p = ss_by_protein[pname]
            helix_vals.append(p.get('helix_frac', 0))
            sheet_vals.append(p.get('sheet_frac', 0))
            coil_vals.append(p.get('coil_frac', 0))
    cluster_ss_stats.append({
        'n': len(cluster_proteins),
        'helix': np.mean(helix_vals) if helix_vals else 0,
        'helix_std': np.std(helix_vals) if helix_vals else 0,
        'sheet': np.mean(sheet_vals) if sheet_vals else 0,
        'sheet_std': np.std(sheet_vals) if sheet_vals else 0,
        'coil': np.mean(coil_vals) if coil_vals else 0,
        'coil_std': np.std(coil_vals) if coil_vals else 0
    })

helix_vals = [s['helix'] for s in cluster_ss_stats]
helix_std = [s['helix_std'] for s in cluster_ss_stats]
sheet_vals = [s['sheet'] for s in cluster_ss_stats]
sheet_std = [s['sheet_std'] for s in cluster_ss_stats]
coil_vals = [s['coil'] for s in cluster_ss_stats]
coil_std = [s['coil_std'] for s in cluster_ss_stats]

ax.bar(x - width, helix_vals, width, yerr=helix_std, label='Helix', color=COLORS['rose'], alpha=0.8,
       error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1, 'elinewidth': 1})
ax.bar(x, sheet_vals, width, yerr=sheet_std, label='Sheet', color=COLORS['blue'], alpha=0.8,
       error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1, 'elinewidth': 1})
ax.bar(x + width, coil_vals, width, yerr=coil_std, label='Coil', color=COLORS['green'], alpha=0.8,
       error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1, 'elinewidth': 1})

ax.set_xlabel('Cluster')
ax.set_ylabel('SS fraction')
ax.set_title('Secondary structure by cluster', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels([f"C{i+1}\n(n={cluster_ss_stats[i]['n']})" for i in range(N_CLUSTERS)])
ax.legend(loc='upper left', fontsize=8)
ax.set_ylim(0, 0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.08, 'B', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# C: Mean skip pattern by cluster
ax = axes[1, 0]
steps = np.arange(500)

for cluster_id in range(N_CLUSTERS):
    cluster_mask = labels == cluster_id
    mean_pattern = masks[cluster_mask].mean(axis=0)
    ax.plot(steps, mean_pattern * 100, label=CLUSTER_NAMES[cluster_id],
            color=CLUSTER_COLORS[cluster_id], alpha=0.8, linewidth=1.5)

ax.axvline(x=10, linestyle='--', alpha=0.5, linewidth=1)
ax.text(15, 10, 'Warmup\nend', fontsize=7)

ax.set_xlabel('Diffusion step')
ax.set_ylabel('Skip probability (%)')
ax.set_title('Mean skip pattern by cluster', fontsize=10)
ax.legend(loc='lower center', fontsize=8)
ax.set_ylim(-5, 105)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.08, 'C', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# D: Skip rate comparison (violin plots)
ax = axes[1, 1]

# Get skip rates by cluster
cluster_rates = []
for cluster_id in range(N_CLUSTERS):
    rates = [np.mean(patterns[i]['skip_mask']) * 100 for i in range(len(patterns)) if labels[i] == cluster_id]
    cluster_rates.append(rates)

parts = ax.violinplot(cluster_rates, positions=range(1, N_CLUSTERS+1), showmeans=True, showextrema=False, widths=0.7)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(CLUSTER_COLORS[i])
    pc.set_alpha(0.4)
parts['cmeans'].set_color('black')
parts['cmeans'].set_linewidth(1.5)

# Add jittered points
np.random.seed(42)
for i, rates in enumerate(cluster_rates):
    ax.scatter(i+1 + np.random.normal(0, 0.08, len(rates)), rates,
               c=CLUSTER_COLORS[i], alpha=0.3, s=10, edgecolors='none')

ax.set_xticks(range(1, N_CLUSTERS+1))
ax.set_xticklabels([CLUSTER_NAMES[i] for i in range(N_CLUSTERS)])
ax.set_ylabel('Skip rate (%)')
ax.set_title('Skip rate by cluster', fontsize=10)
ax.set_ylim(88, 99)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.08, 'D', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

plt.savefig('./FigS1_clustering.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('./FigS1_clustering.pdf', bbox_inches='tight', facecolor='white')
print("  Saved FigS1_clustering.png/pdf")
plt.close()

# ============================================================================
# Figure S2: Skip Pattern Drivers Analysis (2x2: length, disorder, hydro, entropy vs skip rate)
# ============================================================================

print("Generating Fig S2: Skip Pattern Drivers...")

# Compute features
results = []
for p in patterns:
    seq_features = compute_sequence_features(p.get('sequence', ''))
    skip_metrics = compute_skip_metrics(p['skip_mask'])
    results.append({**seq_features, **skip_metrics})

fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)

sizes = [r['length'] for r in results]
skip_rates = [r['skip_rate'] * 100 for r in results]
disorder = [r['disorder_score'] for r in results]
hydro = [r['hydrophobicity'] for r in results]
entropy = [r['sequence_entropy'] for r in results]

# A: Length vs Skip Rate
ax = axes[0, 0]
ax.scatter(sizes, skip_rates, alpha=0.5, s=25, c=COLORS['blue'], edgecolors='white', linewidths=0.3)
r, p = stats.pearsonr(sizes, skip_rates)
ax.set_xlabel('Protein length (residues)')
ax.set_ylabel('Skip rate (%)')
ax.set_title(f'Length vs skip rate (r = {r:.2f})', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.08, 'A', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# B: Disorder vs Skip Rate
ax = axes[0, 1]
ax.scatter(disorder, skip_rates, alpha=0.5, s=25, c=COLORS['teal'], edgecolors='white', linewidths=0.3)
r, p = stats.pearsonr(disorder, skip_rates)
ax.set_xlabel('Disorder propensity')
ax.set_ylabel('Skip rate (%)')
ax.set_title(f'Disorder vs skip rate (r = {r:.2f})', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.08, 'B', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# C: Hydrophobicity vs Skip Rate
ax = axes[1, 0]
ax.scatter(hydro, skip_rates, alpha=0.5, s=25, c=COLORS['rose'], edgecolors='white', linewidths=0.3)
r, p = stats.pearsonr(hydro, skip_rates)
ax.set_xlabel('Mean hydrophobicity')
ax.set_ylabel('Skip rate (%)')
ax.set_title(f'Hydrophobicity vs skip rate (r = {r:.2f})', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.08, 'C', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# D: Entropy vs Skip Rate
ax = axes[1, 1]
ax.scatter(entropy, skip_rates, alpha=0.5, s=25, c=COLORS['purple'], edgecolors='white', linewidths=0.3)
r, p = stats.pearsonr(entropy, skip_rates)
ax.set_xlabel('Sequence entropy')
ax.set_ylabel('Skip rate (%)')
ax.set_title(f'Entropy vs skip rate (r = {r:.2f})', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.08, 'D', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

plt.savefig('./FigS2_skip_drivers.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('./FigS2_skip_drivers.pdf', bbox_inches='tight', facecolor='white')
print("  Saved FigS2_skip_drivers.png/pdf")
plt.close()

# ============================================================================
# Figure S3: Full Quality Comparison (1x3 horizontal layout)
# ============================================================================

print("Generating Fig S3: Full Quality Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)

# A: TM-score
ax = axes[0]
ax.scatter(gt_df['tm_baseline'], gt_df['tm_teacache'], c=COLORS['blue'], alpha=0.5, s=20,
           edgecolors='white', linewidths=0.2)
ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1, alpha=0.7)
r, _ = stats.pearsonr(gt_df['tm_baseline'], gt_df['tm_teacache'])
ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=9, ha='left', va='top')
ax.text(0.05, 0.87, f'Δ = {gt_df["tm_diff"].mean():.4f}', transform=ax.transAxes, fontsize=8,
        ha='left', va='top')
ax.set_xlabel('Baseline TM-score')
ax.set_ylabel('SF-T TM-score')
ax.set_title('TM-score preservation', fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.08, 'A', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# B: RMSD
ax = axes[1]
ax.scatter(gt_df['rmsd_baseline'], gt_df['rmsd_teacache'], c=COLORS['teal'], alpha=0.5, s=20,
           edgecolors='white', linewidths=0.2)
max_rmsd = max(gt_df['rmsd_baseline'].max(), gt_df['rmsd_teacache'].max())
ax.plot([0, max_rmsd], [0, max_rmsd], linestyle='--', linewidth=1, alpha=0.7)
r, _ = stats.pearsonr(gt_df['rmsd_baseline'], gt_df['rmsd_teacache'])
ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=9, ha='left', va='top')
ax.text(0.05, 0.87, f'Δ = {gt_df["rmsd_diff"].mean():.2f} Å', transform=ax.transAxes, fontsize=8,
        ha='left', va='top')
ax.set_xlabel('Baseline RMSD (Å)')
ax.set_ylabel('SF-T RMSD (Å)')
ax.set_title('RMSD preservation', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.08, 'B', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# C: lDDT
ax = axes[2]
ax.scatter(gt_df['lddt_baseline'], gt_df['lddt_teacache'], c=COLORS['rose'], alpha=0.5, s=20,
           edgecolors='white', linewidths=0.2)
ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1, alpha=0.7)
r, _ = stats.pearsonr(gt_df['lddt_baseline'], gt_df['lddt_teacache'])
ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=9, ha='left', va='top')
ax.text(0.05, 0.87, f'Δ = {gt_df["lddt_diff"].mean():.4f}', transform=ax.transAxes, fontsize=8,
        ha='left', va='top')
ax.set_xlabel('Baseline lDDT')
ax.set_ylabel('SF-T lDDT')
ax.set_title('lDDT preservation', fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.08, 'C', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

plt.savefig('./FigS3_quality.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('./FigS3_quality.pdf', bbox_inches='tight', facecolor='white')
print("  Saved FigS3_quality.png/pdf")
plt.close()

# ============================================================================
# Figure S4: Atom Settling Time Analysis (2 panels, no text)
# ============================================================================

print("Generating Fig S4: Atom Settling Time...")

# Extract real settling data from crystallization results
backbone_settling = [p['statistics']['backbone']['mean_settling_pct'] for p in crystallization_data]
sidechain_settling = [p['statistics']['sidechain']['mean_settling_pct'] for p in crystallization_data]
terminal_settling = [p['statistics']['terminal']['mean_settling_pct'] for p in crystallization_data]

# Aggregate by_depth data across all proteins
depth_data = {d: [] for d in range(7)}
for p in crystallization_data:
    for depth_str, stats in p['statistics']['by_depth'].items():
        depth = int(depth_str)
        if depth < 7:
            depth_data[depth].append(stats['mean_settling_pct'])

bond_distances = list(range(7))
settling_by_bond = [np.mean(depth_data[d]) for d in bond_distances]
settling_std = [np.std(depth_data[d]) for d in bond_distances]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)

# A: Box plot by atom category
ax = axes[0]
bp = ax.boxplot([backbone_settling, sidechain_settling, terminal_settling],
                tick_labels=['Backbone\n(N,CA,C,O)', 'Sidechain\n(CB,CG,CD...)', 'Terminal\n(NH2,OXT...)'],
                patch_artist=True, widths=0.6)
colors_box = [COLORS['blue'], COLORS['teal'], COLORS['rose']]
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(1.5)

# Add mean annotations
means = [np.mean(backbone_settling), np.mean(sidechain_settling), np.mean(terminal_settling)]
for i, m in enumerate(means):
    ax.text(i+1, ax.get_ylim()[0]-1.5, f'μ={m:.1f}%', ha='center', va='bottom', fontsize=8,
            color='black')

ax.set_ylabel('Settling step (% of diffusion)')
ax.set_title('Atom settling time by category', fontsize=10)
ax.set_ylim(60, 100)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.08, 'A', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# B: Settling time vs bond distance from backbone
ax = axes[1]
ax.errorbar(bond_distances, settling_by_bond, yerr=settling_std,
            fmt='o', color=COLORS['blue'], markersize=8, capsize=4, elinewidth=1.5)
ax.axhline(y=np.mean(settling_by_bond), linestyle='--', alpha=0.5)

ax.set_xlabel('Bond distance from backbone')
ax.set_ylabel('Settling step (% of diffusion)')
ax.set_title('Settling time vs bond distance', fontsize=10)
ax.set_ylim(70, 95)
ax.set_xticks(bond_distances)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.08, 'B', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

plt.savefig('./FigS4_atom_settling.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('./FigS4_atom_settling.pdf', bbox_inches='tight', facecolor='white')
print("  Saved FigS4_atom_settling.png/pdf")
plt.close()

print("\n✓ All supplementary figures generated!")
print("\nFigure order (by manuscript appearance):")
print("  S1: Clustering analysis")
print("  S2: Skip pattern drivers")
print("  S3: Quality comparison")
print("  S4: Atom settling time")
