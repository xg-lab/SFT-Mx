#!/usr/bin/env python3
"""
Skip Pattern Clustering Analysis

Clusters proteins by their TeaCache skip patterns, then analyzes
whether clusters correlate with secondary structure composition.

Usage:
    python skip_pattern_clustering.py --patterns teacache_patterns_300/skip_patterns.json
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def predict_ss_composition(sequence: str) -> Dict[str, float]:
    """
    Predict SS composition from sequence using amino acid propensities.
    Returns fraction of helix, sheet, coil.
    """
    # Chou-Fasman propensities
    helix_formers = set('AELM')
    sheet_formers = set('VIY')

    if not sequence:
        return {'helix': 0.33, 'sheet': 0.33, 'coil': 0.34}

    h_count = sum(1 for aa in sequence if aa in helix_formers)
    e_count = sum(1 for aa in sequence if aa in sheet_formers)

    total = len(sequence)
    return {
        'helix': h_count / total,
        'sheet': e_count / total,
        'coil': (total - h_count - e_count) / total
    }


def compute_skip_features(skip_mask: List[int], num_steps: int = 500) -> Dict[str, float]:
    """
    Extract features from skip pattern for clustering.
    """
    mask = np.array(skip_mask)

    # Phase-based features
    phases = [
        (0, 50, 'early'),
        (50, 150, 'mid_early'),
        (150, 300, 'mid'),
        (300, 450, 'mid_late'),
        (450, 500, 'late')
    ]

    features = {}
    for start, end, name in phases:
        features[f'skip_{name}'] = np.mean(mask[start:end])

    # Transition features (when does skipping behavior change?)
    # Find first skip after warmup
    warmup_end = 11
    post_warmup = mask[warmup_end:]

    # Count transitions (0->1 or 1->0)
    transitions = np.sum(np.abs(np.diff(mask)))
    features['n_transitions'] = transitions

    # Run length statistics
    runs = []
    current_run = 1
    for i in range(1, len(mask)):
        if mask[i] == mask[i-1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)

    features['mean_run_length'] = np.mean(runs)
    features['max_run_length'] = np.max(runs)

    # Overall skip rate
    features['overall_skip_rate'] = np.mean(mask)

    return features


def cluster_by_pattern(patterns: List[Dict], n_clusters: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster proteins by their full skip pattern using hierarchical clustering.

    Returns:
        labels: Cluster assignment for each protein
        linkage_matrix: For dendrogram plotting
    """
    # Use full skip masks as feature vectors
    masks = np.array([p['skip_mask'] for p in patterns])

    # Compute pairwise distances (Hamming distance makes sense for binary vectors)
    distances = pdist(masks, metric='hamming')

    # Hierarchical clustering
    Z = linkage(distances, method='ward')

    # Cut tree to get clusters
    labels = fcluster(Z, n_clusters, criterion='maxclust')

    return labels, Z, masks


def analyze_clusters(patterns: List[Dict], labels: np.ndarray, output_dir: Path):
    """
    Analyze SS composition and skip characteristics of each cluster.
    """
    n_clusters = len(set(labels))

    # Group proteins by cluster
    clusters = {i: [] for i in range(1, n_clusters + 1)}
    for i, (p, label) in enumerate(zip(patterns, labels)):
        clusters[label].append(p)

    # Analyze each cluster
    cluster_stats = {}

    print(f"\n{'='*70}")
    print(f"CLUSTER ANALYSIS ({n_clusters} clusters)")
    print(f"{'='*70}\n")

    for cluster_id in sorted(clusters.keys()):
        prots = clusters[cluster_id]

        # SS composition
        ss_comps = [predict_ss_composition(p.get('sequence', '')) for p in prots]
        mean_helix = np.mean([s['helix'] for s in ss_comps])
        mean_sheet = np.mean([s['sheet'] for s in ss_comps])
        mean_coil = np.mean([s['coil'] for s in ss_comps])

        # Size statistics
        sizes = [p['num_residues'] for p in prots]
        mean_size = np.mean(sizes)

        # Skip pattern statistics
        skip_rates = [p['hit_rate'] for p in prots]
        mean_skip = np.mean(skip_rates)

        # Phase-specific skip rates
        masks = np.array([p['skip_mask'] for p in prots])
        early_skip = np.mean(masks[:, :50])
        mid_skip = np.mean(masks[:, 150:350])
        late_skip = np.mean(masks[:, 450:])

        cluster_stats[cluster_id] = {
            'n_proteins': len(prots),
            'mean_helix': mean_helix,
            'mean_sheet': mean_sheet,
            'mean_coil': mean_coil,
            'mean_size': mean_size,
            'mean_skip_rate': mean_skip,
            'early_skip': early_skip,
            'mid_skip': mid_skip,
            'late_skip': late_skip,
            'proteins': [p['name'] for p in prots]
        }

        print(f"CLUSTER {cluster_id}: {len(prots)} proteins")
        print(f"  SS Composition: {mean_helix*100:.1f}% helix, {mean_sheet*100:.1f}% sheet, {mean_coil*100:.1f}% coil")
        print(f"  Mean size: {mean_size:.0f} residues")
        print(f"  Skip rate: {mean_skip*100:.1f}%")
        print(f"  Phase skips: early={early_skip:.2f}, mid={mid_skip:.2f}, late={late_skip:.2f}")
        print()

    return cluster_stats


def create_visualizations(patterns: List[Dict], labels: np.ndarray, Z: np.ndarray,
                          masks: np.ndarray, cluster_stats: Dict, output_dir: Path):
    """
    Create visualization of clustering results.
    """
    n_clusters = len(set(labels))

    fig = plt.figure(figsize=(16, 14))

    # 1. Dendrogram (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90,
               leaf_font_size=8, ax=ax1, color_threshold=0.7*max(Z[:,2]))
    ax1.set_title('Hierarchical Clustering of Skip Patterns')
    ax1.set_xlabel('Protein (truncated)')
    ax1.set_ylabel('Distance')

    # 2. t-SNE visualization colored by cluster (top right)
    ax2 = fig.add_subplot(2, 2, 2)

    # Reduce dimensionality for visualization
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(masks)

    # Color by cluster
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for cluster_id in range(1, n_clusters + 1):
        mask = labels == cluster_id
        ax2.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[cluster_id-1]], label=f'Cluster {cluster_id}',
                   alpha=0.6, s=40)

    ax2.set_title('t-SNE of Skip Patterns (colored by cluster)')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend(loc='best', fontsize=8)

    # 3. Cluster SS composition comparison (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)

    x = np.arange(n_clusters)
    width = 0.25

    helix_vals = [cluster_stats[i+1]['mean_helix'] for i in range(n_clusters)]
    sheet_vals = [cluster_stats[i+1]['mean_sheet'] for i in range(n_clusters)]
    coil_vals = [cluster_stats[i+1]['mean_coil'] for i in range(n_clusters)]

    ax3.bar(x - width, helix_vals, width, label='Helix', color='red', alpha=0.7)
    ax3.bar(x, sheet_vals, width, label='Sheet', color='blue', alpha=0.7)
    ax3.bar(x + width, coil_vals, width, label='Coil', color='green', alpha=0.7)

    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('SS Fraction')
    ax3.set_title('Secondary Structure Composition by Cluster')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'C{i+1}\n(n={cluster_stats[i+1]["n_proteins"]})' for i in range(n_clusters)])
    ax3.legend()
    ax3.set_ylim(0, 1)

    # 4. Mean skip pattern per cluster (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)

    steps = np.arange(masks.shape[1])
    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = labels == cluster_id
        mean_pattern = masks[cluster_mask].mean(axis=0)
        ax4.plot(steps, mean_pattern, label=f'Cluster {cluster_id}', alpha=0.7)

    ax4.set_xlabel('Diffusion Step')
    ax4.set_ylabel('Skip Probability')
    ax4.set_title('Mean Skip Pattern by Cluster')
    ax4.legend(loc='lower right', fontsize=8)
    ax4.set_ylim(-0.05, 1.05)
    ax4.axvline(x=11, color='gray', linestyle='--', alpha=0.5, label='Warmup end')

    plt.tight_layout()
    plt.savefig(output_dir / 'skip_clustering.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'skip_clustering.png'}")

    # Additional figure: Heatmap of patterns sorted by cluster
    fig2, ax = plt.subplots(figsize=(14, 10))

    # Sort by cluster
    sorted_idx = np.argsort(labels)
    sorted_masks = masks[sorted_idx]
    sorted_labels = labels[sorted_idx]

    im = ax.imshow(sorted_masks, aspect='auto', cmap='RdYlGn',
                   interpolation='nearest', vmin=0, vmax=1)

    # Add cluster boundaries
    cluster_boundaries = []
    current_label = sorted_labels[0]
    for i, label in enumerate(sorted_labels):
        if label != current_label:
            cluster_boundaries.append(i)
            current_label = label

    for boundary in cluster_boundaries:
        ax.axhline(y=boundary - 0.5, color='black', linewidth=2)

    ax.set_xlabel('Diffusion Step')
    ax.set_ylabel('Protein (sorted by cluster)')
    ax.set_title('Skip Patterns Sorted by Cluster\n(Green=Skip, Red=Compute)')

    # Add cluster labels on y-axis
    cluster_centers = []
    start = 0
    for i, boundary in enumerate(cluster_boundaries + [len(sorted_labels)]):
        center = (start + boundary) / 2
        cluster_centers.append((center, i + 1))
        start = boundary

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Skip (1) vs Compute (0)')

    plt.tight_layout()
    plt.savefig(output_dir / 'skip_clustering_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'skip_clustering_heatmap.png'}")

    return coords


def find_optimal_clusters(masks: np.ndarray, max_k: int = 10) -> int:
    """
    Use silhouette score to find optimal number of clusters.
    """
    from sklearn.metrics import silhouette_score

    distances = pdist(masks, metric='hamming')
    Z = linkage(distances, method='ward')

    scores = []
    for k in range(2, max_k + 1):
        labels = fcluster(Z, k, criterion='maxclust')
        score = silhouette_score(masks, labels, metric='hamming')
        scores.append((k, score))
        print(f"  k={k}: silhouette={score:.3f}")

    best_k = max(scores, key=lambda x: x[1])[0]
    return best_k


def main():
    parser = argparse.ArgumentParser(description='Cluster proteins by skip pattern')
    parser.add_argument('--patterns', type=str, default='teacache_patterns_300/skip_patterns.json',
                       help='Path to skip patterns JSON')
    parser.add_argument('--output-dir', type=str, default='teacache_patterns_300',
                       help='Output directory')
    parser.add_argument('--n-clusters', type=int, default=None,
                       help='Number of clusters (auto-detect if not specified)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading skip patterns...")
    with open(args.patterns) as f:
        data = json.load(f)

    patterns = data['patterns']
    print(f"Loaded {len(patterns)} proteins")

    # Get masks for clustering
    masks = np.array([p['skip_mask'] for p in patterns])

    # Find optimal number of clusters if not specified
    if args.n_clusters is None:
        print("\nFinding optimal number of clusters...")
        n_clusters = find_optimal_clusters(masks, max_k=10)
        print(f"\nOptimal k={n_clusters} based on silhouette score")
    else:
        n_clusters = args.n_clusters

    # Cluster
    print(f"\nClustering into {n_clusters} clusters...")
    labels, Z, masks = cluster_by_pattern(patterns, n_clusters)

    # Analyze clusters
    cluster_stats = analyze_clusters(patterns, labels, output_dir)

    # Create visualizations
    coords = create_visualizations(patterns, labels, Z, masks, cluster_stats, output_dir)

    # Add SS predictions to patterns for correlation analysis
    for p in patterns:
        ss = predict_ss_composition(p.get('sequence', ''))
        p['helix_frac'] = ss['helix']
        p['sheet_frac'] = ss['sheet']
        p['coil_frac'] = ss['coil']

    # Compute correlation between SS composition and cluster assignment
    print("\n" + "="*70)
    print("SS COMPOSITION VS CLUSTERING ANALYSIS")
    print("="*70)

    # ANOVA-like analysis: do clusters differ in SS composition?
    helix_by_cluster = {i: [] for i in range(1, n_clusters + 1)}
    sheet_by_cluster = {i: [] for i in range(1, n_clusters + 1)}

    for p, label in zip(patterns, labels):
        helix_by_cluster[label].append(p['helix_frac'])
        sheet_by_cluster[label].append(p['sheet_frac'])

    # Simple F-statistic approximation
    all_helix = [p['helix_frac'] for p in patterns]
    all_sheet = [p['sheet_frac'] for p in patterns]

    helix_means = [np.mean(helix_by_cluster[i]) for i in range(1, n_clusters + 1)]
    sheet_means = [np.mean(sheet_by_cluster[i]) for i in range(1, n_clusters + 1)]

    helix_variance_between = np.var(helix_means) * len(patterns) / n_clusters
    helix_variance_within = np.mean([np.var(helix_by_cluster[i]) for i in range(1, n_clusters + 1)])

    sheet_variance_between = np.var(sheet_means) * len(patterns) / n_clusters
    sheet_variance_within = np.mean([np.var(sheet_by_cluster[i]) for i in range(1, n_clusters + 1)])

    print(f"\nHelix fraction:")
    print(f"  Variance between clusters: {helix_variance_between:.4f}")
    print(f"  Variance within clusters:  {helix_variance_within:.4f}")
    print(f"  Ratio (higher = clusters differ): {helix_variance_between / helix_variance_within:.2f}")

    print(f"\nSheet fraction:")
    print(f"  Variance between clusters: {sheet_variance_between:.4f}")
    print(f"  Variance within clusters:  {sheet_variance_within:.4f}")
    print(f"  Ratio (higher = clusters differ): {sheet_variance_between / sheet_variance_within:.2f}")

    # Save results
    results = {
        'n_clusters': n_clusters,
        'cluster_stats': {k: {key: val for key, val in v.items() if key != 'proteins'}
                         for k, v in cluster_stats.items()},
        'cluster_assignments': {p['name']: int(label) for p, label in zip(patterns, labels)},
        'ss_cluster_correlation': {
            'helix_variance_ratio': float(helix_variance_between / helix_variance_within),
            'sheet_variance_ratio': float(sheet_variance_between / sheet_variance_within),
        }
    }

    with open(output_dir / 'clustering_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_dir / 'clustering_results.json'}")

    # Final verdict
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    helix_ratio = helix_variance_between / helix_variance_within
    sheet_ratio = sheet_variance_between / sheet_variance_within

    if helix_ratio > 2 or sheet_ratio > 2:
        print("\n→ CLUSTERS SHOW SS-SPECIFIC PATTERNS")
        print("  Skip schedules appear correlated with secondary structure content!")
    elif helix_ratio > 1.2 or sheet_ratio > 1.2:
        print("\n→ WEAK SS CORRELATION DETECTED")
        print("  Some relationship between skip patterns and SS, but not dominant")
    else:
        print("\n→ SKIP PATTERNS ARE SEQUENCE-SPECIFIC, NOT SS-SPECIFIC")
        print("  Each protein has unique refinement dynamics independent of SS composition")


if __name__ == "__main__":
    main()
