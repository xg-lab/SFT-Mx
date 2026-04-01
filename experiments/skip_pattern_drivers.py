#!/usr/bin/env python3
"""
Analyze what drives variation in skip patterns.

Investigates: size, sequence complexity, specific AA composition
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def compute_sequence_features(sequence: str) -> dict:
    """Compute various sequence-based features."""
    if not sequence:
        return {}

    n = len(sequence)

    # AA composition
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

    # Charge
    pos_charge = sum(1 for aa in sequence if aa in 'KRH')
    neg_charge = sum(1 for aa in sequence if aa in 'DE')
    net_charge = (pos_charge - neg_charge) / n

    # Disorder propensity (simplified)
    disorder_prone = set('PSEKQRGD')
    disorder_score = sum(1 for aa in sequence if aa in disorder_prone) / n

    # Aromatic content
    aromatic = sum(1 for aa in sequence if aa in 'FYW') / n

    # Proline content (helix breaker)
    proline = aa_counts.get('P', 0) / n

    # Glycine content (flexibility)
    glycine = aa_counts.get('G', 0) / n

    # Cysteine content (disulfide potential)
    cysteine = aa_counts.get('C', 0) / n

    # Sequence complexity (Shannon entropy)
    probs = np.array(list(aa_counts.values())) / n
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    return {
        'length': n,
        'hydrophobicity': hydro,
        'net_charge': net_charge,
        'disorder_score': disorder_score,
        'aromatic_fraction': aromatic,
        'proline_fraction': proline,
        'glycine_fraction': glycine,
        'cysteine_fraction': cysteine,
        'sequence_entropy': entropy
    }


def compute_skip_metrics(skip_mask: list) -> dict:
    """Extract detailed metrics from skip pattern."""
    mask = np.array(skip_mask)
    n = len(mask)

    # Overall
    skip_rate = np.mean(mask)

    # Phase-specific rates
    phases = {
        'warmup': (0, 20),
        'early': (20, 100),
        'mid_early': (100, 200),
        'mid': (200, 300),
        'mid_late': (300, 400),
        'late': (400, 480),
        'final': (480, 500)
    }

    phase_rates = {}
    for name, (start, end) in phases.items():
        phase_rates[f'skip_{name}'] = np.mean(mask[start:end])

    # Transition characteristics
    transitions = np.abs(np.diff(mask))
    n_transitions = np.sum(transitions)

    # First skip after warmup
    warmup_end = 11
    try:
        first_skip_after_warmup = np.where(mask[warmup_end:] == 1)[0][0] + warmup_end
    except:
        first_skip_after_warmup = warmup_end

    # Mean compute interval (average steps between computes)
    compute_steps = np.where(mask == 0)[0]
    if len(compute_steps) > 1:
        compute_intervals = np.diff(compute_steps)
        mean_compute_interval = np.mean(compute_intervals)
        max_compute_interval = np.max(compute_intervals)
    else:
        mean_compute_interval = 0
        max_compute_interval = 0

    return {
        'skip_rate': skip_rate,
        **phase_rates,
        'n_transitions': n_transitions,
        'first_skip_after_warmup': first_skip_after_warmup,
        'mean_compute_interval': mean_compute_interval,
        'max_compute_interval': max_compute_interval
    }


def main():
    print("Loading skip patterns...")
    with open('publications/data/skip_patterns.json') as f:
        data = json.load(f)

    patterns = data['patterns']
    print(f"Loaded {len(patterns)} proteins")

    # Compute features for all proteins
    results = []
    for p in patterns:
        seq_features = compute_sequence_features(p.get('sequence', ''))
        skip_metrics = compute_skip_metrics(p['skip_mask'])

        results.append({
            'name': p['name'],
            **seq_features,
            **skip_metrics
        })

    # Convert to arrays for correlation analysis
    features = ['length', 'hydrophobicity', 'net_charge', 'disorder_score',
                'aromatic_fraction', 'proline_fraction', 'glycine_fraction',
                'cysteine_fraction', 'sequence_entropy']

    skip_targets = ['skip_rate', 'n_transitions', 'mean_compute_interval',
                    'max_compute_interval', 'skip_warmup', 'skip_mid', 'skip_late']

    print("\n" + "="*70)
    print("CORRELATION ANALYSIS: What drives skip pattern variation?")
    print("="*70)

    # Compute correlations
    correlations = {}
    for target in skip_targets:
        target_vals = np.array([r[target] for r in results])
        if np.std(target_vals) < 1e-10:
            continue

        correlations[target] = {}
        for feat in features:
            feat_vals = np.array([r.get(feat, 0) for r in results])
            if np.std(feat_vals) < 1e-10:
                continue
            r, p = pearsonr(feat_vals, target_vals)
            correlations[target][feat] = (r, p)

    # Print significant correlations
    for target in skip_targets:
        if target not in correlations:
            continue
        print(f"\n{target}:")
        sorted_corrs = sorted(correlations[target].items(),
                             key=lambda x: abs(x[1][0]), reverse=True)
        for feat, (r, p) in sorted_corrs[:5]:
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {feat:25s}: r={r:+.3f} {sig}")

    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('Skip Pattern Drivers Analysis', fontsize=14)

    # 1. Size vs skip rate
    ax = axes[0, 0]
    sizes = [r['length'] for r in results]
    skip_rates = [r['skip_rate'] for r in results]
    ax.scatter(sizes, skip_rates, alpha=0.5, s=30)
    r, p = pearsonr(sizes, skip_rates)
    ax.set_xlabel('Protein Length (residues)')
    ax.set_ylabel('Skip Rate')
    ax.set_title(f'Length vs Skip Rate (r={r:.3f})')

    # 2. Size vs number of transitions
    ax = axes[0, 1]
    n_trans = [r['n_transitions'] for r in results]
    ax.scatter(sizes, n_trans, alpha=0.5, s=30)
    r, p = pearsonr(sizes, n_trans)
    ax.set_xlabel('Protein Length (residues)')
    ax.set_ylabel('Number of Transitions')
    ax.set_title(f'Length vs Transitions (r={r:.3f})')

    # 3. Size vs mean compute interval
    ax = axes[0, 2]
    intervals = [r['mean_compute_interval'] for r in results]
    ax.scatter(sizes, intervals, alpha=0.5, s=30)
    r, p = pearsonr(sizes, intervals)
    ax.set_xlabel('Protein Length (residues)')
    ax.set_ylabel('Mean Compute Interval')
    ax.set_title(f'Length vs Compute Interval (r={r:.3f})')

    # 4. Disorder score vs skip pattern
    ax = axes[1, 0]
    disorder = [r['disorder_score'] for r in results]
    ax.scatter(disorder, skip_rates, alpha=0.5, s=30, c='green')
    r, p = pearsonr(disorder, skip_rates)
    ax.set_xlabel('Disorder Propensity Score')
    ax.set_ylabel('Skip Rate')
    ax.set_title(f'Disorder vs Skip Rate (r={r:.3f})')

    # 5. Hydrophobicity vs skip pattern
    ax = axes[1, 1]
    hydro = [r['hydrophobicity'] for r in results]
    ax.scatter(hydro, skip_rates, alpha=0.5, s=30, c='orange')
    r, p = pearsonr(hydro, skip_rates)
    ax.set_xlabel('Mean Hydrophobicity')
    ax.set_ylabel('Skip Rate')
    ax.set_title(f'Hydrophobicity vs Skip Rate (r={r:.3f})')

    # 6. Sequence entropy vs transitions
    ax = axes[1, 2]
    entropy = [r['sequence_entropy'] for r in results]
    ax.scatter(entropy, n_trans, alpha=0.5, s=30, c='purple')
    r, p = pearsonr(entropy, n_trans)
    ax.set_xlabel('Sequence Entropy')
    ax.set_ylabel('Number of Transitions')
    ax.set_title(f'Entropy vs Transitions (r={r:.3f})')

    plt.tight_layout()
    plt.savefig('publications/data/skip_drivers.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: publications/data/skip_drivers.png")

    # Size-binned analysis
    print("\n" + "="*70)
    print("SIZE-BINNED ANALYSIS")
    print("="*70)

    # Bin by size
    size_bins = [(0, 80), (80, 120), (120, 180), (180, 500)]

    for low, high in size_bins:
        bin_prots = [r for r in results if low <= r['length'] < high]
        if not bin_prots:
            continue

        mean_skip = np.mean([r['skip_rate'] for r in bin_prots])
        mean_trans = np.mean([r['n_transitions'] for r in bin_prots])
        mean_interval = np.mean([r['mean_compute_interval'] for r in bin_prots])

        print(f"\nSize {low}-{high} ({len(bin_prots)} proteins):")
        print(f"  Mean skip rate: {mean_skip:.3f}")
        print(f"  Mean transitions: {mean_trans:.1f}")
        print(f"  Mean compute interval: {mean_interval:.1f}")

    # Final summary
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    # Find strongest correlations across all
    all_corrs = []
    for target, feats in correlations.items():
        for feat, (r, p) in feats.items():
            all_corrs.append((target, feat, r, p))

    all_corrs.sort(key=lambda x: abs(x[2]), reverse=True)

    print("\nStrongest correlations overall:")
    for target, feat, r, p in all_corrs[:10]:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {feat:20s} → {target:25s}: r={r:+.3f} {sig}")

    # Check if any correlation explains substantial variance
    max_r = max(abs(x[2]) for x in all_corrs)
    if max_r > 0.5:
        print(f"\n→ Found strong correlation (r={max_r:.2f}) - skip patterns have detectable drivers!")
    elif max_r > 0.3:
        print(f"\n→ Moderate correlations found (max r={max_r:.2f}) - partial predictability")
    else:
        print(f"\n→ Weak correlations only (max r={max_r:.2f}) - skip patterns are largely sequence-idiosyncratic")


if __name__ == "__main__":
    main()
