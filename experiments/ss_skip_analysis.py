#!/usr/bin/env python3
"""
Secondary Structure Skip Pattern Analysis

Analyzes TeaCache skip patterns in relation to secondary structure content.
Uses phi/psi backbone angles to classify residues as helix/sheet/coil.

Usage:
    python ss_skip_analysis.py --patterns teacache_patterns_300/skip_patterns.json
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'src/simplefold')


def compute_dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """Compute dihedral angle between 4 points in radians."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    if n1_norm < 1e-6 or n2_norm < 1e-6:
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    m1 = np.cross(n1, b2 / np.linalg.norm(b2))

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return np.arctan2(y, x)


def compute_phi_psi(coords: np.ndarray, atom_names: List[str], residue_indices: List[int]) -> Tuple[List[float], List[float]]:
    """
    Compute phi/psi angles for each residue.

    Args:
        coords: (N_atoms, 3) array of coordinates
        atom_names: List of atom names (N, CA, C, O, CB, ...)
        residue_indices: List of residue index for each atom

    Returns:
        phi_angles: List of phi angles (one per residue, nan for first)
        psi_angles: List of psi angles (one per residue, nan for last)
    """
    # Build residue -> backbone atom mapping
    residue_atoms = {}
    for i, (name, res_idx) in enumerate(zip(atom_names, residue_indices)):
        if res_idx not in residue_atoms:
            residue_atoms[res_idx] = {}
        if name in ['N', 'CA', 'C']:
            residue_atoms[res_idx][name] = coords[i]

    n_residues = max(residue_indices) + 1
    phi_angles = [np.nan] * n_residues
    psi_angles = [np.nan] * n_residues

    for res_idx in range(n_residues):
        if res_idx not in residue_atoms:
            continue

        curr = residue_atoms.get(res_idx, {})
        prev = residue_atoms.get(res_idx - 1, {})
        next_res = residue_atoms.get(res_idx + 1, {})

        # Phi: C(i-1) - N(i) - CA(i) - C(i)
        if 'C' in prev and 'N' in curr and 'CA' in curr and 'C' in curr:
            phi_angles[res_idx] = compute_dihedral(
                prev['C'], curr['N'], curr['CA'], curr['C']
            )

        # Psi: N(i) - CA(i) - C(i) - N(i+1)
        if 'N' in curr and 'CA' in curr and 'C' in curr and 'N' in next_res:
            psi_angles[res_idx] = compute_dihedral(
                curr['N'], curr['CA'], curr['C'], next_res['N']
            )

    return phi_angles, psi_angles


def classify_ss_from_angles(phi: float, psi: float) -> str:
    """
    Classify secondary structure from phi/psi angles.

    Regions (in degrees):
    - Alpha helix: phi ~ -60, psi ~ -45
    - Beta sheet: phi ~ -120, psi ~ +120
    - Everything else: coil
    """
    if np.isnan(phi) or np.isnan(psi):
        return 'C'  # Coil/unknown

    # Convert to degrees
    phi_deg = np.degrees(phi)
    psi_deg = np.degrees(psi)

    # Alpha helix region
    if -100 < phi_deg < -30 and -80 < psi_deg < 0:
        return 'H'

    # Beta sheet region
    if (-180 < phi_deg < -60 or 150 < phi_deg < 180) and (80 < psi_deg < 180 or -180 < psi_deg < -120):
        return 'E'

    # 3-10 helix (near alpha)
    if -90 < phi_deg < -40 and -30 < psi_deg < 30:
        return 'H'

    return 'C'


@dataclass
class ProteinSSProfile:
    """Secondary structure profile for a protein."""
    name: str
    num_residues: int
    ss_sequence: str  # H=helix, E=sheet, C=coil
    helix_fraction: float
    sheet_fraction: float
    coil_fraction: float
    skip_mask: List[int]
    skip_rate: float


def predict_ss_from_sequence(sequence: str) -> str:
    """
    Simple SS prediction based on sequence propensities.
    This is a fallback when we don't have structures.

    Uses amino acid propensities for helix/sheet.
    """
    # Helix propensities (Chou-Fasman)
    helix_formers = set('AELM')
    helix_breakers = set('PG')
    sheet_formers = set('VIY')
    sheet_breakers = set('PDE')

    ss = []
    for aa in sequence:
        if aa in helix_formers:
            ss.append('H')
        elif aa in sheet_formers:
            ss.append('E')
        else:
            ss.append('C')

    return ''.join(ss)


def analyze_ss_patterns(patterns_path: Path, output_dir: Path):
    """
    Analyze skip patterns in relation to secondary structure.
    """
    print("Loading skip patterns...")
    with open(patterns_path) as f:
        data = json.load(f)

    patterns = data['patterns']
    num_steps = data['num_steps']

    print(f"Loaded {len(patterns)} proteins, {num_steps} steps each")

    # Group proteins by SS content
    profiles = []

    for p in patterns:
        # Use sequence-based SS prediction as proxy
        ss_seq = predict_ss_from_sequence(p['sequence'] if 'sequence' in p else 'A' * p['num_residues'])

        helix_frac = ss_seq.count('H') / len(ss_seq) if len(ss_seq) > 0 else 0
        sheet_frac = ss_seq.count('E') / len(ss_seq) if len(ss_seq) > 0 else 0
        coil_frac = ss_seq.count('C') / len(ss_seq) if len(ss_seq) > 0 else 0

        profiles.append(ProteinSSProfile(
            name=p['name'],
            num_residues=p['num_residues'],
            ss_sequence=ss_seq,
            helix_fraction=helix_frac,
            sheet_fraction=sheet_frac,
            coil_fraction=coil_frac,
            skip_mask=p['skip_mask'],
            skip_rate=p['hit_rate'],
        ))

    # Analyze correlations
    helix_fracs = [p.helix_fraction for p in profiles]
    sheet_fracs = [p.sheet_fraction for p in profiles]
    skip_rates = [p.skip_rate for p in profiles]
    sizes = [p.num_residues for p in profiles]

    # Correlation analysis
    helix_corr = np.corrcoef(helix_fracs, skip_rates)[0, 1]
    sheet_corr = np.corrcoef(sheet_fracs, skip_rates)[0, 1]
    size_corr = np.corrcoef(sizes, skip_rates)[0, 1]

    print(f"\nCorrelations with skip rate:")
    print(f"  Helix fraction: r = {helix_corr:.3f}")
    print(f"  Sheet fraction: r = {sheet_corr:.3f}")
    print(f"  Protein size:   r = {size_corr:.3f}")

    # Bin proteins by SS content
    high_helix = [p for p in profiles if p.helix_fraction > 0.4]
    high_sheet = [p for p in profiles if p.sheet_fraction > 0.3]
    mixed = [p for p in profiles if p.helix_fraction < 0.3 and p.sheet_fraction < 0.3]

    print(f"\nProteins by SS class:")
    print(f"  High helix (>40%): {len(high_helix)}, mean skip rate: {np.mean([p.skip_rate for p in high_helix]):.3f}" if high_helix else "  High helix: 0")
    print(f"  High sheet (>30%): {len(high_sheet)}, mean skip rate: {np.mean([p.skip_rate for p in high_sheet]):.3f}" if high_sheet else "  High sheet: 0")
    print(f"  Mixed/coil:        {len(mixed)}, mean skip rate: {np.mean([p.skip_rate for p in mixed]):.3f}" if mixed else "  Mixed: 0")

    # Analyze skip patterns by step for each SS class
    def compute_step_skip_probs(profile_list):
        if not profile_list:
            return np.zeros(num_steps)
        masks = np.array([p.skip_mask for p in profile_list])
        return masks.mean(axis=0)

    helix_skip_by_step = compute_step_skip_probs(high_helix)
    sheet_skip_by_step = compute_step_skip_probs(high_sheet)
    mixed_skip_by_step = compute_step_skip_probs(mixed)
    all_skip_by_step = compute_step_skip_probs(profiles)

    # Create visualizations
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Secondary Structure vs Skip Patterns', fontsize=14)

    # Plot 1: Skip rate vs SS content
    ax1 = axes[0, 0]
    ax1.scatter(helix_fracs, skip_rates, alpha=0.5, c='red', label='Helix', s=30)
    ax1.scatter(sheet_fracs, skip_rates, alpha=0.5, c='blue', label='Sheet', s=30)
    ax1.set_xlabel('SS Fraction')
    ax1.set_ylabel('Skip Rate')
    ax1.set_title(f'Skip Rate vs SS Content\n(Helix r={helix_corr:.2f}, Sheet r={sheet_corr:.2f})')
    ax1.legend()
    ax1.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Skip probability by step for each SS class
    ax2 = axes[0, 1]
    steps = np.arange(num_steps)
    if len(high_helix) > 5:
        ax2.plot(steps, helix_skip_by_step, 'r-', alpha=0.7, label=f'High Helix (n={len(high_helix)})')
    if len(high_sheet) > 5:
        ax2.plot(steps, sheet_skip_by_step, 'b-', alpha=0.7, label=f'High Sheet (n={len(high_sheet)})')
    if len(mixed) > 5:
        ax2.plot(steps, mixed_skip_by_step, 'g-', alpha=0.7, label=f'Mixed/Coil (n={len(mixed)})')
    ax2.plot(steps, all_skip_by_step, 'k--', alpha=0.5, label='All proteins')
    ax2.set_xlabel('Diffusion Step')
    ax2.set_ylabel('Skip Probability')
    ax2.set_title('Skip Probability by Step (by SS Class)')
    ax2.legend()
    ax2.set_ylim(-0.05, 1.05)

    # Plot 3: Difference from mean (helix vs sheet)
    ax3 = axes[1, 0]
    if len(high_helix) > 5 and len(high_sheet) > 5:
        helix_diff = helix_skip_by_step - all_skip_by_step
        sheet_diff = sheet_skip_by_step - all_skip_by_step
        ax3.plot(steps, helix_diff, 'r-', alpha=0.7, label='High Helix - Mean')
        ax3.plot(steps, sheet_diff, 'b-', alpha=0.7, label='High Sheet - Mean')
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.fill_between(steps, helix_diff, alpha=0.2, color='red')
        ax3.fill_between(steps, sheet_diff, alpha=0.2, color='blue')
    ax3.set_xlabel('Diffusion Step')
    ax3.set_ylabel('Δ Skip Probability')
    ax3.set_title('Deviation from Mean Skip Pattern by SS Class')
    ax3.legend()

    # Plot 4: Phase analysis by SS class
    ax4 = axes[1, 1]
    phases = ['0-50', '50-100', '100-200', '200-300', '300-400', '400-500']
    phase_ranges = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, 500)]

    def compute_phase_rates(profile_list):
        if not profile_list:
            return [0] * len(phases)
        rates = []
        for start, end in phase_ranges:
            phase_skips = [np.mean(p.skip_mask[start:end]) for p in profile_list]
            rates.append(np.mean(phase_skips))
        return rates

    x = np.arange(len(phases))
    width = 0.25

    if len(high_helix) > 5:
        ax4.bar(x - width, compute_phase_rates(high_helix), width, label='High Helix', color='red', alpha=0.7)
    if len(high_sheet) > 5:
        ax4.bar(x, compute_phase_rates(high_sheet), width, label='High Sheet', color='blue', alpha=0.7)
    if len(mixed) > 5:
        ax4.bar(x + width, compute_phase_rates(mixed), width, label='Mixed/Coil', color='green', alpha=0.7)

    ax4.set_xlabel('Diffusion Phase')
    ax4.set_ylabel('Skip Rate')
    ax4.set_title('Skip Rate by Phase and SS Class')
    ax4.set_xticks(x)
    ax4.set_xticklabels(phases, rotation=45)
    ax4.legend()
    ax4.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'ss_skip_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'ss_skip_analysis.png'}")

    # Additional analysis: which steps differ most between SS classes?
    if len(high_helix) > 5 and len(high_sheet) > 5:
        diff = np.abs(helix_skip_by_step - sheet_skip_by_step)
        top_diff_steps = np.argsort(diff)[-20:][::-1]
        print(f"\nSteps with largest helix vs sheet skip difference:")
        for step in top_diff_steps[:10]:
            print(f"  Step {step}: helix={helix_skip_by_step[step]:.2f}, sheet={sheet_skip_by_step[step]:.2f}, diff={diff[step]:.2f}")

    # Save results
    results = {
        'num_proteins': len(profiles),
        'num_steps': num_steps,
        'correlations': {
            'helix_fraction_vs_skip_rate': float(helix_corr),
            'sheet_fraction_vs_skip_rate': float(sheet_corr),
            'size_vs_skip_rate': float(size_corr),
        },
        'ss_classes': {
            'high_helix': {
                'count': len(high_helix),
                'mean_skip_rate': float(np.mean([p.skip_rate for p in high_helix])) if high_helix else 0,
            },
            'high_sheet': {
                'count': len(high_sheet),
                'mean_skip_rate': float(np.mean([p.skip_rate for p in high_sheet])) if high_sheet else 0,
            },
            'mixed': {
                'count': len(mixed),
                'mean_skip_rate': float(np.mean([p.skip_rate for p in mixed])) if mixed else 0,
            },
        },
    }

    with open(output_dir / 'ss_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_dir / 'ss_analysis_results.json'}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze SS vs skip patterns')
    parser.add_argument('--patterns', type=str, default='teacache_patterns_300/skip_patterns.json',
                       help='Path to skip patterns JSON')
    parser.add_argument('--output-dir', type=str, default='teacache_patterns_300',
                       help='Output directory')
    args = parser.parse_args()

    analyze_ss_patterns(Path(args.patterns), Path(args.output_dir))


if __name__ == "__main__":
    main()
