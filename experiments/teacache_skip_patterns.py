#!/usr/bin/env python3
"""
TeaCache Skip Pattern Analysis

Directly track which timesteps TeaCache skips for diverse proteins.
Visualize: step (x) vs skip decision (y=0 computed, y=1 skipped)

Key questions:
1. Are the same steps always skipped across proteins?
2. Is there protein-specific variation?
3. What's the temporal pattern of skips?

Usage:
    python teacache_skip_patterns.py --num-proteins 10 --num-steps 500
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'src/simplefold')

import mlx.core as mx


@dataclass
class SkipPattern:
    """Skip pattern for a single protein."""
    name: str
    sequence: str
    num_residues: int
    num_atoms: int
    num_steps: int
    skip_mask: List[int]  # 0=computed, 1=skipped
    cache_hits: int
    cache_misses: int
    hit_rate: float


class SkipTrackingTeaCacheSampler:
    """
    TeaCache sampler that tracks WHICH steps are skipped.
    """

    def __init__(
        self,
        num_timesteps: int = 500,
        t_start: float = 1e-4,
        tau: float = 0.1,
        threshold: float = 0.1,
        warmup_steps: int = 10,
        log_timesteps: bool = True,
        w_cutoff: float = 0.99,
    ):
        self.num_timesteps = num_timesteps
        self.t_start = t_start
        self.tau = tau
        self.threshold = threshold
        self.warmup_steps = warmup_steps
        self.w_cutoff = w_cutoff

        # Timestep schedule
        if log_timesteps:
            self.steps = mx.exp(mx.linspace(
                np.log(t_start), np.log(1.0), num=num_timesteps + 1
            ))
        else:
            self.steps = mx.linspace(t_start, 1.0, num=num_timesteps + 1)

        # Cache state
        self.cached_output = None
        self.accumulated_diff = 0.0
        self.prev_modulated_input = None

        # Skip tracking
        self.skip_mask = []  # 0=computed, 1=skipped

    def _compute_modulated_input(self, noised_pos: mx.array, t: mx.array) -> float:
        """Compute timestep-modulated input representation."""
        t_val = t.item() if hasattr(t, 'item') else float(t)
        pos_norm = mx.sqrt((noised_pos ** 2).sum(axis=-1).mean())
        modulated = pos_norm * (1.0 - t_val + 0.1)
        return modulated.item() if hasattr(modulated, 'item') else float(modulated)

    def _compute_input_diff(self, current: float, prev: float) -> float:
        """Compute L1 relative difference."""
        if prev is None:
            return float('inf')
        return abs(current - prev) / (abs(prev) + 1e-6)

    def _should_recompute(self, step_idx: int, input_diff: float) -> bool:
        """Decide whether to recompute or use cache."""
        if step_idx < self.warmup_steps:
            return True
        if self.cached_output is None:
            return True

        self.accumulated_diff += input_diff
        if self.accumulated_diff >= self.threshold:
            self.accumulated_diff = 0.0
            return True
        return False

    def diffusion_coefficient(self, t, eps=0.01):
        """Compute diffusion coefficient."""
        w = (1.0 - t) / (t + eps)
        t_val = t.item() if hasattr(t, 'item') else float(t)
        if t_val >= self.w_cutoff:
            w = 0.0
        return w

    def sample_with_tracking(
        self,
        model_fn,
        flow,
        noise: mx.array,
        batch: Dict,
        verbose: bool = True,
    ) -> Tuple[mx.array, Dict]:
        """Sample while tracking which steps are skipped."""
        from tqdm import tqdm
        from einops.array_api import repeat
        from utils.mlx_utils import center_random_augmentation

        y = noise
        steps = self.steps

        # Reset state
        self.cached_output = None
        self.prev_modulated_input = None
        self.accumulated_diff = 0.0
        self.skip_mask = []

        cache_hits = 0
        cache_misses = 0

        iterator = tqdm(range(self.num_timesteps), desc="TeaCache Tracking") if verbose else range(self.num_timesteps)

        for i in iterator:
            t = steps[i]
            t_next = steps[i + 1]
            dt = t_next - t
            eps = mx.random.normal(y.shape)

            y = center_random_augmentation(
                y, batch["atom_pad_mask"],
                augmentation=False, centering=True,
            )

            # Compute modulated input
            modulated_input = self._compute_modulated_input(y, t)
            input_diff = self._compute_input_diff(modulated_input, self.prev_modulated_input)

            # Cache decision
            if self._should_recompute(i, input_diff):
                # Computed (not skipped)
                batched_t = repeat(t, " -> b", b=y.shape[0])
                output = model_fn(noised_pos=y, t=batched_t, feats=batch)
                velocity = output["predict_velocity"]
                self.cached_output = velocity
                self.skip_mask.append(0)  # 0 = computed
                cache_misses += 1
            else:
                # Skipped (used cache)
                velocity = self.cached_output
                self.skip_mask.append(1)  # 1 = skipped
                cache_hits += 1

            self.prev_modulated_input = modulated_input

            # Euler-Maruyama step
            score = flow.compute_score_from_velocity(velocity, y, t)
            diff_coeff = self.diffusion_coefficient(t)
            drift = velocity + diff_coeff * score
            mean_y = y + drift * dt
            y = mean_y + mx.sqrt(2.0 * dt * diff_coeff * self.tau) * eps

            mx.eval(y)

        return y, {
            'skip_mask': self.skip_mask,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'hit_rate': cache_hits / (cache_hits + cache_misses),
        }


def create_skip_pattern_plot(
    patterns: List[SkipPattern],
    output_path: Path,
    title: str = "TeaCache Skip Patterns",
):
    """
    Create visualization of skip patterns across proteins.

    Plot: step (x) vs protein (y), colored by skip decision
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    num_proteins = len(patterns)
    num_steps = patterns[0].num_steps

    # Build matrix: proteins × steps
    matrix = np.zeros((num_proteins, num_steps))
    for i, p in enumerate(patterns):
        matrix[i, :] = p.skip_mask

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel A: Heatmap of skip patterns
    ax = axes[0, 0]
    cmap = ListedColormap(['steelblue', 'lightcoral'])  # computed=blue, skipped=red
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, interpolation='nearest')

    ax.set_xlabel('Diffusion Step', fontsize=12)
    ax.set_ylabel('Protein', fontsize=12)
    ax.set_title('Skip Patterns by Protein\n(blue=computed, red=skipped)', fontsize=13)

    # Add protein names as y-ticks
    ax.set_yticks(range(num_proteins))
    ax.set_yticklabels([f"{p.name} ({p.num_residues}aa)" for p in patterns], fontsize=8)

    # Panel B: Skip probability by step
    ax = axes[0, 1]
    skip_prob = matrix.mean(axis=0)  # Average across proteins
    steps = np.arange(num_steps)

    ax.fill_between(steps, skip_prob, alpha=0.3, color='red')
    ax.plot(steps, skip_prob, 'r-', linewidth=1.5, label='Skip probability')

    # Highlight warmup region
    warmup = 10
    ax.axvline(warmup, color='green', linestyle='--', label=f'Warmup ({warmup} steps)')
    ax.axhline(0.9, color='gray', linestyle=':', alpha=0.5, label='90% skip rate')

    ax.set_xlabel('Diffusion Step', fontsize=12)
    ax.set_ylabel('Skip Probability', fontsize=12)
    ax.set_title('Skip Probability by Step (averaged across proteins)', fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Cumulative skips
    ax = axes[1, 0]
    cumulative_skips = np.cumsum(skip_prob)
    total_possible = np.arange(1, num_steps + 1)
    cumulative_pct = cumulative_skips / total_possible * 100

    ax.plot(steps, cumulative_pct, 'b-', linewidth=2)
    ax.axhline(90, color='red', linestyle='--', label='90% skip rate')

    ax.set_xlabel('Diffusion Step', fontsize=12)
    ax.set_ylabel('Cumulative Skip Rate (%)', fontsize=12)
    ax.set_title('Cumulative Skip Rate Over Diffusion', fontsize=13)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel D: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    hit_rates = [p.hit_rate for p in patterns]
    skip_counts = [sum(p.skip_mask) for p in patterns]

    # Check for consistent patterns
    # Compute correlation between protein skip patterns
    if num_proteins > 1:
        correlations = []
        for i in range(num_proteins):
            for j in range(i+1, num_proteins):
                corr = np.corrcoef(patterns[i].skip_mask, patterns[j].skip_mask)[0, 1]
                correlations.append(corr)
        mean_corr = np.mean(correlations)
    else:
        mean_corr = 1.0

    # Find consistently skipped/computed steps
    always_skipped = (matrix.sum(axis=0) == num_proteins).sum()
    always_computed = (matrix.sum(axis=0) == 0).sum()
    variable_steps = num_steps - always_skipped - always_computed

    summary_text = f"""
    TEACACHE SKIP PATTERN SUMMARY
    =============================

    Proteins analyzed: {num_proteins}
    Steps per protein: {num_steps}

    Overall Statistics:
    • Mean skip rate: {np.mean(hit_rates)*100:.1f}% ± {np.std(hit_rates)*100:.1f}%
    • Mean skipped steps: {np.mean(skip_counts):.0f} / {num_steps}
    • Mean computed steps: {num_steps - np.mean(skip_counts):.0f} / {num_steps}

    Pattern Consistency:
    • Steps ALWAYS computed: {always_computed} ({100*always_computed/num_steps:.1f}%)
    • Steps ALWAYS skipped: {always_skipped} ({100*always_skipped/num_steps:.1f}%)
    • Steps with VARIABLE behavior: {variable_steps} ({100*variable_steps/num_steps:.1f}%)
    • Mean pairwise correlation: {mean_corr:.3f}

    Interpretation:
    {"• Skip patterns are CONSISTENT across proteins" if mean_corr > 0.8 else "• Skip patterns VARY by protein" if mean_corr < 0.5 else "• Skip patterns show MODERATE consistency"}
    {"• Most steps have deterministic behavior" if variable_steps < num_steps * 0.2 else "• Many steps have protein-dependent behavior"}
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return {
        'mean_skip_rate': np.mean(hit_rates),
        'mean_correlation': mean_corr,
        'always_computed': always_computed,
        'always_skipped': always_skipped,
        'variable_steps': variable_steps,
    }


def create_step_detail_plot(
    patterns: List[SkipPattern],
    output_path: Path,
):
    """Create detailed step-by-step analysis."""
    import matplotlib.pyplot as plt

    num_proteins = len(patterns)
    num_steps = patterns[0].num_steps

    # Build matrix
    matrix = np.array([p.skip_mask for p in patterns])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Early vs Late phase comparison
    ax = axes[0, 0]
    early_end = num_steps // 5  # First 20%
    late_start = 4 * num_steps // 5  # Last 20%

    early_skip_rates = [sum(p.skip_mask[:early_end]) / early_end for p in patterns]
    late_skip_rates = [sum(p.skip_mask[late_start:]) / (num_steps - late_start) for p in patterns]

    x = np.arange(num_proteins)
    width = 0.35
    ax.bar(x - width/2, early_skip_rates, width, label=f'Early (0-{early_end})', color='blue', alpha=0.7)
    ax.bar(x + width/2, late_skip_rates, width, label=f'Late ({late_start}-{num_steps})', color='red', alpha=0.7)

    ax.set_xlabel('Protein Index', fontsize=12)
    ax.set_ylabel('Skip Rate', fontsize=12)
    ax.set_title('Early vs Late Phase Skip Rates', fontsize=13)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel B: Skip run lengths
    ax = axes[0, 1]
    all_run_lengths = []
    for p in patterns:
        # Find consecutive skip runs
        run_length = 0
        for skip in p.skip_mask:
            if skip == 1:
                run_length += 1
            else:
                if run_length > 0:
                    all_run_lengths.append(run_length)
                run_length = 0
        if run_length > 0:
            all_run_lengths.append(run_length)

    if all_run_lengths:
        ax.hist(all_run_lengths, bins=range(1, max(all_run_lengths)+2), color='coral',
                edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(all_run_lengths), color='red', linestyle='--',
                   label=f'Mean: {np.mean(all_run_lengths):.1f}')
    ax.set_xlabel('Consecutive Skip Run Length', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Skip Run Lengths', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Skip probability by step phase
    ax = axes[1, 0]
    phases = 10
    phase_size = num_steps // phases
    phase_skip_rates = []
    phase_labels = []

    for i in range(phases):
        start = i * phase_size
        end = (i + 1) * phase_size if i < phases - 1 else num_steps
        phase_rates = matrix[:, start:end].mean()
        phase_skip_rates.append(phase_rates)
        phase_labels.append(f'{start}-{end}')

    ax.bar(range(phases), phase_skip_rates, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Step Phase', fontsize=12)
    ax.set_ylabel('Skip Rate', fontsize=12)
    ax.set_title('Skip Rate by Diffusion Phase', fontsize=13)
    ax.set_xticks(range(phases))
    ax.set_xticklabels(phase_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1)
    ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90%')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel D: Protein size vs skip rate
    ax = axes[1, 1]
    sizes = [p.num_residues for p in patterns]
    rates = [p.hit_rate for p in patterns]

    ax.scatter(sizes, rates, s=100, alpha=0.7, c='steelblue', edgecolors='black')

    # Fit line
    if len(sizes) > 2:
        z = np.polyfit(sizes, rates, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(sizes), max(sizes), 100)
        ax.plot(x_line, p(x_line), 'r--', label=f'Trend (slope={z[0]:.4f})')

    ax.set_xlabel('Protein Size (residues)', fontsize=12)
    ax.set_ylabel('Skip Rate', fontsize=12)
    ax.set_title('Skip Rate vs Protein Size', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


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


def main():
    parser = argparse.ArgumentParser(description='TeaCache Skip Pattern Analysis')
    parser.add_argument('--num-proteins', type=int, default=10, help='Number of proteins')
    parser.add_argument('--num-steps', type=int, default=500, help='Diffusion steps')
    parser.add_argument('--threshold', type=float, default=0.1, help='TeaCache threshold')
    parser.add_argument('--model', type=str, default='simplefold_100M', help='Model')
    parser.add_argument('--output-dir', type=str, default='teacache_patterns', help='Output dir')
    parser.add_argument('--start-idx', type=int, default=0, help='Starting protein index')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("TEACACHE SKIP PATTERN ANALYSIS")
    print(f"Tracking which steps are skipped across {args.num_proteins} proteins")
    print("=" * 70)

    # Load sequences
    fasta_path = Path("cath_benchmark/diverse_cath_300.fasta")
    print(f"\nLoading sequences from {fasta_path}...")
    sequences = load_fasta_sequences(fasta_path)
    print(f"Loaded {len(sequences)} sequences")

    # Initialize inference
    print(f"\nInitializing inference pipeline...")
    print(f"  Model: {args.model}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Threshold: {args.threshold}")

    from wrapper import ModelWrapper, InferenceWrapper

    model_wrapper = ModelWrapper(
        simplefold_model=args.model,
        plddt=False,
        ckpt_dir='artifacts',
        backend='mlx',
    )

    inference_wrapper = InferenceWrapper(
        output_dir=str(output_dir / 'cache'),
        prediction_dir='predictions',
        num_steps=args.num_steps,
        nsample_per_protein=1,
        tau=0.1,
        device='cpu',
        backend='mlx',
    )

    model = model_wrapper.from_pretrained_folding_model()
    print("Models loaded!")

    # Create skip tracking sampler
    sampler = SkipTrackingTeaCacheSampler(
        num_timesteps=args.num_steps,
        threshold=args.threshold,
        tau=0.1,
        log_timesteps=True,
    )

    # Analyze proteins
    patterns = []

    for i in range(args.start_idx, min(args.start_idx + args.num_proteins, len(sequences))):
        name, sequence = sequences[i]
        print(f"\n{'='*70}")
        print(f"[{i+1-args.start_idx}/{args.num_proteins}] {name} ({len(sequence)} residues)")
        print("="*70)

        # Prepare batch
        print("  Preparing batch...")
        try:
            batch, structure, record = inference_wrapper.process_input(sequence)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        num_atoms = int(batch['atom_pad_mask'][0].sum().item())
        print(f"  Atoms: {num_atoms}")

        # Run with tracking
        noise = mx.random.normal(batch['coords'].shape)

        print(f"  Running TeaCache with {args.num_steps} steps...")
        denoised, tracking = sampler.sample_with_tracking(
            model_fn=model,
            flow=inference_wrapper.flow,
            noise=noise,
            batch=batch,
            verbose=True,
        )

        print(f"  Cache hits: {tracking['cache_hits']}, misses: {tracking['cache_misses']}")
        print(f"  Skip rate: {tracking['hit_rate']*100:.1f}%")

        patterns.append(SkipPattern(
            name=name,
            sequence=sequence,
            num_residues=len(sequence),
            num_atoms=num_atoms,
            num_steps=args.num_steps,
            skip_mask=tracking['skip_mask'],
            cache_hits=tracking['cache_hits'],
            cache_misses=tracking['cache_misses'],
            hit_rate=tracking['hit_rate'],
        ))

        # Free memory and clear cache
        del batch, structure, record, noise, denoised
        import gc
        gc.collect()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()

    if not patterns:
        print("No patterns collected!")
        return

    # Create visualizations
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    main_plot_path = output_dir / "skip_patterns.png"
    summary_stats = create_skip_pattern_plot(patterns, main_plot_path,
                                              f"TeaCache Skip Patterns (threshold={args.threshold}, {args.num_steps} steps)")
    print(f"  {main_plot_path}")

    detail_plot_path = output_dir / "skip_details.png"
    create_step_detail_plot(patterns, detail_plot_path)
    print(f"  {detail_plot_path}")

    # Save results
    results = {
        'config': {
            'num_proteins': len(patterns),
            'num_steps': args.num_steps,
            'threshold': args.threshold,
            'model': args.model,
        },
        'summary': summary_stats,
        'patterns': [
            {
                'name': p.name,
                'sequence': p.sequence,
                'num_residues': p.num_residues,
                'num_atoms': p.num_atoms,
                'skip_mask': p.skip_mask,
                'cache_hits': p.cache_hits,
                'cache_misses': p.cache_misses,
                'hit_rate': p.hit_rate,
            }
            for p in patterns
        ]
    }

    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    results_path = output_dir / "skip_patterns.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"  {results_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"Mean skip rate: {summary_stats['mean_skip_rate']*100:.1f}%")
    print(f"Mean pairwise correlation: {summary_stats['mean_correlation']:.3f}")
    print(f"Steps always computed: {summary_stats['always_computed']}/{args.num_steps}")
    print(f"Steps always skipped: {summary_stats['always_skipped']}/{args.num_steps}")
    print(f"Steps with variable behavior: {summary_stats['variable_steps']}/{args.num_steps}")

    if summary_stats['mean_correlation'] > 0.8:
        print("\n-> Skip patterns are HIGHLY CONSISTENT across proteins")
        print("   This suggests a universal temporal structure to diffusion!")
    elif summary_stats['mean_correlation'] > 0.5:
        print("\n-> Skip patterns show MODERATE consistency")
        print("   Some steps are universal, others protein-dependent")
    else:
        print("\n-> Skip patterns VARY significantly by protein")
        print("   Each protein has unique refinement dynamics")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
