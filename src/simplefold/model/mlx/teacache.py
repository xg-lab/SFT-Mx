#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#
# TeaCache: Training-free acceleration via timestep-aware caching
# Adapted from: https://github.com/ali-vilab/TeaCache
# Paper: "Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model"

import mlx.core as mx
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class TeaCacheConfig:
    """Configuration for TeaCache."""
    # Threshold for cache decision (lower = more quality, less speedup)
    threshold: float = 0.15  # 0.1 = slow/quality, 0.2 = fast

    # Rescaling coefficients
    poly_coeffs: Tuple[float, ...] = (0.0, 1.0)  # Linear: y = x

    # Warmup steps before caching kicks in (early steps are too different)
    warmup_steps: int = 10

    # Minimum steps between cache refreshes
    min_cache_interval: int = 1


class TeaCacheSampler:
    """
    TeaCache-accelerated sampler for SimpleFold.

    Key insight: Later timesteps have smaller output differences,
    so we can reuse cached velocity predictions more often.

    The backbone structure settles early, fine details refine later.
    """

    def __init__(
        self,
        num_timesteps: int = 500,
        t_start: float = 1e-4,
        tau: float = 0.3,
        config: Optional[TeaCacheConfig] = None,
    ):
        self.num_timesteps = num_timesteps
        self.t_start = t_start
        self.tau = tau
        self.config = config or TeaCacheConfig()

        # Create timestep schedule
        self.steps = mx.linspace(t_start, 1.0, num=num_timesteps + 1)

        # Cache state
        self.cached_output = None
        self.cached_t = None
        self.accumulated_diff = 0.0
        self.prev_modulated_input = None

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def _rescale(self, x: float) -> float:
        """Apply polynomial rescaling to input difference."""
        result = 0.0
        for i, coeff in enumerate(self.config.poly_coeffs):
            result += coeff * (x ** i)
        return result

    def _compute_modulated_input(
        self,
        noised_pos: mx.array,
        t: mx.array,
        batch: Dict,
    ) -> mx.array:
        """
        Compute timestep-modulated input representation.

        This approximates what TeaCache calls "modulating noisy inputs
        using timestep embeddings". We use a lightweight proxy that
        correlates with model output differences.
        """
        # Simple modulation: scale coordinates by timestep
        # At t=0 (noise), high variance; at t=1 (clean), low variance
        t_val = t.item() if hasattr(t, 'item') else float(t)

        # Combine position info with timestep
        # Using L2 norm of positions weighted by (1-t)
        pos_norm = mx.sqrt((noised_pos ** 2).sum(axis=-1).mean())

        # Add timestep modulation
        modulated = pos_norm * (1.0 - t_val + 0.1)  # +0.1 to avoid zero

        return modulated

    def _compute_input_diff(
        self,
        current_modulated: mx.array,
        prev_modulated: mx.array,
    ) -> float:
        """Compute L1 relative difference between modulated inputs."""
        if prev_modulated is None:
            return float('inf')

        # L1 relative distance
        diff = mx.abs(current_modulated - prev_modulated)
        rel_diff = diff / (mx.abs(prev_modulated) + 1e-6)

        return rel_diff.item() if hasattr(rel_diff, 'item') else float(rel_diff)

    def _should_recompute(self, step_idx: int, input_diff: float) -> bool:
        """Decide whether to recompute or use cache."""
        # Always compute during warmup
        if step_idx < self.config.warmup_steps:
            return True

        # Always compute if no cache
        if self.cached_output is None:
            return True

        # Rescale and accumulate
        rescaled_diff = self._rescale(input_diff)
        self.accumulated_diff += rescaled_diff

        # Check threshold
        if self.accumulated_diff >= self.config.threshold:
            self.accumulated_diff = 0.0
            return True

        return False

    def diffusion_coefficient(self, t, eps=0.01):
        """Compute diffusion coefficient (same as EMSampler)."""
        w = (1.0 - t) / (t + eps)
        t_val = t.item() if hasattr(t, 'item') else float(t)
        if t_val >= 0.99:
            w = 0.0
        return w

    def sample(
        self,
        model_fn,
        flow,
        noise: mx.array,
        batch: Dict,
        verbose: bool = True,
    ) -> Dict:
        """
        Sample with TeaCache acceleration.

        Returns dict with 'denoised_coords' and 'cache_stats'.
        """
        from tqdm import tqdm
        from einops.array_api import repeat
        from utils.mlx_utils import center_random_augmentation

        y = noise
        steps = self.steps

        # Reset cache state
        self.cached_output = None
        self.prev_modulated_input = None
        self.accumulated_diff = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

        iterator = tqdm(range(self.num_timesteps), desc="TeaCache Sampling") if verbose else range(self.num_timesteps)

        for i in iterator:
            t = steps[i]
            t_next = steps[i + 1]
            dt = t_next - t
            eps = mx.random.normal(y.shape)

            # Center structure
            y = center_random_augmentation(
                y,
                batch["atom_pad_mask"],
                augmentation=False,
                centering=True,
            )

            # Compute modulated input for cache decision
            modulated_input = self._compute_modulated_input(y, t, batch)
            input_diff = self._compute_input_diff(modulated_input, self.prev_modulated_input)

            # Cache decision
            if self._should_recompute(i, input_diff):
                # Cache miss - compute fresh
                batched_t = repeat(t, " -> b", b=y.shape[0])
                output = model_fn(noised_pos=y, t=batched_t, feats=batch)
                velocity = output["predict_velocity"]

                # Update cache
                self.cached_output = velocity
                self.cached_t = t
                self.cache_misses += 1
            else:
                # Cache hit - reuse previous
                velocity = self.cached_output
                self.cache_hits += 1

            # Update previous input
            self.prev_modulated_input = modulated_input

            # Euler-Maruyama step
            score = flow.compute_score_from_velocity(velocity, y, t)
            diff_coeff = self.diffusion_coefficient(t)
            drift = velocity + diff_coeff * score
            mean_y = y + drift * dt
            y = mean_y + mx.sqrt(2.0 * dt * diff_coeff * self.tau) * eps

            mx.eval(y)

        # Compute statistics
        total_steps = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_steps if total_steps > 0 else 0
        speedup = total_steps / self.cache_misses if self.cache_misses > 0 else 1.0

        return {
            "denoised_coords": y,
            "cache_stats": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": hit_rate,
                "theoretical_speedup": speedup,
            }
        }


def calibrate_teacache(
    model,
    flow,
    batch: Dict,
    num_calibration_runs: int = 3,
) -> Tuple[float, ...]:
    """
    Calibrate TeaCache polynomial coefficients for SimpleFold.

    Runs the model and measures actual input/output differences
    across timesteps to fit the rescaling polynomial.

    Returns polynomial coefficients.
    """
    from einops.array_api import repeat
    from utils.mlx_utils import center_random_augmentation
    import numpy as np

    print("Calibrating TeaCache for SimpleFold...")

    input_diffs = []
    output_diffs = []

    num_timesteps = 100  # Use fewer steps for calibration
    steps = mx.linspace(1e-4, 1.0, num=num_timesteps + 1)

    for run in range(num_calibration_runs):
        noise = mx.random.normal(batch['coords'].shape)
        y = noise

        prev_output = None
        prev_input_norm = None

        for i in range(num_timesteps):
            t = steps[i]
            t_next = steps[i + 1]
            dt = t_next - t

            y = center_random_augmentation(
                y, batch["atom_pad_mask"],
                augmentation=False, centering=True,
            )

            # Compute input representation
            pos_norm = mx.sqrt((y ** 2).sum(axis=-1).mean())
            input_repr = pos_norm * (1.0 - t.item() + 0.1)

            # Get model output
            batched_t = repeat(t, " -> b", b=y.shape[0])
            output = model(noised_pos=y, t=batched_t, feats=batch)
            velocity = output["predict_velocity"]

            # Compute differences
            if prev_output is not None and prev_input_norm is not None:
                # Input diff (L1 relative)
                in_diff = abs(input_repr.item() - prev_input_norm) / (abs(prev_input_norm) + 1e-6)

                # Output diff (L1 relative on velocity)
                out_diff = mx.abs(velocity - prev_output).mean()
                out_diff = out_diff / (mx.abs(prev_output).mean() + 1e-6)

                input_diffs.append(in_diff)
                output_diffs.append(out_diff.item())

            prev_output = velocity
            prev_input_norm = input_repr.item()

            # Euler step
            y = y + velocity * dt
            mx.eval(y)

    # Fit polynomial
    input_diffs = np.array(input_diffs)
    output_diffs = np.array(output_diffs)

    # Fit degree-2 polynomial
    coeffs = np.polyfit(input_diffs, output_diffs, 2)
    coeffs = tuple(reversed(coeffs))  # [a0, a1, a2] order

    print(f"  Calibration complete. Coefficients: {coeffs}")
    print(f"  Input diff range: [{input_diffs.min():.4f}, {input_diffs.max():.4f}]")
    print(f"  Output diff range: [{output_diffs.min():.4f}, {output_diffs.max():.4f}]")

    return coeffs


def benchmark_teacache(
    model,
    flow,
    batch: Dict,
    thresholds: list = [0.05, 0.1, 0.15, 0.2, 0.3],
) -> Dict:
    """
    Benchmark TeaCache at different thresholds.

    Returns dict with speedup/quality tradeoffs.
    """
    import time
    from model.mlx.sampler import EMSampler

    results = {}

    # Baseline (no caching)
    print("Running baseline (no caching)...")
    baseline_sampler = EMSampler(num_timesteps=500)
    noise = mx.random.normal(batch['coords'].shape)

    t0 = time.perf_counter()
    baseline_out = baseline_sampler.sample(model, flow, noise, batch)
    mx.eval(baseline_out['denoised_coords'])
    baseline_time = time.perf_counter() - t0
    baseline_coords = baseline_out['denoised_coords']

    results['baseline'] = {
        'time': baseline_time,
        'speedup': 1.0,
    }
    print(f"  Baseline: {baseline_time:.2f}s")

    # Test each threshold
    for threshold in thresholds:
        print(f"\nTesting threshold={threshold}...")
        config = TeaCacheConfig(threshold=threshold)
        sampler = TeaCacheSampler(num_timesteps=500, config=config)

        # Use same noise for fair comparison
        t0 = time.perf_counter()
        out = sampler.sample(model, flow, noise, batch)
        mx.eval(out['denoised_coords'])
        elapsed = time.perf_counter() - t0

        # Compute RMSD vs baseline
        diff = out['denoised_coords'] - baseline_coords
        mask = batch['atom_pad_mask']
        rmsd = mx.sqrt(((diff ** 2).sum(axis=-1) * mask).sum() / mask.sum())

        stats = out['cache_stats']
        actual_speedup = baseline_time / elapsed

        results[f'threshold_{threshold}'] = {
            'time': elapsed,
            'speedup': actual_speedup,
            'theoretical_speedup': stats['theoretical_speedup'],
            'hit_rate': stats['hit_rate'],
            'cache_hits': stats['hits'],
            'cache_misses': stats['misses'],
            'rmsd_vs_baseline': rmsd.item(),
        }

        print(f"  Time: {elapsed:.2f}s (speedup: {actual_speedup:.2f}x)")
        print(f"  Cache: {stats['hits']} hits, {stats['misses']} misses ({stats['hit_rate']*100:.1f}% hit rate)")
        print(f"  RMSD vs baseline: {rmsd.item():.4f} Å")

    return results
