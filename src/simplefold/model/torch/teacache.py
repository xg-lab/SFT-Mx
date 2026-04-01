#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#
# TeaCache: Training-free acceleration via timestep-aware caching
# Adapted from: https://github.com/ali-vilab/TeaCache
# Paper: "Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model"

import torch
from tqdm import tqdm
from einops import repeat
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from utils.boltz_utils import center_random_augmentation


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
    TeaCache-accelerated sampler for SimpleFold (PyTorch version).

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
        self.steps = torch.linspace(t_start, 1.0, steps=num_timesteps + 1)

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
        noised_pos: torch.Tensor,
        t: torch.Tensor,
        batch: Dict,
    ) -> torch.Tensor:
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
        pos_norm = torch.sqrt((noised_pos ** 2).sum(dim=-1).mean())

        # Add timestep modulation
        modulated = pos_norm * (1.0 - t_val + 0.1)  # +0.1 to avoid zero

        return modulated

    def _compute_input_diff(
        self,
        current_modulated: torch.Tensor,
        prev_modulated: torch.Tensor,
    ) -> float:
        """Compute L1 relative difference between modulated inputs."""
        if prev_modulated is None:
            return float('inf')

        # L1 relative distance
        diff = torch.abs(current_modulated - prev_modulated)
        rel_diff = diff / (torch.abs(prev_modulated) + 1e-6)

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

    @torch.no_grad()
    def sample(
        self,
        model_fn,
        flow,
        noise: torch.Tensor,
        batch: Dict,
        verbose: bool = True,
    ) -> Dict:
        """
        Sample with TeaCache acceleration.

        Returns dict with 'denoised_coords' and 'cache_stats'.
        """
        y = noise
        device = noise.device
        steps = self.steps.to(device)

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
            eps = torch.randn_like(y)

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
            y = mean_y + torch.sqrt(2.0 * dt * diff_coeff * self.tau) * eps

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
