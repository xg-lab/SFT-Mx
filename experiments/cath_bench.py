#!/usr/bin/env python3
"""
CATH S40 Benchmark: SimpleFold + TeaCache vs CATH Ground Truth

Uses maximally diverse CATH S40 structures for rigorous benchmarking.
Includes TeaCache ablation: no cache (threshold=0) vs cache (threshold=0.1).

Creates publication-quality figures showing:
- Quality (TM-score, RMSD, lDDT) vs compute tradeoff
- Inference time vs sequence length
- TeaCache speedup analysis

Prerequisites:
    1. Run: python select_diverse_cath.py  (creates diverse_cath_300.json)
    2. Run: python server.py --port 8888
"""

import os
import sys
import time
import json
import requests
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional, List
from scipy.spatial.transform import Rotation

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# Configuration
# =============================================================================

SERVER_URL = "http://0.0.0.0:8888"
CATH_BENCHMARK_FILE = Path("cath_benchmark/diverse_cath_300.json")
OUTPUT_DIR = Path("cath_benchmark")
OUTPUT_DIR.mkdir(exist_ok=True)

# Models to benchmark
MODELS = ['simplefold_100M', 'simplefold_360M', 'simplefold_700M',
          'simplefold_1.1B', 'simplefold_1.6B', 'simplefold_3B']

# TeaCache thresholds: 0 = no caching, 0.1 = aggressive caching
TEACACHE_THRESHOLDS = [0.0, 0.1]

PARALLEL_JOBS = 4

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str
    model: str
    teacache_threshold: float
    length: int
    sequence: str
    tm_score: Optional[float] = None
    rmsd: Optional[float] = None
    lddt: Optional[float] = None
    inference_time: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    error: Optional[str] = None

# =============================================================================
# Structure Parsing
# =============================================================================

def parse_pdb_ca(pdb_path: Path) -> np.ndarray:
    """Extract CA coordinates from PDB file."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') and ' CA ' in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords)

def parse_cif_ca(cif_content: str) -> Optional[np.ndarray]:
    """Extract CA coordinates from CIF content."""
    coords = []
    for line in cif_content.split('\n'):
        if line.startswith('ATOM') and ' CA ' in line:
            parts = line.split()
            for i, p in enumerate(parts):
                try:
                    if '.' in p and -1000 < float(p) < 1000:
                        if i + 2 < len(parts):
                            x = float(parts[i])
                            y = float(parts[i+1])
                            z = float(parts[i+2])
                            coords.append([x, y, z])
                            break
                except ValueError:
                    continue
    return np.array(coords) if coords else None

# =============================================================================
# Structure Comparison Metrics
# =============================================================================

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Align P onto Q using Kabsch algorithm, return aligned P."""
    P_center = P.mean(axis=0)
    Q_center = Q.mean(axis=0)
    P_c = P - P_center
    Q_c = Q - Q_center
    R, _ = Rotation.align_vectors(Q_c, P_c)
    return P_c @ R.as_matrix().T + Q_center

def compute_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute RMSD after Kabsch alignment."""
    P_aligned = kabsch_align(P, Q)
    return np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1)))

def compute_tm_score(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute TM-score."""
    P_aligned = kabsch_align(P, Q)
    L = len(Q)
    d0 = max(1.24 * (L - 15) ** (1/3) - 1.8, 0.5)
    distances = np.sqrt(np.sum((P_aligned - Q) ** 2, axis=1))
    return np.sum(1 / (1 + (distances / d0) ** 2)) / L

def compute_lddt(P: np.ndarray, Q: np.ndarray, cutoff: float = 15.0) -> float:
    """Compute lDDT (local Distance Difference Test)."""
    n = len(P)
    if n < 2:
        return 0.0

    # Pairwise distances
    P_dist = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
    Q_dist = np.sqrt(np.sum((Q[:, None, :] - Q[None, :, :]) ** 2, axis=2))

    # Only consider pairs within cutoff in reference
    mask = (Q_dist < cutoff) & (np.arange(n)[:, None] != np.arange(n)[None, :])

    if mask.sum() == 0:
        return 0.0

    diff = np.abs(P_dist - Q_dist)

    # lDDT thresholds: 0.5, 1, 2, 4 Angstroms
    scores = []
    for t in [0.5, 1.0, 2.0, 4.0]:
        preserved = (diff < t) & mask
        scores.append(preserved.sum() / mask.sum())

    return np.mean(scores)

# =============================================================================
# Server Client
# =============================================================================

def check_server():
    """Check if server is running."""
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False

def submit_fold_job(name: str, sequence: str, model: str = None, teacache_threshold: float = None):
    """Submit fold job (async, returns job_id)."""
    data = {
        'name': name,
        'sequences': [{'proteinChain': {'sequence': sequence, 'count': 1}}]
    }
    if model:
        data['model'] = model
    if teacache_threshold is not None:
        data['teacache_threshold'] = teacache_threshold

    r = requests.post(f"{SERVER_URL}/v1/fold", json=data, timeout=30)
    return r.json()

def poll_job(job_id: str, timeout: int = 600, poll_interval: float = 1.0):
    """Poll for job completion."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{SERVER_URL}/v1/job/{job_id}", timeout=30)
            job = r.json()
            if job['status'] == 'completed':
                return job
            if job['status'] == 'failed':
                return job
        except requests.exceptions.Timeout:
            pass
        time.sleep(poll_interval)
    return {'status': 'timeout', 'error': 'Job timed out'}

def fold_protein(name: str, sequence: str, model: str = None, teacache_threshold: float = None):
    """Fold protein using server API (sync wrapper)."""
    submit_result = submit_fold_job(name, sequence, model, teacache_threshold)
    if 'job_id' not in submit_result:
        return submit_result
    return poll_job(submit_result['job_id'])

def get_result_cif(job_id: str) -> str:
    """Download result CIF content."""
    r = requests.get(f"{SERVER_URL}/v1/job/{job_id}/result", timeout=30)
    return r.text

# =============================================================================
# Benchmark Runner
# =============================================================================

def benchmark_one(struct: dict, model: str, teacache_threshold: float) -> BenchmarkResult:
    """Benchmark a single structure with specific model and cache setting."""
    name = struct['name']
    sequence = struct['sequence']
    pdb_path = Path(struct['path'])
    length = struct['length']

    try:
        # Fold
        cache_str = f"tc{teacache_threshold}"
        result = fold_protein(f"{name}_{model}_{cache_str}", sequence, model, teacache_threshold)

        if result.get('status') != 'completed':
            return BenchmarkResult(
                name=name, model=model, teacache_threshold=teacache_threshold,
                length=length, sequence=sequence,
                error=result.get('error', 'Unknown')
            )

        res = result['results'][0]
        inference_time = res['inference_time_s']
        cache_hit_rate = res.get('cache_hit_rate', 0)

        # Get prediction
        cif_content = get_result_cif(result['job_id'])
        pred_coords = parse_cif_ca(cif_content)
        gt_coords = parse_pdb_ca(pdb_path)

        if pred_coords is None or len(pred_coords) < 10 or len(gt_coords) < 10:
            return BenchmarkResult(
                name=name, model=model, teacache_threshold=teacache_threshold,
                length=length, sequence=sequence,
                inference_time=inference_time, cache_hit_rate=cache_hit_rate,
                error="Failed to parse coordinates"
            )

        # Align lengths
        min_len = min(len(pred_coords), len(gt_coords))
        pred_coords = pred_coords[:min_len]
        gt_coords = gt_coords[:min_len]

        # Compute metrics
        tm_score = compute_tm_score(pred_coords, gt_coords)
        rmsd = compute_rmsd(pred_coords, gt_coords)
        lddt = compute_lddt(pred_coords, gt_coords)

        return BenchmarkResult(
            name=name, model=model, teacache_threshold=teacache_threshold,
            length=length, sequence=sequence,
            tm_score=tm_score, rmsd=rmsd, lddt=lddt,
            inference_time=inference_time, cache_hit_rate=cache_hit_rate
        )

    except Exception as e:
        return BenchmarkResult(
            name=name, model=model, teacache_threshold=teacache_threshold,
            length=length, sequence=sequence,
            error=str(e)
        )

def process_job(args):
    """Process a single benchmark job (for thread pool)."""
    struct, model, threshold = args
    return benchmark_one(struct, model, threshold)

# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("CATH S40 BENCHMARK: SimpleFold + TeaCache Ablation")
    print("=" * 70)

    # Check server
    print("\nChecking server...")
    if not check_server():
        print(f"ERROR: Server not running at {SERVER_URL}")
        print("Start with: python server.py --port 8888")
        sys.exit(1)
    print("Server OK!")

    # Load structures
    print(f"\nLoading diverse CATH set from {CATH_BENCHMARK_FILE}...")
    if not CATH_BENCHMARK_FILE.exists():
        print("ERROR: Run python select_diverse_cath.py first!")
        sys.exit(1)

    with open(CATH_BENCHMARK_FILE) as f:
        structures = json.load(f)

    print(f"Loaded {len(structures)} structures")
    lengths = [s['length'] for s in structures]
    print(f"Length range: {min(lengths)}-{max(lengths)} aa")

    # ==========================================================================
    # Run Benchmark
    # ==========================================================================

    all_results: List[BenchmarkResult] = []

    for model in MODELS:
        for threshold in TEACACHE_THRESHOLDS:
            cache_label = "NO_CACHE" if threshold == 0 else f"CACHE_{threshold}"

            print(f"\n{'='*70}")
            print(f"MODEL: {model} | TeaCache: {cache_label}")
            print('='*70)

            jobs = [(s, model, threshold) for s in structures]

            completed = 0
            with ThreadPoolExecutor(max_workers=PARALLEL_JOBS) as executor:
                futures = {executor.submit(process_job, job): job for job in jobs}

                for future in as_completed(futures):
                    completed += 1
                    result = future.result()

                    if result.error:
                        print(f"[{completed}/{len(jobs)}] {result.name[:20]}... ERROR: {result.error}")
                    else:
                        print(f"[{completed}/{len(jobs)}] {result.name[:20]}... "
                              f"TM={result.tm_score:.3f} | RMSD={result.rmsd:.1f}Å | "
                              f"lDDT={result.lddt:.3f} | Time={result.inference_time:.1f}s | "
                              f"Cache={result.cache_hit_rate:.0%}")
                        all_results.append(result)

    # ==========================================================================
    # Save Results
    # ==========================================================================

    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # CSV
    csv_path = OUTPUT_DIR / "cath_benchmark_full.csv"
    with open(csv_path, 'w') as f:
        f.write("model,teacache_threshold,name,length,tm_score,rmsd,lddt,time_s,cache_hit\n")
        for r in all_results:
            f.write(f"{r.model},{r.teacache_threshold},{r.name},{r.length},"
                    f"{r.tm_score:.4f},{r.rmsd:.2f},{r.lddt:.4f},"
                    f"{r.inference_time:.2f},{r.cache_hit_rate:.3f}\n")
    print(f"Saved: {csv_path}")

    # JSON
    json_path = OUTPUT_DIR / "cath_benchmark_full.json"
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"Saved: {json_path}")

    # ==========================================================================
    # Generate Figures
    # ==========================================================================

    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import pandas as pd

    df = pd.DataFrame([asdict(r) for r in all_results])

    # Color scheme
    model_colors = {
        'simplefold_100M': '#4169E1',
        'simplefold_360M': '#32CD32',
        'simplefold_700M': "#B86720",
        'simplefold_1.1B': '#FF6347',
        'simplefold_1.6B': "#DE60D6",
        'simplefold_3B': "#9D47FF",
    }

    # ==========================================================================
    # Figure 1: Quality vs Speed (with cache comparison)
    # ==========================================================================

    fig1, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Quality vs Speed (bubble plot)
    ax = axes[0]
    sizes = {'simplefold_100M': 150, 'simplefold_360M': 400,
             'simplefold_700M': 800, 'simplefold_1.1B': 1200,
             'simplefold_1.6B': 1800, 'simplefold_3B': 3000}

    for model in MODELS:
        for threshold in TEACACHE_THRESHOLDS:
            mask = (df['model'] == model) & (df['teacache_threshold'] == threshold)
            model_df = df[mask]
            if len(model_df) == 0:
                continue

            avg_tm = model_df['tm_score'].mean()
            avg_time = model_df['time_s' if 'time_s' in df.columns else 'inference_time'].mean()

            marker = 'o' if threshold > 0 else 's'  # circle=cache, square=no cache
            alpha = 0.8 if threshold > 0 else 0.4
            label = f"{model.replace('simplefold_', '')} ({'cache' if threshold > 0 else 'no cache'})"

            ax.scatter(avg_time, avg_tm, s=sizes[model], c=model_colors[model],
                      alpha=alpha, marker=marker, edgecolors='black', linewidth=1.5,
                      label=label)

    ax.set_xlabel('Avg Inference Time (seconds)', fontsize=12)
    ax.set_ylabel('Avg TM-score vs CATH', fontsize=12)
    ax.set_title('Quality vs Speed (○=TeaCache, □=No Cache)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8, ncol=2)

    # Panel B: TeaCache Speedup
    ax = axes[1]
    speedups = []
    for model in MODELS:
        no_cache = df[(df['model'] == model) & (df['teacache_threshold'] == 0)]
        with_cache = df[(df['model'] == model) & (df['teacache_threshold'] == 0.1)]

        if len(no_cache) > 0 and len(with_cache) > 0:
            time_col = 'time_s' if 'time_s' in df.columns else 'inference_time'
            t_nocache = no_cache[time_col].mean()
            t_cache = with_cache[time_col].mean()
            speedup = t_nocache / t_cache if t_cache > 0 else 1.0
            speedups.append({
                'model': model.replace('simplefold_', ''),
                'speedup': speedup,
                't_nocache': t_nocache,
                't_cache': t_cache,
                'color': model_colors[model]
            })

    if speedups:
        x = range(len(speedups))
        bars = ax.bar(x, [s['speedup'] for s in speedups],
                      color=[s['color'] for s in speedups], alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([s['model'] for s in speedups], rotation=45, ha='right')
        ax.set_ylabel('Speedup (no cache / with cache)', fontsize=12)
        ax.set_title('TeaCache Speedup by Model', fontsize=13, fontweight='bold')
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='No speedup')
        ax.grid(True, alpha=0.3, axis='y')

        # Add speedup labels
        for i, s in enumerate(speedups):
            ax.annotate(f"{s['speedup']:.2f}x", (i, s['speedup']), ha='center',
                       va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig1_path = OUTPUT_DIR / "cath_quality_speed.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fig1_path}")

    # ==========================================================================
    # Figure 2: Time vs Length (with cache comparison)
    # ==========================================================================

    fig2, ax = plt.subplots(figsize=(14, 8))

    time_col = 'time_s' if 'time_s' in df.columns else 'inference_time'

    for model in MODELS:
        for threshold in TEACACHE_THRESHOLDS:
            mask = (df['model'] == model) & (df['teacache_threshold'] == threshold)
            model_df = df[mask].sort_values('length')
            if len(model_df) == 0:
                continue

            linestyle = '-' if threshold > 0 else '--'
            alpha = 0.8 if threshold > 0 else 0.4
            label = f"{model.replace('simplefold_', '')} ({'cache' if threshold > 0 else 'no cache'})"

            ax.scatter(model_df['length'], model_df[time_col],
                      s=40, c=model_colors[model], alpha=alpha, edgecolors='black', linewidth=0.3)

            # Trend line
            if len(model_df) > 5:
                window = max(3, len(model_df) // 10)
                rolling = model_df[time_col].rolling(window=window, center=True, min_periods=1).mean()
                ax.plot(model_df['length'], rolling, color=model_colors[model],
                       linewidth=2.5, alpha=alpha, linestyle=linestyle, label=label)

    ax.set_xlabel('Sequence Length (residues)', fontsize=12)
    ax.set_ylabel('Inference Time (seconds)', fontsize=12)
    ax.set_title('Inference Time vs Length (solid=TeaCache, dashed=No Cache)', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fig2_path = OUTPUT_DIR / "cath_time_vs_length.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fig2_path}")

    # ==========================================================================
    # Figure 3: Summary Statistics
    # ==========================================================================

    fig3, axes = plt.subplots(1, 3, figsize=(15, 5))

    # TM-score distribution
    ax = axes[0]
    for model in MODELS:
        model_df = df[(df['model'] == model) & (df['teacache_threshold'] == 0.1)]
        if len(model_df) > 0:
            ax.hist(model_df['tm_score'], bins=20, alpha=0.5, label=model.replace('simplefold_', ''),
                   color=model_colors[model], edgecolor='black')
    ax.set_xlabel('TM-score')
    ax.set_ylabel('Count')
    ax.set_title('TM-score Distribution (with TeaCache)')
    ax.legend()

    # RMSD distribution
    ax = axes[1]
    for model in MODELS:
        model_df = df[(df['model'] == model) & (df['teacache_threshold'] == 0.1)]
        if len(model_df) > 0:
            ax.hist(model_df['rmsd'], bins=20, alpha=0.5, label=model.replace('simplefold_', ''),
                   color=model_colors[model], edgecolor='black')
    ax.set_xlabel('RMSD (Å)')
    ax.set_ylabel('Count')
    ax.set_title('RMSD Distribution (with TeaCache)')

    # lDDT distribution
    ax = axes[2]
    for model in MODELS:
        model_df = df[(df['model'] == model) & (df['teacache_threshold'] == 0.1)]
        if len(model_df) > 0:
            ax.hist(model_df['lddt'], bins=20, alpha=0.5, label=model.replace('simplefold_', ''),
                   color=model_colors[model], edgecolor='black')
    ax.set_xlabel('lDDT')
    ax.set_ylabel('Count')
    ax.set_title('lDDT Distribution (with TeaCache)')

    plt.tight_layout()
    fig3_path = OUTPUT_DIR / "cath_distributions.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fig3_path}")

    # ==========================================================================
    # Print Summary
    # ==========================================================================

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    time_col = 'time_s' if 'time_s' in df.columns else 'inference_time'

    for model in MODELS:
        print(f"\n{model}:")
        for threshold in TEACACHE_THRESHOLDS:
            mask = (df['model'] == model) & (df['teacache_threshold'] == threshold)
            model_df = df[mask]
            if len(model_df) == 0:
                continue
            cache_str = "TeaCache" if threshold > 0 else "No Cache"
            print(f"  [{cache_str}]")
            print(f"    TM-score: {model_df['tm_score'].mean():.3f} ± {model_df['tm_score'].std():.3f}")
            print(f"    RMSD:     {model_df['rmsd'].mean():.2f} ± {model_df['rmsd'].std():.2f} Å")
            print(f"    lDDT:     {model_df['lddt'].mean():.3f} ± {model_df['lddt'].std():.3f}")
            print(f"    Time:     {model_df[time_col].mean():.1f} ± {model_df[time_col].std():.1f} s")
            if threshold > 0:
                print(f"    Cache:    {model_df['cache_hit_rate'].mean()*100:.1f}%")

    # TeaCache Impact Summary
    print(f"\n{'='*70}")
    print("TEACACHE IMPACT SUMMARY")
    print("="*70)

    for model in MODELS:
        no_cache = df[(df['model'] == model) & (df['teacache_threshold'] == 0)]
        with_cache = df[(df['model'] == model) & (df['teacache_threshold'] == 0.1)]

        if len(no_cache) > 0 and len(with_cache) > 0:
            t_nc = no_cache[time_col].mean()
            t_c = with_cache[time_col].mean()
            speedup = t_nc / t_c if t_c > 0 else 1.0
            tm_diff = with_cache['tm_score'].mean() - no_cache['tm_score'].mean()

            print(f"\n{model}:")
            print(f"  Speedup: {speedup:.2f}x ({t_nc:.1f}s → {t_c:.1f}s)")
            print(f"  TM-score change: {tm_diff:+.4f}")

    print("\n✅ CATH BENCHMARK COMPLETE!")


if __name__ == "__main__":
    main()
