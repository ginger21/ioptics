"""ioptics Performance Benchmark Script

Measures execution time and memory for:
1. Transfer Matrix construction (get_transfer_matrix)
2. Forward pass latency (Batch sizes: 1, 32)
3. Backward pass latency (Default, AdjointVariant, ParameterShift)

Establishes a quantitative baseline across network scales (N = 4, 8, 16, 32, 64).
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

from ioptics.backward_functions import BackwardType
from ioptics.layers import ClementsMesh


def benchmark_fn(
    fn,
    warmup: int = 3,
    repeats: int = 10,
) -> Dict[str, float]:
    """Run function with warmup and return median and std latency in milliseconds."""
    for _ in range(warmup):
        fn()

    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    times_tensor = torch.tensor(times, dtype=torch.float64)
    median = float(torch.median(times_tensor).item())
    mean = float(torch.mean(times_tensor).item())
    std = float(torch.std(times_tensor).item()) if len(times) > 1 else 0.0

    return {
        "median_ms": median,
        "mean_ms": mean,
        "std_ms": std,
        "min_ms": float(torch.min(times_tensor).item()),
        "max_ms": float(torch.max(times_tensor).item()),
    }


def run_benchmarks(
    scales: List[int],
    repeats: int = 5,
    warmup: int = 2,
    skip_param_shift_threshold: int = 64,
) -> Dict[str, Any]:
    print("=" * 80, flush=True)
    print(f"Running ioptics Performance Benchmark (Scales: {scales})", flush=True)
    print(f"Warmup runs: {warmup}, Timing repeats: {repeats}", flush=True)
    print("=" * 80, flush=True)

    results: Dict[str, Any] = {
        "metadata": {
            "torch_version": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"),
            "scales": scales,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "benchmarks": {},
    }

    for N in scales:
        print(f"\n--- Benchmark Scale N = {N} (Components: ~{N*(N-1)//2}, Columns: {N}) ---", flush=True)
        scale_result: Dict[str, Any] = {}

        # 1. Transfer Matrix Build (Uncached)
        mesh_default = ClementsMesh(L=N, backward_type=BackwardType.DEFAULT.value)
        tm_stats = benchmark_fn(
            lambda: mesh_default.get_transfer_matrix(use_cache=False),
            warmup=warmup,
            repeats=repeats,
        )
        scale_result["transfer_matrix_build_ms"] = tm_stats
        print(f"  Transfer Matrix Build:   {tm_stats['median_ms']:.4f} ms (std: {tm_stats['std_ms']:.4f})", flush=True)

        # 2. Forward Pass (Batch size = 1)
        x_b1 = torch.randn(1, N, dtype=torch.complex64)
        fwd_b1_stats = benchmark_fn(
            lambda: mesh_default(x_b1),
            warmup=warmup,
            repeats=repeats,
        )
        scale_result["forward_b1_ms"] = fwd_b1_stats
        print(f"  Forward Pass (Batch=1):  {fwd_b1_stats['median_ms']:.4f} ms (std: {fwd_b1_stats['std_ms']:.4f})", flush=True)

        # 3. Forward Pass (Batch size = 32)
        x_b32 = torch.randn(32, N, dtype=torch.complex64)
        fwd_b32_stats = benchmark_fn(
            lambda: mesh_default(x_b32),
            warmup=warmup,
            repeats=repeats,
        )
        scale_result["forward_b32_ms"] = fwd_b32_stats
        print(f"  Forward Pass (Batch=32): {fwd_b32_stats['median_ms']:.4f} ms (std: {fwd_b32_stats['std_ms']:.4f})", flush=True)

        # 3b. Eval Forward Pass (Cached Static Matrix, Batch=1)
        mesh_default.eval()
        with torch.no_grad():
            eval_b1_stats = benchmark_fn(
                lambda: mesh_default(x_b1),
                warmup=warmup,
                repeats=repeats,
            )
        mesh_default.train()
        scale_result["eval_forward_b1_ms"] = eval_b1_stats
        print(f"  Eval Forward (Cached):   {eval_b1_stats['median_ms']:.4f} ms (std: {eval_b1_stats['std_ms']:.4f})", flush=True)

        # 4. Backward Pass: Default Autograd
        def run_backward_default():
            x = x_b32.clone().requires_grad_(True)
            out = mesh_default(x)
            loss = (out.abs() ** 2).sum()
            loss.backward()

        bwd_def_stats = benchmark_fn(
            run_backward_default,
            warmup=warmup,
            repeats=repeats,
        )
        scale_result["backward_default_ms"] = bwd_def_stats
        print(f"  Backward (Autograd):     {bwd_def_stats['median_ms']:.4f} ms (std: {bwd_def_stats['std_ms']:.4f})", flush=True)

        # 5. Backward Pass: Adjoint Variant
        mesh_adj = ClementsMesh(L=N, backward_type=BackwardType.ADJOINT_VARIANT.value)

        def run_backward_adjoint():
            x = x_b32.clone().requires_grad_(True)
            out = mesh_adj(x)
            loss = (out.abs() ** 2).sum()
            loss.backward()

        bwd_adj_stats = benchmark_fn(
            run_backward_adjoint,
            warmup=warmup,
            repeats=repeats,
        )
        scale_result["backward_adjoint_ms"] = bwd_adj_stats
        print(f"  Backward (Adjoint):      {bwd_adj_stats['median_ms']:.4f} ms (std: {bwd_adj_stats['std_ms']:.4f})", flush=True)

        # 6. Backward Pass: Parameter Shift (Skip for N > skip_param_shift_threshold)
        if N <= skip_param_shift_threshold:
            mesh_ps = ClementsMesh(L=N, backward_type=BackwardType.PARAMETER_SHIFT.value)

            def run_backward_param_shift():
                x = x_b32.clone().requires_grad_(True)
                out = mesh_ps(x)
                loss = (out.abs() ** 2).sum()
                loss.backward()

            bwd_ps_stats = benchmark_fn(
                run_backward_param_shift,
                warmup=1,
                repeats=2,
            )
            scale_result["backward_param_shift_ms"] = bwd_ps_stats
            print(f"  Backward (ParamShift):   {bwd_ps_stats['median_ms']:.4f} ms (std: {bwd_ps_stats['std_ms']:.4f})", flush=True)
        else:
            scale_result["backward_param_shift_ms"] = "skipped_due_to_scale"
            print(f"  Backward (ParamShift):   Skipped (scale N={N} > {skip_param_shift_threshold})", flush=True)

        results["benchmarks"][f"N_{N}"] = scale_result

    return results


def print_summary_table(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print("Benchmark Baseline Summary (Median Latency in milliseconds)")
    print("=" * 90)
    header = f"{'Scale (N)':<10} | {'TM Build':<12} | {'Fwd (B=1)':<12} | {'Fwd (B=32)':<12} | {'Bwd (Def)':<12} | {'Bwd (Adj)':<12} | {'Bwd (PS)':<12}"
    print(header)
    print("-" * 90)

    for scale_key, data in results["benchmarks"].items():
        n_str = scale_key.replace("N_", "")
        tm = f"{data['transfer_matrix_build_ms']['median_ms']:.3f}"
        fwd1 = f"{data['forward_b1_ms']['median_ms']:.3f}"
        fwd32 = f"{data['forward_b32_ms']['median_ms']:.3f}"
        bwd_def = f"{data['backward_default_ms']['median_ms']:.3f}"
        bwd_adj = f"{data['backward_adjoint_ms']['median_ms']:.3f}"
        ps = data["backward_param_shift_ms"]
        ps_str = f"{ps['median_ms']:.3f}" if isinstance(ps, dict) else "N/A"

        row = f"{n_str:<10} | {tm:<12} | {fwd1:<12} | {fwd32:<12} | {bwd_def:<12} | {bwd_adj:<12} | {ps_str:<12}"
        print(row)

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="ioptics Performance Benchmark")
    parser.add_argument("--scales", type=str, default="4,8,16,32,64", help="Comma-separated scales (default: 4,8,16,32,64)")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timing repeats (default: 5)")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations (default: 2)")
    parser.add_argument("--output", type=str, default="benchmarks/baseline_results.json", help="Output JSON path")
    args = parser.parse_args()

    scales = [int(s.strip()) for s in args.scales.split(",") if s.strip()]
    results = run_benchmarks(scales=scales, repeats=args.repeats, warmup=args.warmup)

    print_summary_table(results)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nBaseline results saved to: {args.output}")


if __name__ == "__main__":
    main()
