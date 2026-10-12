import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from benchmarks.benchmark_baseline import benchmark_fn, run_benchmarks


def test_benchmark_fn_returns_valid_stats():
    def simple_add():
        a = torch.ones(10, 10)
        return a + a

    stats = benchmark_fn(simple_add, warmup=1, repeats=3)
    assert "median_ms" in stats
    assert "mean_ms" in stats
    assert "std_ms" in stats
    assert stats["median_ms"] > 0.0
    assert stats["max_ms"] >= stats["min_ms"]


def test_run_benchmarks_smoke():
    # Smoke test on small scale N=2,4
    results = run_benchmarks(scales=[2, 4], repeats=2, warmup=1, skip_param_shift_threshold=2)
    assert "metadata" in results
    assert "benchmarks" in results
    assert "N_2" in results["benchmarks"]
    assert "N_4" in results["benchmarks"]
    assert "transfer_matrix_build_ms" in results["benchmarks"]["N_4"]
    assert "forward_b1_ms" in results["benchmarks"]["N_4"]
    assert "backward_adjoint_ms" in results["benchmarks"]["N_4"]
