import time
import csv
import torch
import numpy as np

from DCC3d.src.cpu.selector.selector_torch import KNNSelector as KNNSelectorTorch
from DCC3d.src.cpu.selector.selector_numpy import KNNSelector as KNNSelectorNumpy


def benchmark_selector():
    print("=" * 60)
    print("Benchmarking Selector: Torch (Vectorized) vs Numpy")
    print("=" * 60)

    k = 16
    n_points_list = [1000, 2000, 5000, 10000, 20000, 50000]
    results = []

    # Prepare CSV
    csv_filename = "selector_results.csv"
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "Torch Time (s)", "Numpy Time (s)", "Speedup"])

    print(f"{'N':<10} | {'Torch (s)':<12} | {'Numpy (s)':<12} | {'Speedup':<10}")
    print("-" * 52)

    for N in n_points_list:
        print(f"Running N={N}...", end="\r")

        # Generate Data
        coords_torch = torch.randn(N, 3)
        belonging = torch.zeros(N)  # Single batch

        coords_numpy = coords_torch.numpy()

        # Torch Benchmark
        selector_torch = KNNSelectorTorch()
        # Warmup
        _ = selector_torch.select(coords_torch, belonging, k, N)

        start_time = time.time()
        for _ in range(20):
            _ = selector_torch.select(coords_torch, belonging, k, N)
        torch_time = (time.time() - start_time) / 20

        # Numpy Benchmark
        selector_numpy = KNNSelectorNumpy(n_sample=k)
        # Warmup
        _ = selector_numpy.select(coords_numpy)

        start_time = time.time()
        for _ in range(5):
            _ = selector_numpy.select(coords_numpy)
        numpy_time = (time.time() - start_time) / 20

        speedup = numpy_time / torch_time
        results.append(
            {
                "N": N,
                "Torch Time (s)": torch_time,
                "Numpy Time (s)": numpy_time,
                "Speedup": speedup,
            }
        )
        print(
            f"{N:<10} | {torch_time:<12.5f} | {numpy_time:<12.5f} | {speedup:<10.2f}x"
        )

        # Save to CSV
        with open(csv_filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([N, torch_time, numpy_time, speedup])

    print(f"\nResults saved to {csv_filename}")

    return results


if __name__ == "__main__":
    benchmark_selector()
