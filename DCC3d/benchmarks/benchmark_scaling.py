import csv
import os
import time

# Enable MPS fallback for missing operators (like linalg_eigh)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from DCC3d.src.cpu.module import DCConvNet


# set device to cpu
def benchmark_batch_scaling():
    print("=" * 80)
    print("Benchmarking Batch Size Scaling: DCConvNet (Robust Mode)")
    print("=" * 80)

    # Force GPU if available and print details
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU (Warning: This will be slow and serial-bound!)")

    # Important: Suppress dynamo errors or handle recompilations
    # torch._dynamo.config.suppress_errors = True

    # Model parameters
    num_features = 4
    n_points_per_sample = 128

    model = DCConvNet(num_features=num_features).to(device)
    model.eval()

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    # Prepare CSV
    csv_filename = "scaling_results.csv"
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Batch Size", "Total Points", "Avg Latency (ms)", "Throughput (samples/s)"]
        )

    print("\n[Phase 1] Warming up compilation for ALL batch sizes...")
    # This ensures torch.compile handles all shapes BEFORE we measure time
    for bs in tqdm(batch_sizes):
        total_points = bs * n_points_per_sample
        pos = torch.randn(total_points, 3).to(device)
        x = torch.randn(total_points, num_features).to(device)
        belonging = torch.arange(bs).repeat_interleave(n_points_per_sample).to(device)
        with torch.no_grad():
            _ = model(pos, x, belonging)

    print("\n[Phase 2] Benchmarking...")
    print("-" * 80)
    print(
        f"{'Batch Size':^12} | {'Total Points':^14} | {'Latency (ms)':^14} | {'Throughput (samples/s)':^24}"
    )
    print("-" * 80)

    for batch_size in batch_sizes:
        total_points = batch_size * n_points_per_sample
        pos = torch.randn(total_points, 3).to(device)
        x = torch.randn(total_points, num_features).to(device)
        belonging = (
            torch.arange(batch_size).repeat_interleave(n_points_per_sample).to(device)
        )

        n_repeats = 50  # Increased repeats
        start_event = time.time()

        with torch.no_grad():
            for _ in range(n_repeats):
                _ = model(pos, x, belonging)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "mps":
                    torch.mps.synchronize()

        end_event = time.time()

        total_time_ms = (end_event - start_event) * 1000
        avg_latency_ms = total_time_ms / n_repeats
        throughput = batch_size / (avg_latency_ms / 1000.0)

        print(
            f"{batch_size:^12} | {total_points:^14} | {avg_latency_ms:^14.2f} | {throughput:^24.2f}"
        )

        # Save to CSV
        with open(csv_filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([batch_size, total_points, avg_latency_ms, throughput])

    print("=" * 80)
    print("Interpretation Guide:")
    print("1. Low GPU Utilization? Check if the log says 'Using CPU'.")
    print(
        "2. Linear Scaling on GPU? Likely serial bottleneck in Python loop or kernel launch overhead."
    )
    print(
        "3. Recompilation Warnings? Ignoring them in timing is key (Fixed in this script via Warmup)."
    )


if __name__ == "__main__":
    benchmark_batch_scaling()
    print(f"\nResults saved to scaling_results.csv")
