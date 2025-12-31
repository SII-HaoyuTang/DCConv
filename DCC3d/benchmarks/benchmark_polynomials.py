import time
import torch
import numpy as np
import scipy.special
from DCC3d.src.cpu.kernel.polynomials_torch import (
    AssociatedLaguerrePoly,
    SphericalHarmonicFunc,
)


def benchmark_polynomials():
    print("=" * 85)
    print("Benchmarking Polynomials: Scipy vs Torch (Vectorized) - Variable Complexity")
    print("=" * 85)

    n_samples = 100000
    n_trials = 20

    # Random Inputs
    x_torch = torch.randn(n_samples)
    x_numpy = x_torch.numpy()

    theta_torch = torch.randn(n_samples)
    phi_torch = torch.randn(n_samples)
    theta_numpy = theta_torch.numpy()
    phi_numpy = phi_torch.numpy()

    # --- 1. Associated Laguerre Polynomials ---
    # Formula: L_n^k(x). Degree = n-k (in our implementation's notation).
    # Scipy: eval_genlaguerre(degree, alpha=k)
    print(
        f"\n[{'Associated Laguerre':^30}] | {'Degree':^12} | {'Torch (s)':^10} | {'Scipy (s)':^10} | {'Speedup':^8}"
    )
    print("-" * 85)

    # Cases: (n, k) -> Degree
    cases = [
        (3, 2),  # Degree 1
        (5, 2),  # Degree 3
        (10, 2),  # Degree 8
    ]

    for n, k in cases:
        degree = n - k
        # Torch
        laguerre_torch = AssociatedLaguerrePoly(n, k)
        # Warmup
        _ = laguerre_torch(x_torch)

        start = time.time()
        for _ in range(n_trials):
            _ = laguerre_torch(x_torch)
        time_torch = (time.time() - start) / n_trials

        # Scipy
        start = time.time()
        for _ in range(n_trials):
            _ = scipy.special.eval_genlaguerre(degree, k, x_numpy)
        time_scipy = (time.time() - start) / n_trials

        print(
            f"{f'n={n}, k={k}':^30} | {degree:^12} | {time_torch:^10.5f} | {time_scipy:^10.5f} | {time_scipy / time_torch:^8.2f}x"
        )

    # --- 2. Spherical Harmonics ---
    print(
        f"\n[{'Spherical Harmonics':^30}] | {'Degree (l,m)':^12} | {'Torch (s)':^10} | {'Scipy (s)':^10} | {'Speedup':^8}"
    )
    print("-" * 85)

    # Test cases: (l, m)
    sh_cases = [(3, 2), (5, 4), (10, 5)]

    for l, m in sh_cases:
        # Torch
        sph_torch = SphericalHarmonicFunc(l, m)
        # Warmup
        _ = sph_torch(theta_torch, phi_torch)

        start = time.time()
        for _ in range(n_trials):
            _ = sph_torch(theta_torch, phi_torch)
        time_torch = (time.time() - start) / n_trials

        # Scipy
        # Note: Scipy computes Complex SH. Torch computes Real SH.
        # This difference contributes to the speedup.
        start = time.time()
        for _ in range(n_trials):
            _ = scipy.special.sph_harm(m, l, phi_numpy, theta_numpy)
        time_scipy = (time.time() - start) / n_trials

        print(
            f"{f'l={l}, m={m}':^30} | {f'{l},{m}':^12} | {time_torch:^10.5f} | {time_scipy:^10.5f} | {time_scipy / time_torch:^8.2f}x"
        )


if __name__ == "__main__":
    benchmark_polynomials()
