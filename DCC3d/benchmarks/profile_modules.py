import time
import torch
import numpy as np
from DCC3d.src.cpu.dcconv3d import DistanceContainedConv3d
from DCC3d.src.cpu.selector import default_config


def profile_modules():
    print("=" * 60)
    print("Benchmarking: Execution Time per Module")
    print("=" * 60)

    # Parameters
    n_points = 5000
    in_channels = 16
    out_channels = 32
    conv_num = 16
    N, L, M = 3, 2, 2

    print(f"Config: N={n_points}, conv_num={conv_num}")

    # Data
    positions = torch.randn(n_points, 3)
    features = torch.randn(n_points, in_channels)
    space_points_num = n_points
    outpoint_num = n_points  # Output all points

    # Model
    model = DistanceContainedConv3d(
        in_channels,
        out_channels,
        default_config,
        N,
        L,
        M,
        conv_num,
        use_PCA=True,
        use_resnet=False,
    )

    # Warmup
    _ = model(positions, features, space_points_num, outpoint_num)

    # --- Profiling ---

    # 1. Neighbor Selection (Selector)
    start = time.time()

    # Mocking belonging vector (single batch)
    belonging = torch.zeros(n_points)

    # Direct call to selector (returns indices only)
    # Note: Using min(conv_num, model.conv_nums) logic from dcconv3d.py
    # Here we just pass conv_num assuming it fits.
    conv_num_val = conv_num

    neighbor_indices = model.selector(positions, belonging, conv_num_val, outpoint_num)
    t_selector = time.time() - start

    # 2. Coordinate Transformation
    start = time.time()
    # Use the actual forward method of cotrans
    # It returns: spherical_coords, centers, _, local_features
    # neighbor_indices has type long
    spherical_coords, centers, _, local_features_trans = model.cotrans.forward(
        global_coords=positions,
        neighbor_indices=neighbor_indices,
        global_features=features,
    )
    t_transform = time.time() - start

    # 3. Kernel (Polynomial Evaluation)
    start = time.time()
    # model.kernel.forward expects (outpoint_num, k, 3)
    # spherical_coords shape is correct from cotrans output
    kernel_weights = model.kernel.forward(spherical_coords)
    t_kernel = time.time() - start

    # 4. Aggregation (Conv)
    start = time.time()
    # Pre-permute local_features to (Ci, outpoint_num, k)
    local_features_perm = local_features_trans.permute(2, 0, 1)

    # Aggregation forward: (features, weights)
    # features: (Ci, outpoint_num, k)
    # weights: (Co, Ci, outpoint_num, k)
    output = model.aggregation.forward(local_features_perm, kernel_weights)
    output = output.permute(1, 0)  # (outpoint_num, Co)
    t_conv = time.time() - start

    total = t_selector + t_transform + t_kernel + t_conv

    print(f"\n{'-' * 60}")
    print(f"{'Module':<25} | {'Time (ms)':<10} | {'% of Total':<10}")
    print(f"{'-' * 60}")
    print(
        f"{'Neighbor Selection':<25} | {t_selector * 1000:<10.2f} | {t_selector / total * 100:<10.1f}%"
    )
    print(
        f"{'Coord. Transform (PCA)':<25} | {t_transform * 1000:<10.2f} | {t_transform / total * 100:<10.1f}%"
    )
    print(
        f"{'Kernel (Polynomials)':<25} | {t_kernel * 1000:<10.2f} | {t_kernel / total * 100:<10.1f}%"
    )
    print(
        f"{'Convolution (Agg)':<25} | {t_conv * 1000:<10.2f} | {t_conv / total * 100:<10.1f}%"
    )
    print(f"{'-' * 60}")
    print(f"{'Total':<25} | {total * 1000:<10.2f} | 100.0%")


if __name__ == "__main__":
    profile_modules()
