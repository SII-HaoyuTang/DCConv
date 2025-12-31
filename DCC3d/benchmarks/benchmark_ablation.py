import time
import torch
from DCC3d.src.cpu.dcconv3d import DistanceContainedConv3d
from DCC3d.src.cpu.selector import default_config


def benchmark_ablation():
    print("=" * 60)
    print("Benchmarking Ablation: PCA vs No PCA Latency")
    print("=" * 60)

    N = 2000  # Reasonable size
    in_channels = 16
    out_channels = 32
    conv_num = 16

    # Generate Data
    positions = torch.randn(N, 3)
    features = torch.randn(N, in_channels)

    # Models
    model_pca = DistanceContainedConv3d(
        in_channels,
        out_channels,
        default_config,
        3,
        2,
        2,
        conv_num,
        use_PCA=True,
        use_resnet=False,
    )
    model_no_pca = DistanceContainedConv3d(
        in_channels,
        out_channels,
        default_config,
        3,
        2,
        2,
        conv_num,
        use_PCA=False,
        use_resnet=False,
    )

    # Warmup
    print("Warming up...", end="\r")
    _ = model_pca(positions, features, N, N)
    _ = model_no_pca(positions, features, N, N)

    # Run PCA
    times_pca = []
    for _ in range(10):
        start = time.time()
        _ = model_pca(positions, features, N, N)
        times_pca.append(time.time() - start)
    avg_pca = sum(times_pca) / len(times_pca)

    # Run No PCA
    times_no_pca = []
    for _ in range(10):
        start = time.time()
        _ = model_no_pca(positions, features, N, N)
        times_no_pca.append(time.time() - start)
    avg_no_pca = sum(times_no_pca) / len(times_no_pca)

    print(f"{'Method':<15} | {'Latency (ms)':<15}")
    print("-" * 35)
    print(f"{'With PCA':<15} | {avg_pca * 1000:<15.2f}")
    print(f"{'No PCA':<15} | {avg_no_pca * 1000:<15.2f}")

    diff = avg_pca - avg_no_pca
    print(f"\nPCA Overhead: {diff * 1000:.2f} ms per call")


if __name__ == "__main__":
    benchmark_ablation()
