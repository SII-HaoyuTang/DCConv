"""
Test suite for DistanceContainedConv3d (DCC3d)
================================================

This module provides comprehensive tests for the DistanceContainedConv3d class,
including forward pass functionality, shape verification, gradient checks, and edge cases.
"""

import numpy as np
import torch
import torch.nn as nn

from DCC3d.src.cpu.dcconv3d import DistanceContainedConv3d
from DCC3d.src.cpu.selector import default_config


def generate_test_point_cloud(
    n_points: int = 100,
    n_channels: int = 16,
    coordinate_range: float = 10.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic point cloud data for testing.

    Args:
        n_points: Number of points in the point cloud
        n_channels: Number of input channels/features per point
        coordinate_range: Range of coordinates (points will be in [-range, range])
        seed: Random seed for reproducibility

    Returns:
        tuple: (position_matrix, channel_matrix)
            - position_matrix: (n_points, 3) coordinates
            - channel_matrix: (n_points, n_channels) features
    """
    torch.manual_seed(seed)

    # Generate random 3D coordinates
    position_matrix = torch.randn(n_points, 3) * coordinate_range

    # Generate random channel features
    channel_matrix = torch.randn(n_points, n_channels)

    return position_matrix, channel_matrix


def generate_structured_point_cloud(
    grid_size: int = 5, n_channels: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a structured grid-like point cloud for deterministic testing.

    Args:
        grid_size: Size of the cubic grid (total points = grid_size^3)
        n_channels: Number of input channels

    Returns:
        tuple: (position_matrix, channel_matrix)
    """
    # Create a regular 3D grid
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)

    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")

    # Flatten to get point coordinates
    positions = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
    n_points = positions.shape[0]

    # Generate simple features (e.g., distance from origin)
    distances = torch.norm(positions, dim=1, keepdim=True)
    features = distances.repeat(1, n_channels) + 0.1 * torch.randn(n_points, n_channels)

    return positions, features


def test_forward_pass_basic():
    """Test basic forward pass functionality."""
    print("\n--- Testing Basic Forward Pass ---")

    # Test parameters
    n_points = 50
    in_channels = 8
    out_channels = 16
    N, L, M = 2, 1, 1  # Small polynomial parameters for faster testing

    # Generate test data
    positions, features = generate_test_point_cloud(n_points, in_channels)

    # Create model
    model = DistanceContainedConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        config=default_config,
        N=N,
        L=L,
        M=M,
        conv_num=8,
        use_resnet=False,
        use_PCA=True,
    )

    # Forward pass - test with different n_select values
    try:
        # Test 1: No n_select (use all points)
        centers, output = model(
            positions, features, positions.shape[0], positions.shape[0]
        )
        expected_shape = (n_points, out_channels)
        expected_centers_shape = (n_points, 3)

        if output.shape == expected_shape and centers.shape == expected_centers_shape:
            print(
                f"✅ Forward pass (all points): output {output.shape}, centers {centers.shape}"
            )
        else:
            print(
                f"❌ Shape mismatch (all points): Got output {output.shape}, centers {centers.shape}"
            )
            print(
                f"    Expected output {expected_shape}, centers {expected_centers_shape}"
            )
            return False

        # Test 2: With n_select
        n_select = 30
        centers, output = model(positions, features, positions.shape[0], n_select)
        expected_shape = (n_select, out_channels)
        expected_centers_shape = (n_select, 3)

        if output.shape == expected_shape and centers.shape == expected_centers_shape:
            print(
                f"✅ Forward pass (n_select={n_select}): output {output.shape}, centers {centers.shape}"
            )
            return True
        else:
            print(
                f"❌ Shape mismatch (n_select={n_select}): Got output {output.shape}, centers {centers.shape}"
            )
            print(
                f"    Expected output {expected_shape}, centers {expected_centers_shape}"
            )
            return False

    except Exception as e:
        print(f"❌ Forward pass failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_different_input_sizes():
    """Test the model with different input sizes."""
    print("\n--- Testing Different Input Sizes ---")

    test_sizes = [
        (20, 4, 8),  # Small
        (40, 6, 12),  # Medium
        # (200, 8, 16),  # Large - commented out for faster testing
    ]

    N, L, M = 2, 1, 1

    for n_points, in_channels, out_channels in test_sizes:
        print(
            f"Testing size: {n_points} points, {in_channels} -> {out_channels} channels"
        )

        positions, features = generate_test_point_cloud(n_points, in_channels)

        model = DistanceContainedConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            config=default_config,
            N=N,
            L=L,
            M=M,
            conv_num=16,
            use_resnet=False,
            use_PCA=True,
        )

        try:
            # Test with all points
            centers, output = model(
                positions, features, positions.shape[0], positions.shape[0]
            )
            expected_shape = (n_points, out_channels)
            expected_centers_shape = (n_points, 3)

            if (
                output.shape == expected_shape
                and centers.shape == expected_centers_shape
            ):
                print(
                    f"  ✅ Size {n_points}x{in_channels}->{out_channels}: output {output.shape}, centers {centers.shape}"
                )
            else:
                print(
                    f"  ❌ Shape mismatch: Got output {output.shape}, centers {centers.shape}"
                )
                print(
                    f"      Expected output {expected_shape}, centers {expected_centers_shape}"
                )
                return False

            # Test with n_select (use half the points)
            n_select = max(1, n_points // 2)
            centers, output = model(positions, features, positions.shape[0], n_select)
            expected_shape = (n_select, out_channels)
            expected_centers_shape = (n_select, 3)

            if (
                output.shape == expected_shape
                and centers.shape == expected_centers_shape
            ):
                print(
                    f"  ✅ Size with n_select={n_select}: output {output.shape}, centers {centers.shape}"
                )
            else:
                print(
                    f"  ❌ n_select shape mismatch: Got output {output.shape}, centers {centers.shape}"
                )
                print(
                    f"      Expected output {expected_shape}, centers {expected_centers_shape}"
                )
                return False

        except Exception as e:
            print(f"  ❌ Failed with error: {e}")
            import traceback

            traceback.print_exc()
            return False

    print("✅ All size tests passed!")
    return True


def test_gradient_flow():
    """Test that gradients flow properly through the model."""
    print("\n--- Testing Gradient Flow ---")

    n_points = 30
    in_channels = 4
    out_channels = 8
    N, L, M = 2, 1, 1

    positions, features = generate_test_point_cloud(n_points, in_channels)
    positions.requires_grad_(True)
    features.requires_grad_(True)

    model = DistanceContainedConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        config=default_config,
        N=N,
        L=L,
        M=M,
        conv_num=8,
        use_resnet=False,
        use_PCA=True,
    )

    try:
        # Forward pass with n_select
        n_select = 20
        centers, output = model(positions, features, positions.shape[0], n_select)

        # Create a dummy loss (sum of all outputs)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check if gradients exist
        has_position_grad = positions.grad is not None and not torch.allclose(
            positions.grad, torch.zeros_like(positions.grad)
        )
        has_feature_grad = features.grad is not None and not torch.allclose(
            features.grad, torch.zeros_like(features.grad)
        )

        # Check model parameter gradients
        model_has_grads = any(
            p.grad is not None and not torch.allclose(p.grad, torch.zeros_like(p.grad))
            for p in model.parameters()
        )

        if has_position_grad and has_feature_grad and model_has_grads:
            print("✅ Gradients flow correctly through the model")
            print(f"  Position grad norm: {positions.grad.norm():.6f}")
            print(f"  Feature grad norm: {features.grad.norm():.6f}")
            print(f"  Output shape: {output.shape}, Centers shape: {centers.shape}")
            return True
        else:
            print("❌ Gradient flow issues:")
            print(f"  Position gradients: {'✅' if has_position_grad else '❌'}")
            print(f"  Feature gradients: {'✅' if has_feature_grad else '❌'}")
            print(f"  Model gradients: {'✅' if model_has_grads else '❌'}")
            return False

    except Exception as e:
        print(f"❌ Gradient test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_pca_vs_no_pca():
    """Test the model with and without PCA to ensure both modes work."""
    print("\n--- Testing PCA vs No PCA ---")

    n_points = 40
    in_channels = 6
    out_channels = 12
    N, L, M = 2, 1, 1
    n_select = 25

    positions, features = generate_test_point_cloud(n_points, in_channels, seed=123)

    # Test with PCA
    model_pca = DistanceContainedConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        config=default_config,
        N=N,
        L=L,
        M=M,
        conv_num=16,
        use_resnet=False,
        use_PCA=True,
    )

    # Test without PCA
    model_no_pca = DistanceContainedConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        config=default_config,
        N=N,
        L=L,
        M=M,
        conv_num=16,
        use_resnet=False,
        use_PCA=False,
    )

    try:
        centers_pca, output_pca = model_pca(
            positions, features, positions.shape[0], n_select
        )
        centers_no_pca, output_no_pca = model_no_pca(
            positions, features, positions.shape[0], n_select
        )

        expected_shape = (n_select, out_channels)
        expected_centers_shape = (n_select, 3)

        pca_shapes_ok = (
            output_pca.shape == expected_shape
            and centers_pca.shape == expected_centers_shape
        )
        no_pca_shapes_ok = (
            output_no_pca.shape == expected_shape
            and centers_no_pca.shape == expected_centers_shape
        )

        if pca_shapes_ok and no_pca_shapes_ok:
            print(
                f"✅ Both PCA modes work correctly: output {expected_shape}, centers {expected_centers_shape}"
            )
            print(f"  PCA output norm: {output_pca.norm():.6f}")
            print(f"  No-PCA output norm: {output_no_pca.norm():.6f}")
            print(f"  PCA centers norm: {centers_pca.norm():.6f}")
            print(f"  No-PCA centers norm: {centers_no_pca.norm():.6f}")
            return True
        else:
            print("❌ Shape issues:")
            print(f"  PCA: output {output_pca.shape}, centers {centers_pca.shape}")
            print(
                f"  No-PCA: output {output_no_pca.shape}, centers {centers_no_pca.shape}"
            )
            print(
                f"  Expected: output {expected_shape}, centers {expected_centers_shape}"
            )
            return False

    except Exception as e:
        print(f"❌ PCA comparison failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_structured_data():
    """Test with structured grid data for more deterministic results."""
    print("\n--- Testing Structured Grid Data ---")

    grid_size = 3  # 3x3x3 = 27 points
    in_channels = 4
    out_channels = 8
    N, L, M = 2, 1, 1
    n_select = 15

    positions, features = generate_structured_point_cloud(grid_size, in_channels)

    model = DistanceContainedConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        config=default_config,
        N=N,
        L=L,
        M=M,
        conv_num=8,
        use_resnet=False,
        use_PCA=True,
    )

    try:
        centers, output = model(positions, features, positions.shape[0], n_select)
        n_points = grid_size**3
        expected_shape = (n_select, out_channels)
        expected_centers_shape = (n_select, 3)

        if output.shape == expected_shape and centers.shape == expected_centers_shape:
            print(
                f"✅ Structured data test passed: output {output.shape}, centers {centers.shape}"
            )
            print(
                f"  Grid size: {grid_size}x{grid_size}x{grid_size} = {n_points} points -> {n_select} selected"
            )
            print(
                f"  Output statistics: mean={output.mean():.6f}, std={output.std():.6f}"
            )
            print(
                f"  Centers statistics: mean={centers.mean():.6f}, std={centers.std():.6f}"
            )
            return True
        else:
            print(f"❌ Shape mismatch:")
            print(f"    Got output {output.shape}, centers {centers.shape}")
            print(
                f"    Expected output {expected_shape}, centers {expected_centers_shape}"
            )
            return False

    except Exception as e:
        print(f"❌ Structured data test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_minimal_case():
    """Test with minimal input to check edge cases."""
    print("\n--- Testing Minimal Case ---")

    # Very small test case
    n_points = 20
    in_channels = 2
    out_channels = 4
    N, L, M = 2, 1, 1
    n_select = 10

    positions, features = generate_test_point_cloud(n_points, in_channels, seed=999)

    model = DistanceContainedConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        config=default_config,
        N=N,
        L=L,
        M=M,
        conv_num=5,
        use_resnet=False,
        use_PCA=False,  # Disable PCA for minimal case
    )

    try:
        centers, output = model(positions, features, positions.shape[0], n_select)
        expected_shape = (n_select, out_channels)
        expected_centers_shape = (n_select, 3)

        if output.shape == expected_shape and centers.shape == expected_centers_shape:
            print(
                f"✅ Minimal case passed: output {output.shape}, centers {centers.shape}"
            )
            return True
        else:
            print(f"❌ Shape mismatch:")
            print(f"    Got output {output.shape}, centers {centers.shape}")
            print(
                f"    Expected output {expected_shape}, centers {expected_centers_shape}"
            )
            return False

    except Exception as e:
        print(f"❌ Minimal case failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_n_select_variations():
    """Test different n_select values including edge cases."""
    print("\n--- Testing n_select Variations ---")

    n_points = 50
    in_channels = 4
    out_channels = 8
    N, L, M = 2, 1, 1

    positions, features = generate_test_point_cloud(n_points, in_channels, seed=42)

    model = DistanceContainedConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        config=default_config,
        N=N,
        L=L,
        M=M,
        conv_num=16,
        use_resnet=False,
        use_PCA=True,
    )

    # Test different n_select values
    test_cases = [
        (1, "minimum selection"),
        (10, "small selection"),
        (25, "half selection"),
        (40, "large selection"),
        (n_points, "all points"),
        (n_points + 10, "more than available"),  # Should clamp to available points
    ]

    for n_select, description in test_cases:
        try:
            print(f"  Testing {description} (n_select={n_select})...")
            centers, output = model(positions, features, positions.shape[0], n_select)

            # Expected n_select should be clamped to available points
            expected_n = min(n_select, n_points)
            expected_shape = (expected_n, out_channels)
            expected_centers_shape = (expected_n, 3)

            if (
                output.shape == expected_shape
                and centers.shape == expected_centers_shape
            ):
                print(
                    f"    ✅ {description}: output {output.shape}, centers {centers.shape}"
                )
            else:
                print(f"    ❌ Shape mismatch for {description}:")
                print(f"       Got output {output.shape}, centers {centers.shape}")
                print(
                    f"       Expected output {expected_shape}, centers {expected_centers_shape}"
                )
                return False

        except Exception as e:
            print(f"    ❌ {description} failed with error: {e}")
            return False

    print("✅ All n_select variation tests passed!")
    return True


def run_all_tests():
    """Run all test functions and summarize results."""
    print("=" * 60)
    print("Running DistanceContainedConv3d Test Suite")
    print("=" * 60)

    test_functions = [
        test_forward_pass_basic,
        test_different_input_sizes,
        test_gradient_flow,
        test_pca_vs_no_pca,
        test_structured_data,
        test_minimal_case,
        test_n_select_variations,
    ]

    results = []
    for test_func in test_functions:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"❌ {test_func.__name__} crashed with error: {e}")
            results.append((test_func.__name__, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    run_all_tests()
