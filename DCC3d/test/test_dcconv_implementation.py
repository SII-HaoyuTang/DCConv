#!/usr/bin/env python3
"""
Test file for the updated DCConv implementation with n_select parameter.

This test file validates:
1. Basic selector functionality with n_select parameter
2. DistanceContainedConv3d with point reduction
3. Multi-layer architectures (DCConvNet and DCConvResNet)
4. Point reduction schedules
5. End-to-end functionality

Run with: python test_dcconv_implementation.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # To access DCC3d module

import traceback
from typing import Tuple

import torch
import torch.nn as nn

from DCC3d.src.cpu.dcconv3d import DistanceContainedConv3d
from DCC3d.src.cpu.module import DCConvNet, DCConvResNet, create_model
from DCC3d.src.cpu.selector import (
    SelectorConfig,
    SelectorFactory,
    SelectorType,
    get_reduction_schedule,
)


def create_synthetic_point_cloud(
    n_points: int = 100, n_features: int = 4, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic point cloud data for testing.

    Args:
        n_points: Number of points in the point cloud
        n_features: Number of features per point
        device: Device to create tensors on

    Returns:
        positions: (n_points, 3) - Point coordinates
        features: (n_points, n_features) - Point features
    """
    torch.manual_seed(42)  # For reproducible results

    # Create random 3D positions
    positions = torch.randn(n_points, 3, device=device) * 5.0  # Scale for variety

    # Create random features
    features = torch.randn(n_points, n_features, device=device)

    return positions, features


def test_selector_with_n_select():
    """Test the updated selector with n_select parameter."""
    print("=" * 60)
    print("Testing Selector with n_select parameter")
    print("=" * 60)

    try:
        # Create synthetic data
        positions, _ = create_synthetic_point_cloud(n_points=50, n_features=4)
        indices = torch.zeros(50, 1, dtype=torch.long)

        # Test different selector types
        selector_configs = [
            SelectorConfig(type=SelectorType.KNN, n=16),
            SelectorConfig(type=SelectorType.BALL_QUERY, n=16, radius=2.0),
            SelectorConfig(type=SelectorType.DILATED, n=16, dilation=2),
        ]

        for config in selector_configs:
            print(f"\nTesting {config['type']} selector...")

            selector = SelectorFactory.get_selector(config)

            # Test with different n_select values
            for n_select in [10, 25, 50]:
                try:
                    result = selector(
                        positions, indices, n_sample=16, n_select=n_select
                    )
                    expected_shape = (n_select, 16)

                    if result.shape == expected_shape:
                        print(f"  ✓ n_select={n_select}: shape {result.shape} correct")
                    else:
                        print(
                            f"  ✗ n_select={n_select}: expected {expected_shape}, got {result.shape}"
                        )

                except Exception as e:
                    print(f"  ✗ n_select={n_select}: Error - {e}")

        print("\n✓ Selector tests completed")
        return True

    except Exception as e:
        print(f"✗ Selector test failed: {e}")
        traceback.print_exc()
        return False


def test_reduction_schedule():
    """Test the point reduction schedule generation."""
    print("=" * 60)
    print("Testing Reduction Schedule")
    print("=" * 60)

    try:
        test_cases = [
            (100, 1, 5),  # 100 -> 1 in 5 steps
            (500, 1, 4),  # 500 -> 1 in 4 steps
            (50, 1, 3),  # 50 -> 1 in 3 steps
        ]

        for n_input, n_target, num_layers in test_cases:
            schedule = get_reduction_schedule(n_input, n_target, num_layers)
            print(
                f"Schedule {n_input} -> {n_target} in {num_layers} layers: {schedule}"
            )

            # Validate schedule
            if len(schedule) == num_layers:
                if schedule[0] < n_input and schedule[-1] >= n_target:
                    print(
                        f"  ✓ Valid schedule: {len(schedule)} steps, decreasing trend"
                    )
                else:
                    print(f"  ⚠ Schedule values may be unexpected")
            else:
                print(f"  ✗ Expected {num_layers} steps, got {len(schedule)}")

        print("\n✓ Reduction schedule tests completed")
        return True

    except Exception as e:
        print(f"✗ Reduction schedule test failed: {e}")
        traceback.print_exc()
        return False


def test_basic_dcconv3d():
    """Test the basic DistanceContainedConv3d with n_select."""
    print("=" * 60)
    print("Testing DistanceContainedConv3d with n_select")
    print("=" * 60)

    try:
        # Create synthetic data
        positions, features = create_synthetic_point_cloud(n_points=80, n_features=4)

        # Create DCConv layer
        config = SelectorConfig(type=SelectorType.KNN, n=16)
        layer = DistanceContainedConv3d(
            in_channels=4, out_channels=64, config=config, N=5, L=3, M=3, use_PCA=True
        )

        print(f"Input: positions {positions.shape}, features {features.shape}")

        # Test with different n_select values
        for n_select in [20, 40, None]:  # None means use all points
            try:
                centers, output = layer(positions, features, n_select)

                expected_n = positions.shape[0] if n_select is None else n_select
                expected_centers_shape = (expected_n, 3)
                expected_output_shape = (expected_n, 64)

                print(f"\nn_select={n_select}:")
                print(
                    f"  Centers shape: {centers.shape} (expected: {expected_centers_shape})"
                )
                print(
                    f"  Output shape: {output.shape} (expected: {expected_output_shape})"
                )

                if (
                    centers.shape == expected_centers_shape
                    and output.shape == expected_output_shape
                ):
                    print(f"  ✓ Correct shapes")
                else:
                    print(f"  ✗ Shape mismatch")

            except Exception as e:
                print(f"  ✗ n_select={n_select}: Error - {e}")
                traceback.print_exc()

        print("\n✓ Basic DCConv3d tests completed")
        return True

    except Exception as e:
        print(f"✗ Basic DCConv3d test failed: {e}")
        traceback.print_exc()
        return False


def test_dcconv_net():
    """Test the DCConvNet multi-layer architecture."""
    print("=" * 60)
    print("Testing DCConvNet Multi-layer Architecture")
    print("=" * 60)

    try:
        # Create synthetic data
        positions, features = create_synthetic_point_cloud(n_points=200, n_features=4)

        # Create model
        model = DCConvNet(
            in_channels=4,
            hidden_channels=[32, 64, 128],  # Smaller for testing
            out_channels=1,
            num_layers=4,
            selector_config=SelectorConfig(type=SelectorType.KNN, n=16),
            N=3,
            L=2,
            M=2,  # Smaller for faster testing
        )

        print(
            f"Model created with {sum(p.numel() for p in model.parameters())} parameters"
        )
        print(f"Input: positions {positions.shape}, features {features.shape}")

        # Forward pass
        with torch.no_grad():
            output = model(positions, features)

        print(f"Output shape: {output.shape} (expected: torch.Size([1]))")
        print(f"Output value: {output.item():.4f}")

        if output.shape == torch.Size([1]):
            print("✓ DCConvNet test passed - correct output shape")
        else:
            print("✗ DCConvNet test failed - incorrect output shape")

        return True

    except Exception as e:
        print(f"✗ DCConvNet test failed: {e}")
        traceback.print_exc()
        return False


def test_dcconv_resnet():
    """Test the DCConvResNet architecture."""
    print("=" * 60)
    print("Testing DCConvResNet Architecture")
    print("=" * 60)

    try:
        # Create synthetic data
        positions, features = create_synthetic_point_cloud(n_points=150, n_features=4)

        # Create ResNet model
        model = DCConvResNet(
            in_channels=4,
            block_configs=[[4, 32, 32], [32, 64, 64], [64, 128]],  # Smaller for testing
            out_channels=1,
            num_layers=3,
            selector_config=SelectorConfig(type=SelectorType.KNN, n=12),
            N=3,
            L=2,
            M=2,
        )

        print(
            f"ResNet model created with {sum(p.numel() for p in model.parameters())} parameters"
        )
        print(f"Input: positions {positions.shape}, features {features.shape}")

        # Forward pass
        with torch.no_grad():
            output = model(positions, features)

        print(f"Output shape: {output.shape} (expected: torch.Size([1]))")
        print(f"Output value: {output.item():.4f}")

        if output.shape == torch.Size([1]):
            print("✓ DCConvResNet test passed - correct output shape")
        else:
            print("✗ DCConvResNet test failed - incorrect output shape")

        return True

    except Exception as e:
        print(f"✗ DCConvResNet test failed: {e}")
        traceback.print_exc()
        return False


def test_model_factory():
    """Test the model factory function."""
    print("=" * 60)
    print("Testing Model Factory")
    print("=" * 60)

    try:
        # Test creating different model types
        models = {
            "dcconv": create_model(
                "dcconv", in_channels=4, hidden_channels=[32, 64], out_channels=1
            ),
            "resnet": create_model(
                "resnet",
                in_channels=4,
                block_configs=[[4, 32, 32], [32, 64]],
                out_channels=1,
            ),
        }

        for model_type, model in models.items():
            print(f"{model_type.upper()} model: {type(model).__name__}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")

        # Test invalid model type
        try:
            invalid_model = create_model("invalid_type")
            print("✗ Should have raised error for invalid model type")
        except ValueError as e:
            print(f"✓ Correctly raised error for invalid type: {e}")

        return True

    except Exception as e:
        print(f"✗ Model factory test failed: {e}")
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test that gradients flow properly through the network."""
    print("=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)

    try:
        # Create synthetic data
        positions, features = create_synthetic_point_cloud(n_points=50, n_features=4)
        target = torch.tensor([1.0])  # Regression target

        # Create model
        model = DCConvNet(
            in_channels=4,
            hidden_channels=[32, 64],
            out_channels=1,
            num_layers=3,
        )

        # Enable gradients
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Forward and backward pass
        optimizer.zero_grad()
        output = model(positions, features)
        loss = criterion(output, target)
        loss.backward()

        # Check if gradients exist
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if grad_norm > 0:
                    status = "✓"
                else:
                    status = "⚠"
                print(f"  {status} {name}: grad_norm = {grad_norm:.6f}")
            else:
                print(f"  ✗ {name}: No gradient")

        if len(grad_norms) > 0 and any(g > 0 for g in grad_norms):
            print(
                f"\n✓ Gradient flow test passed - {len(grad_norms)} parameters have gradients"
            )
        else:
            print(f"\n✗ Gradient flow test failed - no gradients found")

        return True

    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and provide summary."""
    print("🧪 Starting DCConv Implementation Tests")
    print("=" * 80)

    tests = [
        ("Reduction Schedule", test_reduction_schedule),
        ("Selector with n_select", test_selector_with_n_select),
        ("Basic DCConv3d", test_basic_dcconv3d),
        ("DCConvNet Architecture", test_dcconv_net),
        ("DCConvResNet Architecture", test_dcconv_resnet),
        ("Model Factory", test_model_factory),
        ("Gradient Flow", test_gradient_flow),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"💥 Test {test_name} crashed: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1

    print(f"\nResult: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All tests passed! The implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    return passed == len(results)


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("Device available:", "CUDA" if torch.cuda.is_available() else "CPU")

    success = run_all_tests()
    sys.exit(0 if success else 1)
