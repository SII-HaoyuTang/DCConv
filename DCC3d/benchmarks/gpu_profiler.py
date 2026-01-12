#!/usr/bin/env python3
"""
GPU Profiling Script for DCConv Training Analysis
==================================================

This script provides comprehensive GPU monitoring during training to identify
performance bottlenecks. Run this on your CUDA cluster.

Usage:
    python -m DCC3d.benchmarks.gpu_profiler [--profile-mode full|quick|torch]

Outputs:
    - gpu_profile_results.json: Detailed timing breakdown
    - gpu_timeline.csv: GPU utilization over time
    - Console summary with bottleneck analysis
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from DCC3d.src.cpu.module import DCConvNet
from DCC3d.src.cpu.dcconv3d import DistanceContainedConv3d
from DCC3d.src.cpu.selector import default_config


@dataclass
class GPUStats:
    """Container for GPU statistics at a point in time"""
    timestamp: float
    gpu_util: float  # GPU utilization %
    memory_used: float  # Memory used in MB
    memory_total: float  # Total memory in MB
    temperature: float  # GPU temperature
    power_draw: float  # Power consumption in W


@dataclass
class ProfileResult:
    """Container for profiling results"""
    module_name: str
    forward_time_ms: float
    backward_time_ms: float = 0.0
    cuda_time_ms: float = 0.0
    cpu_time_ms: float = 0.0
    memory_allocated_mb: float = 0.0
    memory_peak_mb: float = 0.0
    kernel_launches: int = 0


class GPUMonitor:
    """Background thread to monitor GPU utilization using nvidia-smi"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.stats: List[GPUStats] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def _query_nvidia_smi(self) -> Optional[GPUStats]:
        """Query nvidia-smi for current GPU stats"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=1.0
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) >= 5:
                    return GPUStats(
                        timestamp=time.time(),
                        gpu_util=float(parts[0].strip()),
                        memory_used=float(parts[1].strip()),
                        memory_total=float(parts[2].strip()),
                        temperature=float(parts[3].strip()),
                        power_draw=float(parts[4].strip().replace(' W', ''))
                    )
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        return None
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._running:
            stats = self._query_nvidia_smi()
            if stats:
                self.stats.append(stats)
            time.sleep(self.interval)
    
    def start(self):
        """Start background monitoring"""
        self.stats = []
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> List[GPUStats]:
        """Stop monitoring and return collected stats"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return self.stats
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.stats:
            return {"error": "No stats collected"}
        
        gpu_utils = [s.gpu_util for s in self.stats]
        return {
            "samples": len(self.stats),
            "duration_seconds": self.stats[-1].timestamp - self.stats[0].timestamp if len(self.stats) > 1 else 0,
            "gpu_util_mean": sum(gpu_utils) / len(gpu_utils),
            "gpu_util_max": max(gpu_utils),
            "gpu_util_min": min(gpu_utils),
            "memory_peak_mb": max(s.memory_used for s in self.stats),
        }


class Timer:
    """Context manager for timing code blocks"""
    
    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda
        self.elapsed_ms = 0.0
    
    def __enter__(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000


def profile_forward_pass(model: nn.Module, pos: torch.Tensor, x: torch.Tensor, 
                         belonging: torch.Tensor, n_warmup: int = 3, n_runs: int = 10) -> Dict:
    """Profile the forward pass of the model"""
    device = pos.device
    
    # Warmup
    print("  Warming up...", end=" ", flush=True)
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(pos, x, belonging)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("done")
    
    # Profile forward pass
    print(f"  Running {n_runs} forward passes...", end=" ", flush=True)
    
    forward_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    for _ in range(n_runs):
        with Timer(sync_cuda=(device.type == "cuda")) as t:
            with torch.no_grad():
                _ = model(pos, x, belonging)
        forward_times.append(t.elapsed_ms)
    
    print("done")
    
    memory_stats = {}
    if device.type == "cuda":
        memory_stats = {
            "memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "memory_peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }
    
    return {
        "forward_time_mean_ms": sum(forward_times) / len(forward_times),
        "forward_time_std_ms": (sum((t - sum(forward_times)/len(forward_times))**2 for t in forward_times) / len(forward_times)) ** 0.5,
        "forward_time_min_ms": min(forward_times),
        "forward_time_max_ms": max(forward_times),
        **memory_stats
    }


def profile_backward_pass(model: nn.Module, pos: torch.Tensor, x: torch.Tensor,
                          belonging: torch.Tensor, n_warmup: int = 2, n_runs: int = 5) -> Dict:
    """Profile the backward pass of the model"""
    device = pos.device
    
    # Warmup
    print("  Warming up backward...", end=" ", flush=True)
    for _ in range(n_warmup):
        pos_grad = pos.clone().detach().requires_grad_(True)
        x_grad = x.clone().detach().requires_grad_(True)
        output = model(pos_grad, x_grad, belonging)
        loss = output.sum()
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("done")
    
    # Profile backward pass
    print(f"  Running {n_runs} backward passes...", end=" ", flush=True)
    
    backward_times = []
    for _ in range(n_runs):
        pos_grad = pos.clone().detach().requires_grad_(True)
        x_grad = x.clone().detach().requires_grad_(True)
        
        # Forward
        output = model(pos_grad, x_grad, belonging)
        loss = output.sum()
        
        # Time backward only
        with Timer(sync_cuda=(device.type == "cuda")) as t:
            loss.backward()
        backward_times.append(t.elapsed_ms)
    
    print("done")
    
    return {
        "backward_time_mean_ms": sum(backward_times) / len(backward_times),
        "backward_time_std_ms": (sum((t - sum(backward_times)/len(backward_times))**2 for t in backward_times) / len(backward_times)) ** 0.5,
    }


def profile_torch_profiler(model: nn.Module, pos: torch.Tensor, x: torch.Tensor,
                           belonging: torch.Tensor) -> Dict:
    """Use PyTorch's built-in profiler for detailed kernel analysis"""
    device = pos.device
    
    if device.type != "cuda":
        return {"error": "PyTorch profiler requires CUDA"}
    
    print("  Running PyTorch profiler...")
    
    # Warmup
    for _ in range(2):
        with torch.no_grad():
            _ = model(pos, x, belonging)
    torch.cuda.synchronize()
    
    # Profile with PyTorch profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            _ = model(pos, x, belonging)
    
    # Get key averages
    key_averages = prof.key_averages()
    
    # Find top CUDA operations
    cuda_ops = []
    for item in sorted(key_averages, key=lambda x: x.cuda_time_total, reverse=True)[:15]:
        cuda_ops.append({
            "name": item.key,
            "cuda_time_ms": item.cuda_time_total / 1000,
            "cpu_time_ms": item.cpu_time_total / 1000,
            "calls": item.count,
        })
    
    # Print table
    print("\n  Top 15 CUDA Operations:")
    print("  " + "-" * 80)
    print(f"  {'Operation':<40} | {'CUDA (ms)':>10} | {'CPU (ms)':>10} | {'Calls':>6}")
    print("  " + "-" * 80)
    for op in cuda_ops:
        print(f"  {op['name'][:40]:<40} | {op['cuda_time_ms']:>10.3f} | {op['cpu_time_ms']:>10.3f} | {op['calls']:>6}")
    print("  " + "-" * 80)
    
    # Calculate totals
    total_cuda_ms = sum(op['cuda_time_ms'] for op in cuda_ops)
    total_cpu_ms = sum(op['cpu_time_ms'] for op in cuda_ops)
    
    return {
        "top_cuda_ops": cuda_ops,
        "total_cuda_time_ms": total_cuda_ms,
        "total_cpu_time_ms": total_cpu_ms,
        "kernel_count": sum(op['calls'] for op in cuda_ops),
    }


def profile_module_breakdown(model: DCConvNet, pos: torch.Tensor, x: torch.Tensor,
                             belonging: torch.Tensor) -> List[Dict]:
    """Profile each major module separately"""
    device = pos.device
    results = []
    
    # Get the first DCConv layer for detailed profiling
    first_conv = model.init_conv
    
    # 1. PreSelector
    print("  Profiling PreSelector...", end=" ", flush=True)
    with Timer(sync_cuda=(device.type == "cuda")) as t:
        for _ in range(10):
            pos_out, x_out, schedule = model.pre_selector(pos, x, belonging)
    results.append({"module": "PreSelector", "time_ms": t.elapsed_ms / 10})
    print(f"{t.elapsed_ms/10:.2f}ms")
    
    # 2. Profile selector from first conv
    print("  Profiling KNN Selector...", end=" ", flush=True)
    space_num = int(pos_out.shape[0] / schedule[0])
    linspace = torch.linspace(1, space_num, space_num, device=device)
    belong_vec = linspace.repeat_interleave(schedule[0])
    conv_num = min(schedule[0] - schedule[1] + 1, first_conv.conv_nums)
    
    with Timer(sync_cuda=(device.type == "cuda")) as t:
        for _ in range(10):
            neighbor_idx = first_conv.selector(pos_out, belong_vec, conv_num, schedule[1])
    results.append({"module": "KNN Selector", "time_ms": t.elapsed_ms / 10})
    print(f"{t.elapsed_ms/10:.2f}ms")
    
    # 3. Profile coordinate transformer
    print("  Profiling CoordinateTransformer...", end=" ", flush=True)
    with Timer(sync_cuda=(device.type == "cuda")) as t:
        for _ in range(10):
            spherical, centers, _, local_feat = first_conv.cotrans.forward(
                global_coords=pos_out,
                neighbor_indices=neighbor_idx,
                global_features=x_out
            )
    results.append({"module": "CoordinateTransformer", "time_ms": t.elapsed_ms / 10})
    print(f"{t.elapsed_ms/10:.2f}ms")
    
    # 4. Profile kernel polynomial evaluation
    print("  Profiling Kernel (Polynomials)...", end=" ", flush=True)
    with Timer(sync_cuda=(device.type == "cuda")) as t:
        for _ in range(10):
            kernel_weights = first_conv.kernel.forward(spherical)
    results.append({"module": "Kernel (Polynomials)", "time_ms": t.elapsed_ms / 10})
    print(f"{t.elapsed_ms/10:.2f}ms")
    
    # 5. Profile aggregation
    print("  Profiling Aggregation...", end=" ", flush=True)
    local_feat_perm = local_feat.permute(2, 0, 1)
    with Timer(sync_cuda=(device.type == "cuda")) as t:
        for _ in range(10):
            output = first_conv.aggregation.forward(local_feat_perm, kernel_weights)
    results.append({"module": "Aggregation", "time_ms": t.elapsed_ms / 10})
    print(f"{t.elapsed_ms/10:.2f}ms")
    
    return results


def run_profiling(mode: str = "full", batch_size: int = 32, n_points: int = 128):
    """Run the full profiling suite"""
    print("=" * 80)
    print("DCConv GPU Profiling Tool")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Mode: {mode}")
    print(f"Batch size: {batch_size}, Points per sample: {n_points}")
    print()
    
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("⚠ Using MPS (Apple Silicon) - CUDA profiling not available")
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU - GPU profiling not available")
    print()
    
    # Create model and data
    num_features = 4
    total_points = batch_size * n_points
    
    model = DCConvNet(num_features=num_features).to(device)
    model.eval()
    
    pos = torch.randn(total_points, 3, device=device)
    x = torch.randn(total_points, num_features, device=device)
    belonging = torch.arange(batch_size, device=device).repeat_interleave(n_points)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: pos={pos.shape}, x={x.shape}")
    print()
    
    results = {
        "config": {
            "batch_size": batch_size,
            "n_points": n_points,
            "total_points": total_points,
            "device": str(device),
            "timestamp": datetime.now().isoformat(),
        }
    }
    
    # Start GPU monitoring
    gpu_monitor = GPUMonitor(interval=0.05)
    if device.type == "cuda":
        gpu_monitor.start()
    
    # 1. Forward pass profiling
    print("[1/4] Profiling Forward Pass")
    print("-" * 40)
    results["forward"] = profile_forward_pass(model, pos, x, belonging)
    print(f"  Mean forward time: {results['forward']['forward_time_mean_ms']:.2f}ms")
    print()
    
    # 2. Backward pass profiling
    if mode in ["full"]:
        print("[2/4] Profiling Backward Pass")
        print("-" * 40)
        results["backward"] = profile_backward_pass(model, pos, x, belonging)
        print(f"  Mean backward time: {results['backward']['backward_time_mean_ms']:.2f}ms")
        print()
    
    # 3. Module breakdown
    print("[3/4] Module-Level Breakdown")
    print("-" * 40)
    results["modules"] = profile_module_breakdown(model, pos, x, belonging)
    print()
    
    # 4. PyTorch profiler (CUDA only)
    if mode in ["full", "torch"] and device.type == "cuda":
        print("[4/4] PyTorch Profiler Analysis")
        print("-" * 40)
        results["torch_profiler"] = profile_torch_profiler(model, pos, x, belonging)
        print()
    
    # Stop GPU monitoring
    if device.type == "cuda":
        gpu_stats = gpu_monitor.stop()
        results["gpu_stats"] = gpu_monitor.get_summary()
        
        # Save timeline
        if gpu_stats:
            with open("gpu_timeline.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "gpu_util", "memory_used_mb", "temperature", "power_draw"])
                t0 = gpu_stats[0].timestamp
                for s in gpu_stats:
                    writer.writerow([s.timestamp - t0, s.gpu_util, s.memory_used, s.temperature, s.power_draw])
            print(f"GPU timeline saved to: gpu_timeline.csv")
    
    # Save results
    with open("gpu_profile_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: gpu_profile_results.json")
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if "gpu_stats" in results and "gpu_util_mean" in results["gpu_stats"]:
        gs = results["gpu_stats"]
        print(f"GPU Utilization: {gs['gpu_util_mean']:.1f}% (min: {gs['gpu_util_min']:.1f}%, max: {gs['gpu_util_max']:.1f}%)")
        print(f"Memory Peak: {gs['memory_peak_mb']:.0f} MB")
    elif "gpu_stats" in results:
        print("GPU stats: No samples collected (profiling too fast)")
    
    print(f"\nForward Pass: {results['forward']['forward_time_mean_ms']:.2f}ms ± {results['forward']['forward_time_std_ms']:.2f}ms")
    
    if "backward" in results:
        print(f"Backward Pass: {results['backward']['backward_time_mean_ms']:.2f}ms")
    
    print("\nModule Breakdown:")
    total_module_time = sum(m["time_ms"] for m in results["modules"])
    for m in results["modules"]:
        pct = m["time_ms"] / total_module_time * 100
        bar = "█" * int(pct / 5)
        print(f"  {m['module']:<25}: {m['time_ms']:>8.2f}ms ({pct:>5.1f}%) {bar}")
    
    # Bottleneck analysis
    print("\n" + "=" * 80)
    print("BOTTLENECK ANALYSIS")
    print("=" * 80)
    
    sorted_modules = sorted(results["modules"], key=lambda x: x["time_ms"], reverse=True)
    top_bottleneck = sorted_modules[0]
    print(f"⚠ Primary bottleneck: {top_bottleneck['module']} ({top_bottleneck['time_ms']:.2f}ms, {top_bottleneck['time_ms']/total_module_time*100:.1f}%)")
    
    if "gpu_stats" in results and "gpu_util_mean" in results["gpu_stats"] and results["gpu_stats"]["gpu_util_mean"] < 50:
        print(f"⚠ Low GPU utilization ({results['gpu_stats']['gpu_util_mean']:.1f}%) indicates CPU-bound or serialization issues")
    
    print()
    return results


def main():
    parser = argparse.ArgumentParser(description="GPU Profiling for DCConv")
    parser.add_argument("--mode", choices=["full", "quick", "torch"], default="full",
                        help="Profiling mode: full (all tests), quick (forward only), torch (PyTorch profiler)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for profiling")
    parser.add_argument("--n-points", type=int, default=128, help="Points per sample")
    args = parser.parse_args()
    
    run_profiling(mode=args.mode, batch_size=args.batch_size, n_points=args.n_points)


if __name__ == "__main__":
    main()
