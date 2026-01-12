#!/usr/bin/env python3
"""
Minimal Training Script with GPU Profiling
===========================================

A lightweight training script designed for profiling GPU utilization.
Smaller dataset, fewer epochs, but comprehensive logging.

Usage:
    python -m DCC3d.benchmarks.train_with_profiling [--epochs 5] [--batch-size 64]

This will create:
    - training_profile.log: Detailed training logs with timing
    - training_metrics.json: Metrics per batch/epoch
    - gpu_usage.csv: GPU utilization timeline
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from DCC3d.src.cpu.module import DCConvNet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('training_profile.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GPULogger:
    """Background logger for GPU metrics"""
    
    def __init__(self, output_file: str = "gpu_usage.csv", interval: float = 0.2):
        self.output_file = output_file
        self.interval = interval
        self._running = False
        self._thread = None
        self._file = None
        self._writer = None
        
    def start(self):
        self._file = open(self.output_file, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow(['elapsed_sec', 'gpu_util_pct', 'mem_used_mb', 'mem_total_mb', 'temp_c', 'power_w'])
        self._start_time = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._log_loop, daemon=True)
        self._thread.start()
        logger.info(f"GPU logging started -> {self.output_file}")
        
    def _log_loop(self):
        while self._running:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=1.0
                )
                if result.returncode == 0:
                    parts = [p.strip() for p in result.stdout.strip().split(',')]
                    elapsed = time.time() - self._start_time
                    self._writer.writerow([f"{elapsed:.2f}"] + parts[:5])
                    self._file.flush()
            except Exception:
                pass
            time.sleep(self.interval)
            
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._file:
            self._file.close()
        logger.info("GPU logging stopped")


def create_synthetic_dataset(n_samples: int, n_points: int, n_features: int) -> TensorDataset:
    """Create a synthetic dataset for profiling"""
    logger.info(f"Creating synthetic dataset: {n_samples} samples, {n_points} points, {n_features} features")
    
    # Create batched data
    positions = torch.randn(n_samples, n_points, 3)
    features = torch.randn(n_samples, n_points, n_features)
    targets = torch.randn(n_samples, 1)  # Regression target
    
    return TensorDataset(positions, features, targets)


def collate_fn(batch):
    """Custom collate function to flatten batches for DCConvNet"""
    positions, features, targets = zip(*batch)
    
    batch_size = len(positions)
    n_points = positions[0].shape[0]
    
    # Stack and flatten
    pos = torch.cat(positions, dim=0)  # (batch_size * n_points, 3)
    feat = torch.cat(features, dim=0)  # (batch_size * n_points, n_features)
    target = torch.stack(targets, dim=0)  # (batch_size, 1)
    
    # Create belonging vector
    belonging = torch.arange(batch_size).repeat_interleave(n_points)
    
    return pos, feat, belonging, target


def train_epoch(model, dataloader, optimizer, criterion, device, epoch: int, 
                metrics_log: List[Dict]) -> Dict:
    """Train for one epoch with detailed logging"""
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    batch_times = []
    forward_times = []
    backward_times = []
    
    epoch_start = time.time()
    
    for batch_idx, (pos, feat, belonging, target) in enumerate(dataloader):
        batch_start = time.time()
        
        # Move to device
        pos = pos.to(device)
        feat = feat.to(device)
        belonging = belonging.to(device)
        target = target.to(device)
        
        # Forward pass
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_start = time.time()
        
        output = model(pos, feat, belonging)
        loss = criterion(output, target)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_end = time.time()
        forward_times.append((forward_end - forward_start) * 1000)
        
        # Backward pass
        backward_start = time.time()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        backward_end = time.time()
        backward_times.append((backward_end - backward_start) * 1000)
        
        batch_end = time.time()
        batch_time_ms = (batch_end - batch_start) * 1000
        batch_times.append(batch_time_ms)
        
        total_loss += loss.item() * target.size(0)
        total_samples += target.size(0)
        
        # Log every batch
        batch_metrics = {
            "epoch": epoch,
            "batch": batch_idx,
            "loss": loss.item(),
            "batch_time_ms": batch_time_ms,
            "forward_time_ms": forward_times[-1],
            "backward_time_ms": backward_times[-1],
            "samples": target.size(0),
        }
        metrics_log.append(batch_metrics)
        
        if batch_idx % 5 == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Forward: {forward_times[-1]:.1f}ms | "
                f"Backward: {backward_times[-1]:.1f}ms | "
                f"Total: {batch_time_ms:.1f}ms"
            )
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / total_samples
    
    epoch_summary = {
        "epoch": epoch,
        "avg_loss": avg_loss,
        "epoch_time_sec": epoch_time,
        "avg_batch_time_ms": sum(batch_times) / len(batch_times),
        "avg_forward_time_ms": sum(forward_times) / len(forward_times),
        "avg_backward_time_ms": sum(backward_times) / len(backward_times),
        "throughput_samples_per_sec": total_samples / epoch_time,
    }
    
    logger.info(
        f"Epoch {epoch} Summary | "
        f"Avg Loss: {avg_loss:.4f} | "
        f"Time: {epoch_time:.1f}s | "
        f"Throughput: {epoch_summary['throughput_samples_per_sec']:.1f} samples/s | "
        f"Avg Forward: {epoch_summary['avg_forward_time_ms']:.1f}ms | "
        f"Avg Backward: {epoch_summary['avg_backward_time_ms']:.1f}ms"
    )
    
    return epoch_summary


def main():
    parser = argparse.ArgumentParser(description="Training with GPU Profiling")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples in synthetic dataset")
    parser.add_argument("--n-points", type=int, default=64, help="Points per sample")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--no-gpu-log", action="store_true", help="Disable GPU logging")
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("DCConv Training with GPU Profiling")
    logger.info("=" * 80)
    logger.info(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, "
                f"n_samples={args.n_samples}, n_points={args.n_points}")
    
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Device: CUDA - {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Device: CPU")
    
    # Create model
    num_features = 4
    model = DCConvNet(num_features=num_features).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
    # Create synthetic dataset
    dataset = create_synthetic_dataset(args.n_samples, args.n_points, num_features)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,  # Simpler for profiling
        pin_memory=(device.type == "cuda")
    )
    
    logger.info(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches per epoch")
    
    # Setup training
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Start GPU logging
    gpu_logger = None
    if not args.no_gpu_log and device.type == "cuda":
        gpu_logger = GPULogger()
        gpu_logger.start()
    
    # Training loop
    metrics_log = []
    epoch_summaries = []
    
    training_start = time.time()
    
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_summary = train_epoch(
                model, dataloader, optimizer, criterion, device, epoch, metrics_log
            )
            epoch_summaries.append(epoch_summary)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    finally:
        if gpu_logger:
            gpu_logger.stop()
    
    training_time = time.time() - training_start
    
    # Save metrics
    results = {
        "config": vars(args),
        "device": str(device),
        "n_params": n_params,
        "training_time_sec": training_time,
        "epoch_summaries": epoch_summaries,
        "batch_metrics": metrics_log,
    }
    
    with open("training_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total time: {training_time:.1f}s")
    logger.info(f"Final loss: {epoch_summaries[-1]['avg_loss']:.4f}")
    logger.info(f"Average throughput: {sum(e['throughput_samples_per_sec'] for e in epoch_summaries)/len(epoch_summaries):.1f} samples/s")
    logger.info("")
    logger.info("Output files:")
    logger.info("  - training_profile.log: Detailed logs")
    logger.info("  - training_metrics.json: Metrics data")
    if gpu_logger:
        logger.info("  - gpu_usage.csv: GPU timeline")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Check gpu_usage.csv for GPU utilization patterns")
    logger.info("  2. Analyze training_metrics.json for timing breakdown")
    logger.info("  3. Run 'python -m DCC3d.benchmarks.gpu_profiler' for detailed module profiling")


if __name__ == "__main__":
    main()
