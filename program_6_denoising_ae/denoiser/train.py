#!/usr/bin/env python3
"""
Denoising Autoencoder (DAE) – Design, Implement, Evaluate

Features
- Datasets: MNIST (default), FashionMNIST, CIFAR10
- Noise types: Gaussian, Salt-and-Pepper, Uniform (adjustable intensity)
- Conv Autoencoder with BatchNorm + Dropout (regularization) + weight decay
- Metrics: MSE, PSNR, SSIM (per-image averaged)
- Proper visualizations: grids of (clean | noisy | recon) saved each epoch
- Training logs with timestamps saved as CSV; training curves plotted at end
- Best-checkpoint saving by highest validation PSNR

Run examples
------------
# MNIST + Gaussian noise
python denoising_autoencoder.py --dataset MNIST --noise gaussian --noise-level 0.2 --epochs 10

# CIFAR10 + Salt-and-Pepper
python denoising_autoencoder.py --dataset CIFAR10 --noise sp --noise-level 0.1 --epochs 20

# FashionMNIST + Uniform noise, heavier aug
python denoising_autoencoder.py --dataset FashionMNIST --noise uniform --noise-level 0.3 --epochs 15

Outputs
-------
- logs/log_YYYYmmdd_HHMMSS.csv : training/validation metrics per epoch with timestamps
- outputs/samples_epoch_{k}.png : clean/noisy/reconstructed grids
- outputs/best.ckpt : best model weights (by val PSNR)
- outputs/curves.png : loss & metric curves (MSE, PSNR, SSIM)

Notes
-----
- SSIM implementation here is a standard PyTorch version (Gaussian window) that
  works on 1- or 3-channel images in [0,1].
- For reproducibility, seeds are set; cudnn deterministic option provided.
"""

import argparse
import os
import math
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils as vutils
import matplotlib.pyplot as plt
import csv
import random
from dataset import get_dataset  # Assuming dataset.py is in the same directory

# -------------------------
# Utils: Reproducibility
# -------------------------
def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -------------------------
# Noise Functions
# -------------------------
class AddNoise:
    def __init__(self, noise_type: str = "gaussian", noise_level: float = 0.2):
        self.noise_type = noise_type.lower()
        self.noise_level = float(noise_level)
        assert 0 <= self.noise_level <= 1, "noise-level should be in [0,1]"
        assert self.noise_type in {"gaussian", "sp", "salt-and-pepper", "uniform"}

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x in [0,1], shape (C,H,W) or (N,C,H,W)"""
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        if self.noise_type == "gaussian":
            # std scales with noise_level
            std = self.noise_level
            noise = torch.randn_like(x) * std
            noisy = x + noise
        elif self.noise_type in {"sp", "salt-and-pepper"}:
            p = self.noise_level
            # Generate mask for salt (1) and pepper (0)
            rnd = torch.rand_like(x)
            noisy = x.clone()
            noisy[rnd < (p / 2)] = 0.0        # pepper
            noisy[(rnd >= (p / 2)) & (rnd < p)] = 1.0  # salt
        elif self.noise_type == "uniform":
            amp = self.noise_level
            noise = (torch.rand_like(x) * 2 - 1) * amp  # in [-amp, amp]
            noisy = x + noise
        else:
            raise ValueError("Unsupported noise type")

        noisy = torch.clamp(noisy, 0.0, 1.0)
        if squeeze_back:
            noisy = noisy.squeeze(0)
        return noisy


# -------------------------
# Model: Conv Autoencoder
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, use_bn=True, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.do = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.do(x)
        return x


class DenoiseAutoencoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, dropout=0.1):
        super().__init__()
        # Encoder
        self.e1 = ConvBlock(in_ch, base_ch, dropout=dropout)
        self.e2 = ConvBlock(base_ch, base_ch, s=2, p=1, dropout=dropout)   # down 1/2
        self.e3 = ConvBlock(base_ch, base_ch*2, dropout=dropout)
        self.e4 = ConvBlock(base_ch*2, base_ch*2, s=2, p=1, dropout=dropout)  # down 1/4
        self.e5 = ConvBlock(base_ch*2, base_ch*4, dropout=dropout)
        self.e6 = ConvBlock(base_ch*4, base_ch*4, s=2, p=1, dropout=dropout)  # down 1/8

        # Decoder
        self.d1 = ConvBlock(base_ch*4, base_ch*4)
        self.up1 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, 2)  # up x2
        self.d2 = ConvBlock(base_ch*2, base_ch*2)
        self.up2 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, 2)
        self.d3 = ConvBlock(base_ch, base_ch)
        self.up3 = nn.ConvTranspose2d(base_ch, base_ch, 2, 2)
        self.out = nn.Conv2d(base_ch, in_ch, kernel_size=1)
        self.tanh = nn.Sigmoid()  # outputs in [0,1]

    def forward(self, x):
        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)
        x = self.e4(x)
        x = self.e5(x)
        x = self.e6(x)

        x = self.d1(x)
        x = self.up1(x)
        x = self.d2(x)
        x = self.up2(x)
        x = self.d3(x)
        x = self.up3(x)
        x = self.out(x)
        x = self.tanh(x)
        return x


# -------------------------
# Metrics: MSE, PSNR, SSIM
# -------------------------

def mse_metric(x, y):
    return F.mse_loss(x, y, reduction='mean')


def psnr_metric(mse: torch.Tensor, max_val: float = 1.0):
    # PSNR = 10 * log10(MAX^2 / MSE)
    eps = 1e-10
    return 10.0 * torch.log10((max_val * max_val) / torch.clamp(mse, min=eps))


def _gaussian_window(window_size: int, sigma: float, channels: int, device):
    gauss = torch.tensor([
        math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ], device=device)
    gauss = (gauss / gauss.sum()).unsqueeze(1)
    window_2d = gauss @ gauss.t()
    window = window_2d.expand(channels, 1, window_size, window_size)
    return window


def ssim_metric(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5, max_val: float = 1.0):
    """
    Computes mean SSIM for a batch of images in [0,1].
    Shape: (N, C, H, W)
    """
    device = img1.device
    C = img1.size(1)
    window = _gaussian_window(window_size, sigma, C, device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=C) - mu1_mu2

    # constants
    L = max_val
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# -------------------------
# Training / Evaluation
# -------------------------

def save_grid(clean, noisy, recon, epoch, outdir, nrow=8, noise_type="gaussian"):
    os.makedirs(outdir, exist_ok=True)
    # Stack as rows: clean | noisy | recon
    grid = torch.cat([
        vutils.make_grid(clean, nrow=nrow, padding=2),
        vutils.make_grid(noisy, nrow=nrow, padding=2),
        vutils.make_grid(recon, nrow=nrow, padding=2)
    ], dim=1)
    path = os.path.join(outdir, f"samples_epoch_{epoch:03}_{noise_type}.png")
    vutils.save_image(grid, path)


def plot_curves(history, outdir):
    os.makedirs(outdir, exist_ok=True)
    epochs = [h['epoch'] for h in history]
    # Loss
    plt.figure()
    plt.plot(epochs, [h['train_mse'] for h in history], label='Train MSE')
    plt.plot(epochs, [h['val_mse'] for h in history], label='Val MSE')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.title('MSE over epochs')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f'curve_mse_{args.noise}.png')); plt.close()
    # PSNR
    plt.figure()
    plt.plot(epochs, [h['train_psnr'] for h in history], label='Train PSNR')
    plt.plot(epochs, [h['val_psnr'] for h in history], label='Val PSNR')
    plt.xlabel('Epoch'); plt.ylabel('PSNR (dB)'); plt.legend(); plt.title('PSNR over epochs')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f'curve_psnr_{args.noise}.png')); plt.close()
    # SSIM
    plt.figure()
    plt.plot(epochs, [h['train_ssim'] for h in history], label='Train SSIM')
    plt.plot(epochs, [h['val_ssim'] for h in history], label='Val SSIM')
    plt.xlabel('Epoch'); plt.ylabel('SSIM'); plt.legend(); plt.title('SSIM over epochs')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f'curve_ssim_{args.noise}.png')); plt.close()


def evaluate(model, loader, device, criterion, max_batches=None):
    model.eval()
    mse_vals, psnr_vals, ssim_vals = [], [], []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            noisy = noise_fn(imgs)
            recon = model(noisy)
            mse = criterion(recon, imgs)
            psnr = psnr_metric(mse)
            ssim = ssim_metric(recon, imgs)
            mse_vals.append(mse.item())
            psnr_vals.append(psnr.item())
            ssim_vals.append(ssim.item())
            if max_batches is not None and (i + 1) >= max_batches:
                break
    return np.mean(mse_vals), np.mean(psnr_vals), np.mean(ssim_vals)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoising Autoencoder Trainer")
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'FashionMNIST', 'CIFAR10'])
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument('--as-rgb', action='store_true', help='Expand grayscale MNIST/FashionMNIST to 3 channels')
    parser.add_argument('--noise', type=str, default='gaussian', choices=['gaussian', 'sp', 'salt-and-pepper', 'uniform'])
    parser.add_argument('--noise-level', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--max-eval-batches', type=int, default=None, help='For quick eval during debugging')
    args = parser.parse_args()

    set_seed(args.seed, args.deterministic)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    dataset, testset, in_ch = get_dataset(args.dataset, image_size=args.image_size, as_rgb=args.as_rgb)

    # Train/Val split
    val_len = int(len(dataset) * args.val_split)
    train_len = len(dataset) - val_len
    trainset, valset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Noise fn (global for evaluate(); could be improved via closure)
    global noise_fn
    noise_fn = AddNoise(args.noise, args.noise_level)

    # Model
    model = DenoiseAutoencoder(in_ch=in_ch, base_ch=64, dropout=args.dropout).to(device)

    # Loss & Optim
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Logging setup
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(args.logdir, f'log_{run_id}_{args.noise}.csv')
    ckpt_path = os.path.join(args.outdir, f'best_{args.noise}.ckpt')

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "epoch", "train_mse", "train_psnr", "train_ssim", "val_mse", "val_psnr", "val_ssim"])    

    print(f"Logging to: {log_path}")

    best_val_psnr = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            noisy = noise_fn(imgs)
            recon = model(noisy)
            loss = criterion(recon, imgs)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # stability
            optimizer.step()

            epoch_losses.append(loss.item())

        # Train metrics (on a few batches for speed)
        train_mse, train_psnr, train_ssim = evaluate(model, train_loader, device, criterion, max_batches=args.max_eval_batches)
        val_mse, val_psnr, val_ssim = evaluate(model, val_loader, device, criterion, max_batches=args.max_eval_batches)

        # Save sample grid
        with torch.no_grad():
            imgs, _ = next(iter(val_loader))
            imgs = imgs.to(device)[:64]
            noisy = noise_fn(imgs)
            recon = model(noisy)
            save_grid(imgs.cpu(), noisy.cpu(), recon.cpu(), epoch, args.outdir, nrow=8, noise_type=args.noise)

        # Log epoch
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = {
            'timestamp': ts,
            'epoch': epoch,
            'train_mse': float(train_mse),
            'train_psnr': float(train_psnr),
            'train_ssim': float(train_ssim),
            'val_mse': float(val_mse),
            'val_psnr': float(val_psnr),
            'val_ssim': float(val_ssim),
        }
        history.append(row)
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([row['timestamp'], row['epoch'], row['train_mse'], row['train_psnr'], row['train_ssim'], row['val_mse'], row['val_psnr'], row['val_ssim']])

        # Checkpointing by best val PSNR
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'args': vars(args)
            }, ckpt_path)

        print(f"Epoch {epoch:03d}/{args.epochs} | Train MSE {train_mse:.5f} PSNR {train_psnr:.2f} SSIM {train_ssim:.4f} || Val MSE {val_mse:.5f} PSNR {val_psnr:.2f} SSIM {val_ssim:.4f}")

    # Final plots
    plot_curves(history, args.outdir)

    # Final test evaluation (optional)
    test_mse, test_psnr, test_ssim = evaluate(model, test_loader, device, criterion)
    print(f"Test -> MSE: {test_mse:.5f}, PSNR: {test_psnr:.2f} dB, SSIM: {test_ssim:.4f}")

    # Save a final grid from test set
    with torch.no_grad():
        imgs, _ = next(iter(test_loader))
        imgs = imgs.to(device)[:64]
        noisy = noise_fn(imgs)
        recon = model(noisy)
        save_grid(imgs.cpu(), noisy.cpu(), recon.cpu(), epoch+1, args.outdir, nrow=8, noise_type=args.noise)

    print("Done. Artifacts saved to:")
    print(f"  Logs:     {log_path}")
    print(f"  Checkpoint (best): {ckpt_path}")
    print(f"  Samples & curves:  {args.outdir}")
