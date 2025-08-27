#!/usr/bin/env python3
"""
beta_vae.py

Run a beta-VAE (unsupervised) on an image folder (images only).
- AdamW optimizer
- Orthogonal weight init (small gain for final decoder conv)
- Optional KL annealing
- Save checkpoints and reconstruction grids
- Latent traversal utility after training

Example:
    python beta_vae.py --data_root /mnt/c/Users/HP/Downloads/img_align_celeba/celeba --epochs 30 --batch_size 128 --beta 4.0
"""

import os
import argparse
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils as vutils
import torchvision
import matplotlib.pyplot as plt

# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------
# Custom Dataset (Raw Images)
# -------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # collect only files (no subdirs)
        files = [f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.files = [os.path.join(root, f) for f in files]
        self.files.sort()
        print(f"[Dataset] Found {len(self.files)} images in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if idx % 1000 == 0:
            # img is a tensor (C,H,W) after transform
            try:
                shape = tuple(img.shape)
            except Exception:
                shape = str(img)
            print(f"[Dataset] Loaded image {idx}: {img_path} -> shape {shape}")
        return img, 0  # dummy label for DataLoader compatibility


# -------------------------
# Variational Autoencoder (β-VAE)
# -------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: 64x64 input -> downsample to 8x8 with channels 128
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 32x32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 64x16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # 128x8x8
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

        # Decoder
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # -> 64x16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # -> 32x32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),   # -> 3x64x64
            nn.Sigmoid()  # assume inputs in [0,1]
        )

        print("[Model] Initializing weights (orthogonal + small gain for final conv)")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Orthogonal for convs and linears (preserves gradients)
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            try:
                nn.init.orthogonal_(m.weight)
            except Exception:
                # fallback (some layers might not have .weight like Flatten)
                pass
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

        # Make final decoder conv with small initialization to avoid saturating Sigmoid
        if isinstance(m, nn.ConvTranspose2d) and getattr(m, "out_channels", None) == 3:
            try:
                # tiny gain to center outputs near 0.5 initially
                nn.init.xavier_uniform_(m.weight, gain=0.01)
            except Exception:
                pass

    def encode(self, x):
        h = self.enc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc(z)
        h = h.view(-1, 128, 8, 8)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar


# -------------------------
# Loss (β-VAE)
# -------------------------
def vae_loss(recon_x, x, mu, logvar, beta=1.0, reduction="sum"):
    """
    recon_x, x: tensors with shape (B,C,H,W)
    mu, logvar: (B, latent_dim)
    beta: scalar multiplier for KL (beta-VAE)
    reduction: "sum" (default) to compute sum across all pixels then divide by batch
    """
    # Reconstruction: MSE (sum)
    recon_loss = F.mse_loss(recon_x, x, reduction=reduction)
    # KL: sum over latent dims
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Return per-batch-average losses so logging is consistent.
    # We'll divide by batch size in the caller (or here).
    return recon_loss, kl_loss, (recon_loss + beta * kl_loss)


# -------------------------
# Utilities: save image grid
# -------------------------
def save_reconstructions(save_dir, epoch, inputs, reconstructions, n_display=8):
    os.makedirs(save_dir, exist_ok=True)
    inputs = inputs[:n_display].cpu()
    recons = reconstructions[:n_display].cpu()
    # Interleave originals and recons
    grid = torch.cat([inputs, recons], dim=0)
    grid_img = vutils.make_grid(grid, nrow=n_display, normalize=True, value_range=(0,1))
    out_path = os.path.join(save_dir, f"recon_epoch_{epoch:04d}.png")
    ndarr = grid_img.permute(1, 2, 0).numpy()
    plt.imsave(out_path, ndarr)
    print(f"[Reconstructions] Saved: {out_path}")


# -------------------------
# Training Loop
# -------------------------
def train_vae(model, dataloader, epochs, beta, lr, device, save_dir, kl_anneal=False, anneal_epochs=10):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    print(f"[Training] device={device} | beta={beta} | kl_anneal={kl_anneal} (anneal_epochs={anneal_epochs})")

    for epoch in range(1, epochs + 1):
        model.train()
        total_recon = 0.0
        total_kl = 0.0
        total_loss = 0.0
        total_samples = 0

        # KL annealing schedule (if enabled): ramp beta from 0 -> beta over anneal_epochs
        if kl_anneal:
            current_beta = beta * min(1.0, epoch / max(1, anneal_epochs))
        else:
            current_beta = beta

        for batch_idx, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            batch_size = imgs.size(0)
            optimizer.zero_grad()
            recon, mu, logvar = model(imgs)

            recon_sum, kl_sum, combined = vae_loss(recon, imgs, mu, logvar, beta=current_beta, reduction="sum")
            # normalize by batch_size to match earlier behavior
            loss = combined / batch_size
            loss.backward()
            optimizer.step()

            total_recon += (recon_sum.item() / batch_size)
            total_kl += (kl_sum.item() / batch_size)
            total_loss += loss.item()
            total_samples += 1

            if batch_idx % 50 == 0:
                print(f"[Epoch {epoch}/{epochs}] Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss={loss.item():.4f} Recon={recon_sum.item()/batch_size:.4f} KL={kl_sum.item()/batch_size:.4f} beta={current_beta:.3f}")

        avg_loss = total_loss / max(1, total_samples)
        avg_recon = total_recon / max(1, total_samples)
        avg_kl = total_kl / max(1, total_samples)

        print(f"[Epoch {epoch}] AVG Loss={avg_loss:.4f} AVG Recon={avg_recon:.4f} AVG KL={avg_kl:.4f} (beta={current_beta:.3f})")

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_loss": avg_loss,
            "beta": beta
        }
        ckpt_path = os.path.join(save_dir, f"beta_vae_epoch_{epoch:04d}.pth")
        torch.save(ckpt, ckpt_path)
        print(f"[Checkpoint] Saved: {ckpt_path}")

        # Save reconstructions using last batch
        model.eval()
        with torch.no_grad():
            save_reconstructions(save_dir, epoch, imgs, recon, n_display=min(8, imgs.size(0)))

    print("[Training] Finished.")


# -------------------------
# Latent Traversal (visualize effect of varying a single latent dim)
# -------------------------
def latent_traversal(model, latent_dim_idx, device, steps=8, base_z=None, span=3.0, save_path=None):
    model.to(device)
    model.eval()
    zdim = model.latent_dim
    if base_z is None:
        base_z = torch.zeros(1, zdim).to(device)
    else:
        base_z = base_z.to(device)

    vals = np.linspace(-span, span, steps)
    imgs = []
    with torch.no_grad():
        for v in vals:
            z = base_z.clone()
            z[0, latent_dim_idx] = float(v)
            out = model.decode(z)
            img = out.cpu().squeeze(0).permute(1, 2, 0).numpy()
            imgs.append(img)

    # plot and optionally save
    fig, axs = plt.subplots(1, steps, figsize=(2 * steps, 2))
    for i in range(steps):
        axs[i].imshow(np.clip(imgs[i], 0, 1))
        axs[i].axis("off")
        axs[i].set_title(f"{vals[i]:.2f}", fontsize=8)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        print(f"[Traversal] Saved traversal image to {save_path}")
    plt.show()


# -------------------------
# Argument parser & main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="Path to folder with images")
    p.add_argument("--save_dir", type=str, default="./checkpoints", help="Where to save checkpoints & recons")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=4.0, help="beta multiplier for KL in beta-VAE")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kl_anneal", action="store_true", help="Enable linear KL annealing (0->beta over anneal_epochs)")
    p.add_argument("--anneal_epochs", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[System] Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),  # maps to [0,1] for PIL images
    ])

    dataset = ImageFolderDataset(args.data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"))

    model = VAE(latent_dim=args.latent_dim)
    print(f"[System] Model instantiated with latent_dim={args.latent_dim}")

    train_vae(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        beta=args.beta,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir,
        kl_anneal=args.kl_anneal,
        anneal_epochs=args.anneal_epochs
    )

    # After training -> example latent traversal on dimension 0
    # load last checkpoint (optional)
    latest_ckpt = os.path.join(args.save_dir, f"beta_vae_epoch_{args.epochs:04d}.pth")
    if os.path.exists(latest_ckpt):
        ck = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        print(f"[System] Loaded checkpoint: {latest_ckpt}")

    # Create traversal images for first few dims
    os.makedirs(os.path.join(args.save_dir, "traversals"), exist_ok=True)
    for dim in range(min(4, args.latent_dim)):
        save_path = os.path.join(args.save_dir, "traversals", f"traversal_dim_{dim:02d}.png")
        latent_traversal(model, latent_dim_idx=dim, device=device, steps=8, span=3.0, save_path=save_path)

    print("[System] Done.")


if __name__ == "__main__":
    main()
