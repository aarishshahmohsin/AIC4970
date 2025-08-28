import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Custom Dataset (Raw Images)
# -------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = [os.path.join(root, f) for f in os.listdir(root) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.files.sort()  # consistent ordering
        print(f"[Dataset] Found {len(self.files)} images in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if idx % 1000 == 0:
            print(f"[Dataset] Loaded image {idx}: {img_path} -> shape {tuple(img.shape)}")
        return img, 0   # dummy label


# -------------------------
# Variational Autoencoder
# -------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # -> 32x32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # -> 64x16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),# -> 128x8x8
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128*8*8, latent_dim)
        self.fc_logvar = nn.Linear(128*8*8, latent_dim)

        # Decoder
        self.fc = nn.Linear(latent_dim, 128*8*8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

        print("[Model] Initializing weights with Xavier/Kaiming")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def encode(self, x):
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc(z).view(-1, 128, 8, 8)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# -------------------------
# Loss Function
# -------------------------
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl_loss) / x.size(0), recon_loss / x.size(0), kl_loss / x.size(0)


# -------------------------
# Training Loop
# -------------------------
def train_vae(model, dataloader, epochs=10, lr=1e-3, device="cuda", save_dir="./checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    print(f"[Training] Starting on device: {device}")

    for epoch in range(epochs):
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0

        for batch_idx, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(imgs)
            loss, recon_loss, kl_loss = vae_loss(recon, imgs, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

            if batch_idx % 50 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] Batch {batch_idx}/{len(dataloader)} "
                      f"| Loss={loss.item():.4f} Recon={recon_loss.item():.4f} KL={kl_loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)

        print(f"[Epoch {epoch+1}] Avg Loss={avg_loss:.4f} "
              f"Avg Recon={avg_recon:.4f} Avg KL={avg_kl:.4f}")

        # -------------------------
        # Save checkpoint
        # -------------------------
        checkpoint_path = os.path.join(save_dir, f"vae_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, checkpoint_path)
        print(f"[Checkpoint] Saved to {checkpoint_path}")

        # -------------------------
        # Save reconstructions
        # -------------------------
        model.eval()
        with torch.no_grad():
            sample = imgs[:8]  # take a few samples from last batch
            recon, _, _ = model(sample)
            grid = torch.cat([sample.cpu(), recon.cpu()], dim=0)
            grid_img = torchvision.utils.make_grid(grid, nrow=8, normalize=True, value_range=(0,1))
            plt.figure(figsize=(12,4))
            plt.imshow(grid_img.permute(1,2,0).numpy())
            plt.axis("off")
            plt.savefig(os.path.join(save_dir, f"recon_epoch_{epoch+1}.png"))
            plt.close()
            print(f"[Reconstructions] Saved recon_epoch_{epoch+1}.png")


# -------------------------
# Latent Traversal
# -------------------------
def latent_traversal(model, latent_dim, base_z=None, steps=10, device="cuda"):
    model.eval()
    print(f"[Traversal] Exploring latent dimension {latent_dim} over {steps} steps")

    if base_z is None:
        base_z = torch.randn(1, model.latent_dim).to(device)
        print("[Traversal] Generated random base latent vector")

    z = base_z.repeat(steps, 1)
    vals = np.linspace(-3, 3, steps)
    images = []

    for i, val in enumerate(vals):
        z_mod = z.clone()
        z_mod[:, latent_dim] = val
        print(f"[Traversal] Step {i+1}/{steps}: latent[{latent_dim}] = {val:.2f}")
        with torch.no_grad():
            img = model.decode(z_mod[i].unsqueeze(0)).cpu().squeeze(0).permute(1,2,0).numpy()
        images.append(img)

    fig, axs = plt.subplots(1, steps, figsize=(15, 3))
    for i, ax in enumerate(axs):
        ax.imshow(images[i])
        ax.axis("off")
    plt.show()


# -------------------------
# Main Script
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[System] Using device: {device}")

    # Path to your image folder
    data_root = "/mnt/c/Users/HP/Downloads/img_align_celeba/celeba"
    print(f"[System] Loading dataset from {data_root}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = ImageFolderDataset(root=data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    model = VAE(latent_dim=32)
    print("[System] Model created with latent_dim=32")

    train_vae(model, dataloader, epochs=20, lr=1e-3, device=device)

    # Latent traversal example (e.g., dimension 12)
    latent_traversal(model, latent_dim=12, steps=8, device=device)