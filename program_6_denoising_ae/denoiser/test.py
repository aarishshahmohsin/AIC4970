# --- 1. Imports ---
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --- 2. CIFAR-10 Loader ---
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR10Dataset(Dataset):
    def __init__(self, root="./cifar-10-batches-py", train=True):
        if train:
            files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            files = ["test_batch"]

        data_list, labels_list = [], []
        for file in files:
            batch = unpickle(os.path.join(root, file))
            data_list.append(batch[b'data'])
            labels_list.extend(batch[b'labels'])

        self.data = np.concatenate(data_list).reshape(-1, 3, 32, 32).astype("float32") / 255.0
        self.labels = np.array(labels_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.tensor(self.data[idx])  # (3,32,32)
        label = torch.tensor(self.labels[idx])
        return img, label

# --- 3. Add noise ---
def add_gaussian_noise(img, sigma=0.2):
    noisy = img + sigma * torch.randn_like(img)
    return torch.clamp(noisy, 0., 1.)

# --- 4. Simple Autoencoder (must match training) ---
import torch
import torch.nn as nn

from train import DenoiseAutoencoder


# --- 5. Load dataset & model ---
testset = CIFAR10Dataset(root="/mnt/c/Users/HP/Downloads/cifar-10-python/cifar-10-batches-py", train=False)
test_loader = DataLoader(testset, batch_size=8, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DenoiseAutoencoder().to(device)

# Load checkpoint (replace with your path)
model.load_state_dict(torch.load("outputs/best_gaussian.ckpt", map_location=device)['model_state'])
model.eval()

# --- 6. Visualize ---
imgs, _ = next(iter(test_loader))
noisy_imgs = add_gaussian_noise(imgs, sigma=0.2)

with torch.no_grad():
    recon = model(noisy_imgs.to(device))

# Convert to numpy
imgs = imgs.permute(0,2,3,1).numpy()
noisy_imgs = noisy_imgs.permute(0,2,3,1).numpy()
recon = recon.cpu().permute(0,2,3,1).numpy()

# Plot
print(imgs.shape[0])
n = min(8, imgs.shape[0])
fig, axes = plt.subplots(3, n, figsize=(15, 6))

for i in range(n):
    axes[0,i].imshow(imgs[i]); axes[0,i].set_title("Clean"); axes[0,i].axis("off")
    axes[1,i].imshow(noisy_imgs[i]); axes[1,i].set_title("Noisy"); axes[1,i].axis("off")
    axes[2,i].imshow(recon[i]); axes[2,i].set_title("Reconstructed"); axes[2,i].axis("off")

plt.tight_layout()
plt.show()
