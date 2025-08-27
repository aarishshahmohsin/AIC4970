# file: latent_explorer_tk.py
import os
import argparse
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageTk

# -------------------------
# VAE (must match training)
# -------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128*8*8, latent_dim)
        self.fc_logvar = nn.Linear(128*8*8, latent_dim)
        self.fc = nn.Linear(latent_dim, 128*8*8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

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
        h = self.fc(z).view(-1, 128, 8, 8)
        return self.dec(h)

def load_checkpoint(ckpt_path, latent_dim, device):
    model = VAE(latent_dim=latent_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Load] Loaded '{ckpt_path}' (epoch={ckpt.get('epoch','?')}, loss={ckpt.get('loss','?')})")
    return model

# -------------------------
# Latent Explorer GUI
# -------------------------
class LatentExplorerApp:
    def __init__(self, root, model, device, latent_dim=32, slider_range=(-3.0, 3.0), slider_step=0.1, imsize=256, default_image=None):
        self.root = root
        self.model = model
        self.device = device
        self.latent_dim = latent_dim
        self.low, self.high = slider_range
        self.step = slider_step
        self.imsize = imsize
        self.default_image = default_image

        self.root.title("VAE Latent Explorer")
        self.root.geometry("950x750")

        # Left: image panel
        self.image_label = tk.Label(self.root)
        self.image_label.pack(side=tk.LEFT, padx=10, pady=10)

        # Right: controls
        controls = tk.Frame(self.root)
        controls.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Buttons
        btns = tk.Frame(controls)
        btns.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(btns, text="Randomize", command=self.randomize).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Reset (zeros)", command=self.reset).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Save Image", command=self.save_image).pack(side=tk.LEFT, padx=4)

        # Sliders container
        self.slider_vars = []
        slider_canvas = tk.Canvas(controls)
        slider_scroll = ttk.Scrollbar(controls, orient="vertical", command=slider_canvas.yview)
        self.slider_frame = tk.Frame(slider_canvas)

        self.slider_frame.bind(
            "<Configure>",
            lambda e: slider_canvas.configure(scrollregion=slider_canvas.bbox("all"))
        )
        slider_canvas.create_window((0, 0), window=self.slider_frame, anchor="nw")
        slider_canvas.configure(yscrollcommand=slider_scroll.set)
        slider_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        slider_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Create sliders
        for i in range(self.latent_dim):
            var = tk.DoubleVar()
            self.slider_vars.append(var)
            row = tk.Frame(self.slider_frame)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=f"z[{i:02d}]").pack(side=tk.LEFT, padx=4)
            s = tk.Scale(
                row, from_=self.low, to=self.high, resolution=self.step,
                orient=tk.HORIZONTAL, length=300, variable=var,
                command=lambda _=None: self.update_image()
            )
            s.pack(side=tk.LEFT, padx=4)

        # Initialize with default image if available
        if self.default_image and os.path.exists(self.default_image):
            self.encode_image(self.default_image)
        else:
            self.randomize()

    def preprocess_image(self, img_path):
        img = Image.open(img_path).convert("RGB").resize((64, 64))
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.tensor(arr).permute(2,0,1).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def encode_image(self, img_path):
        x = self.preprocess_image(img_path)
        mu, logvar = self.model.encode(x)
        z = mu  # use mean, deterministic
        for i, v in enumerate(self.slider_vars):
            v.set(float(z[0, i].cpu().item()))
        self.update_image()
        print(f"[Encode] Loaded {img_path} -> latent vector set")

    @torch.no_grad()
    def _decode_current(self):
        z = torch.tensor([v.get() for v in self.slider_vars], dtype=torch.float32, device=self.device).unsqueeze(0)
        x = self.model.decode(z).clamp(0,1).cpu()[0]  # 3x64x64
        arr = (x.permute(1,2,0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr).resize((self.imsize, self.imsize), Image.NEAREST)
        return img

    def update_image(self):
        img = self._decode_current()
        self._last_img = ImageTk.PhotoImage(img)  # keep ref to avoid GC
        self.image_label.config(image=self._last_img)

    def randomize(self):
        for v in self.slider_vars:
            v.set(np.random.normal(0, 1))
        self.update_image()

    def reset(self):
        for v in self.slider_vars:
            v.set(0.0)
        self.update_image()

    def load_image(self):
        fpath = filedialog.askopenfilename()
        if fpath:
            self.encode_image(fpath)

    def save_image(self):
        img = self._decode_current()
        fpath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg")]
        )
        if fpath:
            img.save(fpath)
            print(f"[Save] Wrote {fpath}")

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to saved .pth checkpoint")
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imsize", type=int, default=256, help="Display size for preview")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory to look for default image")
    args = parser.parse_args()

    model = load_checkpoint(args.ckpt, args.latent_dim, args.device)

    # find a default image in data_dir
    default_image = None
    for f in os.listdir(args.data_dir):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            default_image = os.path.join(args.data_dir, f)
            break

    root = tk.Tk()
    app = LatentExplorerApp(root, model=model, device=args.device,
                            latent_dim=args.latent_dim,
                            imsize=args.imsize,
                            default_image=default_image)
    root.mainloop()

if __name__ == "__main__":
    main()
