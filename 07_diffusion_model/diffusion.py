"""Minimal DDPM: forward noising, a small UNet, and the training loop."""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class SmallUNet(nn.Module):
    """Tiny UNet with sinusoidal time embedding (for 28x28 grayscale)."""
    def __init__(self, ch=64, tdim=128):
        super().__init__()
        self.tdim = tdim
        self.time_mlp = nn.Sequential(nn.Linear(tdim, ch), nn.SiLU(), nn.Linear(ch, ch))
        self.d1 = nn.Conv2d(1, ch, 3, padding=1)
        self.d2 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        self.mid = nn.Conv2d(ch, ch, 3, padding=1)
        self.u2 = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
        self.out = nn.Conv2d(ch, 1, 3, padding=1)

    def temb(self, t):
        half = self.tdim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device) *
                          (torch.log(torch.tensor(10000.0)) / (half - 1)))
        a = t[:, None].float() * freqs[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)

    def forward(self, x, t):
        temb = self.time_mlp(self.temb(t))[:, :, None, None]
        h1 = F.silu(self.d1(x)) + temb
        h2 = F.silu(self.d2(h1))
        m = F.silu(self.mid(h2))
        u = F.silu(self.u2(m))
        return self.out(u + h1)


class Diffusion:
    def __init__(self, T=300, device="cpu"):
        self.T = T
        self.beta = torch.linspace(1e-4, 0.02, T, device=device)
        self.alpha = 1 - self.beta
        self.abar = torch.cumprod(self.alpha, dim=0)
        self.device = device

    def q_sample(self, x0, t, noise):
        ab = self.abar[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def loss(self, model, x0):
        t = torch.randint(0, self.T, (x0.size(0),), device=self.device)
        noise = torch.randn_like(x0)
        pred = model(self.q_sample(x0, t, noise), t)
        return F.mse_loss(pred, noise)


def train(model, diff, loader, epochs=5, lr=2e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        for x, _ in loader:
            x = x.to(diff.device)
            loss = diff.loss(model, x)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {ep} loss {loss.item():.4f}")


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.5], [0.5])])
    ds = datasets.MNIST(".", download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=128, shuffle=True)
    model, diff = SmallUNet().to(dev), Diffusion(device=dev)
    train(model, diff, dl)
