# =============================================================================
# src/models.py — all architectures used in the assignment (verbatim from the
# notebooks): MLP classifier + frozen latent decoder (Bagian 2 proof) and the
# six generative models (Vanilla AE, Sparse AE, VAE, GAN, DCGAN, DDPM).
# =============================================================================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

Z_DIM = 64          # latent dim shared by every latent model (control variable)
DDPM_T = 400        # diffusion timesteps

# --------------------------- Bagian 2: MLP proof ------------------------------
class MLPClassifier(nn.Module):
    """Tiered MLP classifier 784 -> 256 -> 64 -> 10 (Cross-Entropy)."""
    def __init__(self, in_dim=784, h1=256, h2=64, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, n_classes)

    def features(self, x):
        x = x.view(x.size(0), -1)
        return F.relu(self.fc2(F.relu(self.fc1(x))))   # z_MLP in R^64, pre-logit

    def forward(self, x):
        return self.fc3(self.features(x))


class FrozenLatentDecoder(nn.Module):
    """Standalone decoder 64 -> 256 -> 784 trained with MSE on frozen z_MLP."""
    def __init__(self, z_dim=64, hid=256, out_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hid), nn.ReLU(),
            nn.Linear(hid, out_dim), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


# --------------------------- latent autoencoder family ------------------------
class VanillaAE(nn.Module):
    """784 -> 256 -> 64 -> 256 -> 784, MSE. Encoder topology == MLP classifier encoder."""
    def __init__(self, z=Z_DIM):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, z), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(z, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

    def forward(self, x):
        z = self.enc(x.flatten(1))
        return self.dec(z).view(-1, 1, 28, 28), z


class SparseAE(nn.Module):
    """Same topology as VanillaAE + L1 penalty on the latent activations."""
    def __init__(self, z=Z_DIM):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, z), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(z, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

    def forward(self, x):
        z = self.enc(x.flatten(1))
        return self.dec(z).view(-1, 1, 28, 28), z


class VAE(nn.Module):
    """Gaussian VAE: mu/logvar heads, reparameterization, MSE + KL to N(0,I)."""
    def __init__(self, z=Z_DIM):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU())
        self.mu = nn.Linear(256, z)
        self.lv = nn.Linear(256, z)
        self.dec = nn.Sequential(nn.Linear(z, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x.flatten(1))
        return self.mu(h), self.lv(h)

    @staticmethod
    def reparam(mu, lv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)

    def forward(self, x):
        mu, lv = self.encode(x)
        zz = self.reparam(mu, lv)
        return self.dec(zz).view(-1, 1, 28, 28), mu, lv


# --------------------------- GAN family ---------------------------------------
class MLPG(nn.Module):
    """Vanilla GAN generator (MLP): z in R^64 -> 28x28 image in (0,1)."""
    def __init__(self, z=Z_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class MLPD(nn.Module):
    """Vanilla GAN discriminator (MLP)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x.flatten(1)).squeeze(1)


class DCGAN_G(nn.Module):
    """DCGAN generator: conv-transpose stack, BatchNorm + ReLU, Sigmoid output."""
    def __init__(self, z=Z_DIM, ngf=64):
        super().__init__()
        self.fc = nn.Linear(z, ngf * 2 * 7 * 7)
        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False), nn.Sigmoid())

    def forward(self, z):
        h = self.fc(z).view(-1, 128, 7, 7)
        return self.main(h)


class DCGAN_D(nn.Module):
    """DCGAN discriminator: strided convs, LeakyReLU + BatchNorm."""
    def __init__(self, ndf=32):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Flatten(), nn.Linear(ndf * 2 * 7 * 7, 1))

    def forward(self, x):
        return self.main(x).squeeze(1)


def dcgan_weights_init(m):
    """DCGAN convention: all weights N(0, 0.02)."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# --------------------------- DDPM ---------------------------------------------
# linear beta schedule (standard): betas 1e-4 -> 0.02 over T_DIFF steps
betas = torch.linspace(1e-4, 0.02, DDPM_T)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, 0)


class TimeMLP(nn.Module):
    """Sinusoidal timestep embedding -> small MLP."""
    def __init__(self, dim=64, out=128):
        super().__init__()
        self.dim, self.out = dim, out
        self.net = nn.Sequential(nn.Linear(dim, out), nn.SiLU(), nn.Linear(out, out))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        ang = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        return self.net(emb)


class ResBlock(nn.Module):
    def __init__(self, ch, t_dim=128):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.tproj = nn.Linear(t_dim, ch)

    def forward(self, x, temb):
        h = F.silu(self.c1(x))
        h = h + self.tproj(temb)[:, :, None, None]
        h = self.c2(h)
        return F.silu(x + h)


class TinyUNet(nn.Module):
    """Small UNet for eps-prediction on 28x28, 32/64 channels, skip connections."""
    def __init__(self):
        super().__init__()
        self.tmlp = TimeMLP()
        self.inc = nn.Conv2d(1, 32, 3, padding=1)
        self.rb1 = ResBlock(32)
        self.down1 = nn.Conv2d(32, 32, 4, 2, 1)          # 28 -> 14
        self.rb2 = ResBlock(32)
        self.down2 = nn.Conv2d(32, 64, 4, 2, 1)          # 14 -> 7
        self.rbm = ResBlock(64)
        self.up2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)   # 7 -> 14
        self.mix2 = nn.Conv2d(64, 32, 3, padding=1)
        self.rb3 = ResBlock(32)
        self.up1 = nn.ConvTranspose2d(32, 32, 4, 2, 1)   # 14 -> 28
        self.mix1 = nn.Conv2d(64, 32, 3, padding=1)
        self.rb4 = ResBlock(32)
        self.out = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x, t):
        temb = self.tmlp(t)
        h0 = F.silu(self.inc(x))
        h1 = self.rb1(h0, temb)
        h2 = self.rb2(self.down1(h1), temb)
        hm = self.rbm(self.down2(h2), temb)
        u2 = self.up2(hm)
        u2 = self.rb3(F.silu(self.mix2(torch.cat([u2, h2], 1))), temb)
        u1 = self.up1(u2)
        u1 = self.rb4(F.silu(self.mix1(torch.cat([u1, h1], 1))), temb)
        return self.out(u1)


def load_model(name):
    """Instantiate a trained model and load its checkpoint from
    artifacts_generative_compare/weights/{name}.pt."""
    import os
    ART = "artifacts_generative_compare"
    ck = os.path.join(ART, "weights", f"{name}.pt")
    classes = {
        "vanilla_ae": VanillaAE, "sparse_ae": SparseAE, "vae": VAE,
        "gan": MLPG, "dcgan": DCGAN_G, "ddpm": TinyUNet,
    }
    if name not in classes:
        raise ValueError(f"unknown model: {name}")
    m = classes[name]()
    m.load_state_dict(torch.load(ck, map_location="cpu"))
    return m.eval()


@torch.no_grad()
def ddpm_sample(model, n, device, snapshot_ts=None):
    """Full reverse diffusion x_T ~ N(0,I) -> x_0 in [0,1].
    If snapshot_ts (sorted desc) is given, also returns [(t, x_snapshot), ...]."""
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)
    snaps = []
    want = set(snapshot_ts) if snapshot_ts is not None else None
    for i in reversed(range(DDPM_T)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        eps = model(x, t)
        ab = alpha_bar[i].to(device); a = alphas[i].to(device); b = betas[i].to(device)
        mean = (1 / a.sqrt()) * (x - (b / (1 - ab).sqrt()) * eps)
        if i > 0:
            x = mean + b.sqrt() * torch.randn_like(x)
        else:
            x = mean
        if want is not None and i in want:
            snaps.append((i, x.clone().clamp(0, 1)))
    x = x.clamp(0, 1)
    return (x, snaps) if snapshot_ts is not None else x
