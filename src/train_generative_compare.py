# =============================================================================
# src/train_generative_compare.py — trains the six generative models on the
# SAME standardized MNIST split (seed 42) and logs Test MSE/PSNR/SSIM (latent
# family) plus sample diversity stats (GAN family).
#
# Models: Vanilla AE | Sparse AE | VAE | GAN (MLP) | DCGAN | DDPM
# Checkpoint-or-train: if artifacts_generative_compare/weights/{name}.pt exists
# it is loaded and metrics are recomputed fresh; delete it to retrain.
#
# Usage:  python -m src.train_generative_compare [--train]
# By default retrains nothing if all checkpoints exist; pass --train to skip
# the checkpoint cache and retrain from scratch.
# =============================================================================
import argparse, json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.data import SEED, BATCH, set_seed, build_dataloaders
from src.models import (Z_DIM, DDPM_T, VanillaAE, SparseAE, VAE,
                        MLPG, MLPD, DCGAN_G, DCGAN_D, dcgan_weights_init,
                        TinyUNet, ddpm_sample)
from src.metrics import eval_recon, gen_diversity_stats

ART = "artifacts_generative_compare"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = {"vanilla_ae": 15, "sparse_ae": 15, "vae": 15,
          "gan": 30, "dcgan": 10, "ddpm": 8}
LR = 1e-3
LR_GAN = 2e-4
LAMBDA_L1 = 1e-3

for sub in ("weights", "grids", "metrics"):
    os.makedirs(os.path.join(ART, sub), exist_ok=True)


def save_json(obj, name):
    with open(os.path.join(ART, "metrics", name), "w") as f:
        json.dump(obj, f, indent=2)


def load_json(name):
    with open(os.path.join(ART, "metrics", name)) as f:
        return json.load(f)


def save_grid(imgs, name, nrow=8):
    from torchvision.utils import make_grid, save_image
    save_image(make_grid(imgs.clamp(0, 1), nrow=nrow, padding=2),
               os.path.join(ART, "grids", name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true",
                    help="retrain from scratch even if a checkpoint exists")
    args = ap.parse_args()

    set_seed()
    train_loader, val_loader, test_loader = build_dataloaders()
    print(f"device={DEVICE} train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}", flush=True)

    test_pool = torch.cat([xb.flatten(1) for xb, _ in test_loader])[:2000]
    X10 = torch.stack([test_loader.dataset[i][0] for i in range(10)]).to(DEVICE)

    # ------------------------------ Vanilla AE --------------------------------
    set_seed(); t0 = time.perf_counter()
    ae = VanillaAE().to(DEVICE)
    ck = os.path.join(ART, "weights", "vanilla_ae.pt")
    if os.path.exists(ck) and not args.train:
        ae.load_state_dict(torch.load(ck, map_location=DEVICE))
        hist = load_json("vanilla_ae.json")["history"]
        print("[vanilla_ae] checkpoint loaded", flush=True)
    else:
        opt = torch.optim.Adam(ae.parameters(), lr=LR)
        hist = []
        for ep in range(1, EPOCHS["vanilla_ae"] + 1):
            run, n = 0.0, 0
            for xb, _ in train_loader:
                xb = xb.to(DEVICE)
                opt.zero_grad(); xh, _ = ae(xb); loss = F.mse_loss(xh, xb)
                loss.backward(); opt.step()
                run += loss.item() * len(xb); n += len(xb)
            hist.append({"epoch": ep, "train_mse": run / n})
            print(f"  vanilla_ae ep{ep} train_mse={run/n:.5f}", flush=True)
        torch.save(ae.state_dict(), ck)
    ae.eval()
    print(f"[vanilla_ae] {time.perf_counter()-t0:.1f}s", flush=True)
    m = eval_recon(lambda x: ae(x)[0], test_loader, DEVICE)
    with torch.no_grad():
        xh10, _ = ae(X10)
    save_grid(torch.cat([X10.cpu(), xh10.clamp(0, 1).cpu()]), "vanilla_ae_recon_grid.png", nrow=10)
    save_json({"history": hist, **m}, "vanilla_ae.json")

    # ------------------------------ Sparse AE ---------------------------------
    set_seed(); t0 = time.perf_counter()
    sae = SparseAE().to(DEVICE)
    ck = os.path.join(ART, "weights", "sparse_ae.pt")
    if os.path.exists(ck) and not args.train:
        sae.load_state_dict(torch.load(ck, map_location=DEVICE))
        hist = load_json("sparse_ae.json")["history"]
        print("[sparse_ae] checkpoint loaded", flush=True)
    else:
        opt = torch.optim.Adam(sae.parameters(), lr=LR)
        hist = []
        for ep in range(1, EPOCHS["sparse_ae"] + 1):
            run, sp, n = 0.0, 0.0, 0
            for xb, _ in train_loader:
                xb = xb.to(DEVICE)
                opt.zero_grad(); xh, z = sae(xb)
                loss = F.mse_loss(xh, xb) + LAMBDA_L1 * z.abs().mean()
                loss.backward(); opt.step()
                run += F.mse_loss(xh, xb).item() * len(xb)
                sp += (z > 1e-6).float().mean().item(); n += 1
            hist.append({"epoch": ep, "train_mse": run / len(train_loader.dataset),
                         "latent_active_frac": sp / n})
            print(f"  sparse_ae ep{ep} train_mse={run/len(train_loader.dataset):.5f}", flush=True)
        torch.save(sae.state_dict(), ck)
    sae.eval()
    print(f"[sparse_ae] {time.perf_counter()-t0:.1f}s", flush=True)
    m = eval_recon(lambda x: sae(x)[0], test_loader, DEVICE)
    with torch.no_grad():
        xh10, z10 = sae(X10)
    m["latent_active_frac_test"] = (z10 > 1e-6).float().mean().item()
    save_grid(torch.cat([X10.cpu(), xh10.clamp(0, 1).cpu()]), "sparse_ae_recon_grid.png", nrow=10)
    save_json({"history": hist, **m, "lambda_l1": LAMBDA_L1}, "sparse_ae.json")

    # ------------------------------ VAE ----------------------------------------
    set_seed(); t0 = time.perf_counter()
    vae = VAE().to(DEVICE)
    ck = os.path.join(ART, "weights", "vae.pt")
    if os.path.exists(ck) and not args.train:
        vae.load_state_dict(torch.load(ck, map_location=DEVICE))
        hist = load_json("vae.json")["history"]
        print("[vae] checkpoint loaded", flush=True)
    else:
        opt = torch.optim.Adam(vae.parameters(), lr=LR)
        hist = []
        for ep in range(1, EPOCHS["vae"] + 1):
            r_sum = k_sum = 0.0; n = 0
            for xb, _ in train_loader:
                xb = xb.to(DEVICE)
                opt.zero_grad(); xh, mu, lv = vae(xb)
                recon = F.mse_loss(xh, xb, reduction="sum") / len(xb)
                kl = (-0.5 * torch.sum(1 + lv - mu ** 2 - lv.exp())) / len(xb)
                loss = recon + kl
                loss.backward(); opt.step()
                r_sum += recon.item(); k_sum += kl.item(); n += 1
            hist.append({"epoch": ep, "recon_mse": r_sum / n, "kl": k_sum / n})
            print(f"  vae ep{ep} recon={r_sum/n:.4f} kl={k_sum/n:.2f}", flush=True)
        torch.save(vae.state_dict(), ck)
    vae.eval()
    print(f"[vae] {time.perf_counter()-t0:.1f}s", flush=True)
    m = eval_recon(lambda x: vae(x)[0], test_loader, DEVICE)
    with torch.no_grad():
        xh10, _, _ = vae(X10)
        zs = torch.randn(25, Z_DIM, device=DEVICE)
        prior = vae.dec(zs).view(-1, 1, 28, 28).clamp(0, 1).cpu()
    save_grid(torch.cat([X10.cpu(), xh10.clamp(0, 1).cpu()]), "vae_recon_grid.png", nrow=10)
    save_grid(prior, "vae_prior_samples_25.png", nrow=5)
    save_json({"history": hist, **m}, "vae.json")

    bce = nn.BCEWithLogitsLoss()

    # ------------------------------ GAN (MLP) ----------------------------------
    set_seed(); t0 = time.perf_counter()
    G, D = MLPG().to(DEVICE), MLPD().to(DEVICE)
    ck = os.path.join(ART, "weights", "gan.pt")
    if os.path.exists(ck) and not args.train:
        G.load_state_dict(torch.load(ck, map_location=DEVICE))
        hist = load_json("gan.json")["history"]
        print("[gan] checkpoint loaded", flush=True)
    else:
        optG = torch.optim.Adam(G.parameters(), lr=LR_GAN, betas=(0.5, 0.999))
        optD = torch.optim.Adam(D.parameters(), lr=LR_GAN, betas=(0.5, 0.999))
        hist = []
        for ep in range(1, EPOCHS["gan"] + 1):
            for xb, _ in train_loader:
                xb = xb.to(DEVICE); bs = len(xb)
                real = torch.full((bs,), 0.9, device=DEVICE)
                fake = torch.zeros(bs, device=DEVICE)
                optD.zero_grad()
                d_loss = bce(D(xb), real) + bce(D(G(torch.randn(bs, Z_DIM, device=DEVICE)).detach()), fake)
                d_loss.backward(); optD.step()
                optG.zero_grad()
                g_loss = bce(D(G(torch.randn(bs, Z_DIM, device=DEVICE))), real)
                g_loss.backward(); optG.step()
            hist.append({"epoch": ep, "d_loss": d_loss.item(), "g_loss": g_loss.item()})
            print(f"  gan ep{ep} D={d_loss.item():.3f} G={g_loss.item():.3f}", flush=True)
        torch.save(G.state_dict(), ck)
    G.eval()
    print(f"[gan] {time.perf_counter()-t0:.1f}s", flush=True)
    with torch.no_grad():
        samples = G(torch.randn(64, Z_DIM, device=DEVICE)).clamp(0, 1).cpu()
    stats = gen_diversity_stats(samples, test_pool)
    save_grid(samples, "gan_samples_64.png", nrow=8)
    save_json({"history": hist, **stats}, "gan.json")

    # ------------------------------ DCGAN --------------------------------------
    set_seed(); t0 = time.perf_counter()
    G2, D2 = DCGAN_G().to(DEVICE), DCGAN_D().to(DEVICE)
    ck = os.path.join(ART, "weights", "dcgan.pt")
    if os.path.exists(ck) and not args.train:
        G2.load_state_dict(torch.load(ck, map_location=DEVICE))
        hist = load_json("dcgan.json")["history"]
        print("[dcgan] checkpoint loaded", flush=True)
    else:
        G2.apply(dcgan_weights_init); D2.apply(dcgan_weights_init)
        optG = torch.optim.Adam(G2.parameters(), lr=LR_GAN, betas=(0.5, 0.999))
        optD = torch.optim.Adam(D2.parameters(), lr=LR_GAN, betas=(0.5, 0.999))
        hist = []
        for ep in range(1, EPOCHS["dcgan"] + 1):
            for xb, _ in train_loader:
                xb = xb.to(DEVICE); bs = len(xb)
                real = torch.full((bs,), 0.9, device=DEVICE)
                fake = torch.zeros(bs, device=DEVICE)
                optD.zero_grad()
                d_loss = bce(D2(xb), real) + bce(D2(G2(torch.randn(bs, Z_DIM, device=DEVICE)).detach()), fake)
                d_loss.backward(); optD.step()
                optG.zero_grad()
                g_loss = bce(D2(G2(torch.randn(bs, Z_DIM, device=DEVICE))), real)
                g_loss.backward(); optG.step()
            hist.append({"epoch": ep, "d_loss": d_loss.item(), "g_loss": g_loss.item()})
            print(f"  dcgan ep{ep} D={d_loss.item():.3f} G={g_loss.item():.3f}", flush=True)
        torch.save(G2.state_dict(), ck)
    G2.eval()
    print(f"[dcgan] {time.perf_counter()-t0:.1f}s", flush=True)
    with torch.no_grad():
        samples = G2(torch.randn(64, Z_DIM, device=DEVICE)).clamp(0, 1).cpu()
    stats = gen_diversity_stats(samples, test_pool)
    save_grid(samples, "dcgan_samples_64.png", nrow=8)
    save_json({"history": hist, **stats}, "dcgan.json")

    # ------------------------------ DDPM ---------------------------------------
    set_seed(); t0 = time.perf_counter()
    unet = TinyUNet().to(DEVICE)
    ck = os.path.join(ART, "weights", "ddpm.pt")
    if os.path.exists(ck) and not args.train:
        unet.load_state_dict(torch.load(ck, map_location=DEVICE))
        hist = load_json("ddpm.json")["history"]
        print("[ddpm] checkpoint loaded", flush=True)
    else:
        opt = torch.optim.Adam(unet.parameters(), lr=2e-4)
        from src.models import betas, alphas, alpha_bar
        hist = []
        for ep in range(1, EPOCHS["ddpm"] + 1):
            run, n = 0.0, 0
            for xb, _ in train_loader:
                xb = xb.to(DEVICE); bs = len(xb)
                t = torch.randint(0, DDPM_T, (bs,), device=DEVICE)
                noise = torch.randn_like(xb)
                ab = alpha_bar[t].to(DEVICE).view(-1, 1, 1, 1)
                x_t = ab.sqrt() * xb + (1 - ab).sqrt() * noise
                opt.zero_grad()
                loss = F.mse_loss(unet(x_t, t), noise)
                loss.backward(); opt.step()
                run += loss.item(); n += 1
            hist.append({"epoch": ep, "eps_mse": run / n})
            print(f"  ddpm ep{ep} eps_mse={run/n:.4f}", flush=True)
        torch.save(unet.state_dict(), ck)
    unet.eval()
    print(f"[ddpm] {time.perf_counter()-t0:.1f}s", flush=True)
    samples = ddpm_sample(unet, 64, DEVICE).cpu()
    stats = gen_diversity_stats(samples, test_pool)
    save_grid(samples, "ddpm_samples_64.png", nrow=8)
    save_json({"history": hist, **stats}, "ddpm.json")

    print("[done] generative comparison pipeline complete", flush=True)


if __name__ == "__main__":
    main()
