# =============================================================================
# src/evaluate.py — evaluation & visualization suite over the trained models:
#   1) inference latency per batch-64 (single-pass models vs DDPM 400-step)
#   2) FID (5,000 synthetic samples per generative model vs 5,000 real test)
#   3) linear latent interpolation (Figur 4)
#   4) assembly of the five mandatory report figures (Figur 1-5)
#
# Usage:
#   python -m src.evaluate latency
#   python -m src.evaluate fid            # generates 5k samples + computes FID
#   python -m src.evaluate interpolation
#   python -m src.evaluate figures
#   python -m src.evaluate all
# =============================================================================
import json, os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

from src.data import SEED, set_seed, build_dataloaders
from src.models import Z_DIM, load_model, ddpm_sample
from src.metrics import fid_features, frechet_distance

ART = "artifacts_generative_compare"
ART3 = "artifacts_mlp_gen_proof"
GR = os.path.join(ART, "grids")
CACHE = os.path.join("_work", "fid_cache")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_MODELS = ["vanilla_ae", "sparse_ae", "vae", "gan", "dcgan"]


def save_json(obj, name):
    with open(os.path.join(ART, "metrics", name), "w") as f:
        json.dump(obj, f, indent=2)


def load_json(name):
    with open(os.path.join(ART, "metrics", name)) as f:
        return json.load(f)


# --------------------------- 1) latency ---------------------------------------
@torch.no_grad()
def run_latency():
    _, _, test_loader = build_dataloaders()
    XB64 = torch.stack([test_loader.dataset[i][0] for i in range(64)]).to(DEVICE)
    lat = {}
    for name in ["vanilla_ae", "sparse_ae", "vae", "gan", "dcgan"]:
        m = load_model(name)
        if name in ("vanilla_ae", "sparse_ae", "vae"):
            fn = lambda: m(XB64)
        else:
            def fn():
                return m(torch.randn(64, Z_DIM, device=DEVICE))
        for _ in range(3):
            fn()
        ts = []
        for _ in range(30):
            t0 = time.perf_counter(); fn()
            ts.append((time.perf_counter() - t0) * 1000)
        lat[name] = {"ms_per_batch64_mean": float(np.mean(ts)),
                     "ms_per_batch64_std": float(np.std(ts))}
        print(f"[latency] {name}: {np.mean(ts):.2f} ± {np.std(ts):.2f} ms/batch64", flush=True)

    def fn_vae_gen():
        return load_model("vae").dec(torch.randn(64, Z_DIM, device=DEVICE)).view(-1, 1, 28, 28)
    ts = []
    for _ in range(30):
        t0 = time.perf_counter(); fn_vae_gen(); ts.append((time.perf_counter() - t0) * 1000)
    lat["vae_prior_sampling"] = {"ms_per_batch64_mean": float(np.mean(ts)),
                                 "ms_per_batch64_std": float(np.std(ts))}
    unet = load_model("ddpm")
    runs = []
    for _ in range(3):
        t0 = time.perf_counter(); ddpm_sample(unet, 64, DEVICE)
        runs.append((time.perf_counter() - t0) * 1000)
    lat["ddpm"] = {"path": "400-step iterative denoising",
                   "runs_ms": runs,
                   "ms_per_batch64_mean": float(np.mean(runs)),
                   "ms_per_batch64_std": float(np.std(runs, ddof=1))}
    print(f"[latency] ddpm: {np.mean(runs):.0f} ms/batch64", flush=True)
    save_json(lat, "latency.json")
    print("latency.json saved")


# --------------------------- 2) FID -------------------------------------------
def _generate_fid_cache(n=5000, batch=250):
    os.makedirs(CACHE, exist_ok=True)
    set_seed()
    _, _, test_loader = build_dataloaders()
    if not os.path.exists(os.path.join(CACHE, "real.pt")):
        acc = []
        for xb, _ in test_loader:
            acc.append(xb)
            if sum(a.shape[0] for a in acc) >= n:
                break
        torch.save(torch.cat(acc)[:n], os.path.join(CACHE, "real.pt"))
        print("[fid] real reference cached", flush=True)
    for name in ["vae", "gan", "dcgan", "ddpm"]:
        out_path = os.path.join(CACHE, f"{name}.pt")
        if os.path.exists(out_path):
            print(f"[fid] {name}.pt exists, skip", flush=True)
            continue
        set_seed()
        m = load_model(name)
        out = []
        with torch.no_grad():
            while sum(o.shape[0] for o in out) < n:
                z = torch.randn(batch, Z_DIM, device=DEVICE)
                if name == "vae":
                    out.append(m.dec(z).view(-1, 1, 28, 28).clamp(0, 1).cpu())
                elif name in ("gan", "dcgan"):
                    out.append(m(z).clamp(0, 1).cpu())
                else:
                    out.append(ddpm_sample(m, batch, DEVICE).cpu())
                    print(f"[fid] ddpm batch {len(out)} done", flush=True)
        torch.save(torch.cat(out)[:n], out_path)
        print(f"[fid] {name} cached ({n} samples)", flush=True)


def run_fid(n=5000):
    _generate_fid_cache(n)
    real = torch.load(os.path.join(CACHE, "real.pt"), weights_only=True)
    f_real = fid_features(real, DEVICE)
    scores = {"n_samples_per_set": n,
              "real_reference": "first 5,000 MNIST test images (seed-42 split)"}
    # baseline: real vs a disjoint half of itself (sanity lower bound)
    half = n // 2
    scores["fid_real_vs_real_half"] = frechet_distance(f_real[:half], f_real[half:])
    for name in ["vae", "gan", "dcgan", "ddpm"]:
        syn = torch.load(os.path.join(CACHE, f"{name}.pt"), weights_only=True)
        scores[f"fid_{name}"] = frechet_distance(f_real, fid_features(syn, DEVICE))
        print(f"[fid] {name}: {scores[f'fid_{name}']:.2f}", flush=True)
    save_json(scores, "fid_scores.json")
    print("fid_scores.json saved")


# --------------------------- 3) interpolation (Figur 4) -----------------------
@torch.no_grad()
def run_interpolation():
    ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    _, _, test_loader = build_dataloaders()
    idx1 = next(i for i in range(len(test_loader.dataset)) if test_loader.dataset[i][1] == 1)
    idx8 = next(i for i in range(len(test_loader.dataset)) if test_loader.dataset[i][1] == 8)
    xA = test_loader.dataset[idx1][0].unsqueeze(0).to(DEVICE)
    xB = test_loader.dataset[idx8][0].unsqueeze(0).to(DEVICE)
    set_seed()
    zA_noise = torch.randn(1, Z_DIM, device=DEVICE)
    zB_noise = torch.randn(1, Z_DIM, device=DEVICE)
    rows = []
    for name in LATENT_MODELS:
        m = load_model(name)
        if name in ("vanilla_ae", "sparse_ae"):
            za, zb = m.enc(xA.flatten(1)), m.enc(xB.flatten(1))
        elif name == "vae":
            za, zb = m.encode(xA)[0], m.encode(xB)[0]
        else:
            za, zb = zA_noise, zB_noise
        imgs = []
        for a in ALPHAS:
            z = (1 - a) * za + a * zb
            imgs.append((m.dec(z).view(1, 1, 28, 28) if name in ("vanilla_ae", "sparse_ae", "vae")
                         else m(z)).clamp(0, 1))
        rows.append((name, torch.cat(imgs).cpu()))
    labels = {"vanilla_ae": "Vanilla AE", "sparse_ae": "Sparse AE", "vae": "VAE",
              "gan": "Vanilla GAN (ruang noise)", "dcgan": "DCGAN (ruang noise)"}
    fig, axes = plt.subplots(5, 1, figsize=(7.2, 11))
    for ax, (name, img) in zip(axes, rows):
        ax.imshow(make_grid(img, nrow=6, padding=2).permute(1, 2, 0).squeeze(), cmap="gray")
        ax.set_ylabel(labels[name], fontsize=9, rotation=0, ha="right", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_title("Figur 4 — Linear Latent Interpolation z(α)=(1−α)z_A+αz_B, α∈{0.0,...,1.0}\n"
                      "Anchors: digit '1' → digit '8' (model laten); dua vektor noise tetap (GAN/DCGAN)",
                      fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(GR, "figur4_interpolation.png"), dpi=150)
    plt.close(fig)
    print("figur4 saved")


# --------------------------- 4) the five mandatory figures --------------------
@torch.no_grad()
def run_figures():
    _, _, test_loader = build_dataloaders()
    X10 = torch.stack([test_loader.dataset[i][0] for i in range(10)]).to(DEVICE)

    # Figur 1
    m_ae, m_sae, m_vae = load_model("vanilla_ae"), load_model("sparse_ae"), load_model("vae")
    xh_ae, _ = m_ae(X10); xh_sae, _ = m_sae(X10); xh_vae, _, _ = m_vae(X10)
    rows1 = [X10.cpu(), xh_ae.clamp(0, 1).cpu(), xh_sae.clamp(0, 1).cpu(), xh_vae.clamp(0, 1).cpu()]
    fig, axes = plt.subplots(4, 1, figsize=(11, 6.2))
    for ax, img, lab in zip(axes, rows1,
                            ["Baris 1: Citra Asli (Ground Truth)", "Baris 2: Rekonstruksi Vanilla AE",
                             "Baris 3: Rekonstruksi Sparse AE", "Baris 4: Rekonstruksi VAE"]):
        ax.imshow(make_grid(img, nrow=10, padding=2).permute(1, 2, 0).squeeze(), cmap="gray")
        ax.set_title(lab, fontsize=10, loc="left"); ax.axis("off")
    fig.suptitle("Figur 1 — Grid Rekonstruksi Side-by-Side pada 10 Citra Uji MNIST", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(GR, "figur1_recon_compare.png"), dpi=150); plt.close(fig)
    print("figur1 saved", flush=True)

    # Figur 2
    set_seed()
    s_vae = m_vae.dec(torch.randn(16, Z_DIM, device=DEVICE)).view(-1, 1, 28, 28).clamp(0, 1).cpu()
    s_gan = load_model("gan")(torch.randn(16, Z_DIM, device=DEVICE)).clamp(0, 1).cpu()
    s_dcg = load_model("dcgan")(torch.randn(16, Z_DIM, device=DEVICE)).clamp(0, 1).cpu()
    s_ddpm = ddpm_sample(load_model("ddpm"), 16, DEVICE).cpu()
    panels = [(s_vae, "Panel A — VAE (prior z ~ N(0,I))"), (s_gan, "Panel B — Vanilla GAN (z ~ N(0,I))"),
              (s_dcg, "Panel C — DCGAN (z ~ N(0,I))"), (s_ddpm, "Panel D — DDPM (reverse diffusion x_T → x_0)")]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 8))
    for ax, (img, lab) in zip(axes.flat, panels):
        ax.imshow(make_grid(img, nrow=4, padding=2).permute(1, 2, 0).squeeze(), cmap="gray")
        ax.set_title(lab, fontsize=10); ax.axis("off")
    fig.suptitle("Figur 2 — Grid Random Generative Sampling (16 sampel per model)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(GR, "figur2_sampling_panels.png"), dpi=150); plt.close(fig)
    print("figur2 saved", flush=True)

    # Figur 3
    protos = torch.load(os.path.join(ART3, "grids", "step2_class_prototypes.pt"), weights_only=True)
    step3 = torch.load(os.path.join(ART3, "grids", "step3_side_by_side.pt"), weights_only=True)
    step4 = torch.load(os.path.join(ART3, "grids", "step4_test_reconstructions.pt"), weights_only=True)[:10]
    step5 = torch.load(os.path.join(ART3, "grids", "step5_generated_samples.pt"), weights_only=True)
    ae_recon10, _ = m_ae(X10)
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.2))
    specs = [
        (X10.cpu(), "(1) Citra Asli (Ground Truth)", 5),
        (torch.cat([protos, step3[10:20]]), "(2) Proyeksi balik Wᵀ — atas: prototype kelas y_c, bawah: proyeksi z_MLP", 5),
        (torch.cat([step4, ae_recon10.clamp(0, 1).cpu()]), "(3) Atas: Frozen Latent Decoder (z_MLP) vs Bawah: Vanilla AE (z_AE)", 5),
        (step5, "(4) Random sampling N(μ_z, diag(σ_z²)) — 25 sampel", 5),
    ]
    for ax, (img, lab, nrow) in zip(axes, specs):
        ax.imshow(make_grid(img.clamp(0, 1), nrow=nrow, padding=2).permute(1, 2, 0).squeeze(), cmap="gray")
        ax.set_title(lab, fontsize=9.5); ax.axis("off")
    fig.suptitle("Figur 3 — Bukti Empiris Batasan Bobot & Fitur Laten MLP Classifier untuk Image Generation",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(GR, "figur3_mlp_limitation.png"), dpi=150); plt.close(fig)
    print("figur3 saved", flush=True)

    # Figur 5
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.30)
    specs5 = [
        ("vanilla_ae", "Vanilla AE — train MSE", [("train_mse", "MSE")], "mse"),
        ("sparse_ae", "Sparse AE — train MSE + fraksi laten aktif", [("train_mse", "MSE")], "mse"),
        ("vae", "VAE — recon MSE + KL", [("recon_mse", "Recon MSE (sum/batch)"), ("kl", "KL")], "log"),
        ("gan", "Vanilla GAN — loss D & G", [("d_loss", "D loss"), ("g_loss", "G loss")], "lin"),
        ("dcgan", "DCGAN — loss D & G", [("d_loss", "D loss"), ("g_loss", "G loss")], "lin"),
        ("ddpm", "DDPM — eps-prediction MSE", [("eps_mse", "eps MSE")], "lin"),
    ]
    for k, (name, title, keys, scale) in enumerate(specs5):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        h = load_json(f"{name}.json")["history"]
        ep = [r["epoch"] for r in h]
        for key, lab in keys:
            ax.plot(ep, [r[key] for r in h], marker="o", ms=3, label=lab)
        if name == "sparse_ae":
            ax2 = ax.twinx()
            ax2.plot(ep, [r["latent_active_frac"] for r in h], color="tab:green", ls="--",
                     label="fraksi laten aktif")
            ax2.set_ylabel("fraksi aktif", color="tab:green")
            ax2.tick_params(axis="y", colors="tab:green")
            ax2.legend(loc="center right", fontsize=8)
        if scale == "log":
            ax.set_yscale("log")
        ax.set_xlabel("Epoch"); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    snap_ts = [399, 360, 320, 280, 240, 200, 160, 120, 80, 50, 25, 10, 0]
    set_seed(123)
    _, snaps = ddpm_sample(load_model("ddpm"), 4, DEVICE, snapshot_ts=snap_ts)
    gs2 = fig.add_gridspec(1, len(snaps) + 1, left=0.04, right=0.98, bottom=0.02, top=0.30)
    ax0 = fig.add_subplot(gs2[0, 0]); ax0.axis("off")
    ax0.text(0.5, 0.5, "DDPM reverse\ndiffusion\nx_T → x_0", ha="center", va="center", fontsize=10)
    for j, (t, img) in enumerate(snaps):
        ax = fig.add_subplot(gs2[0, j + 1]); ax.axis("off")
        ax.imshow(img[0].cpu().squeeze(), cmap="gray"); ax.set_title(f"t={t}", fontsize=8)
    fig.suptitle("Figur 5 — Kurva Training Loss Keenam Model & Sekuens Reverse Diffusion DDPM",
                 fontsize=12, y=0.995)
    fig.savefig(os.path.join(GR, "figur5_loss_curves_ddpm_trajectory.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("figur5 saved", flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("latency", "all"):
        run_latency()
    if cmd in ("figures", "all"):
        run_figures()
    if cmd in ("interpolation", "all"):
        run_interpolation()
    if cmd in ("fid", "all"):
        run_fid()


if __name__ == "__main__":
    main()
