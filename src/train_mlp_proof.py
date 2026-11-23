# =============================================================================
# src/train_mlp_proof.py — 5-stage empirical proof that MLP-classifier weights
# and latents fail at image reconstruction/generation (Bagian 2 of the brief).
#
# Tahap 1: train MLP classifier 784->256->64->10 (CE) to >=88% test acc
# Tahap 2: top-down weight transposition from one-hot class vectors
# Tahap 3: latent transpose projection of real test images
# Tahap 4: frozen-latent decoder (64->256->784, MSE) vs Vanilla AE z=64
# Tahap 5: random sampling z ~ N(mu_z, diag(sigma_z^2)) -> frozen decoder
#
# Usage:  python -m src.train_mlp_proof
# =============================================================================
import json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid, save_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import SEED, set_seed, build_dataloaders
from src.models import MLPClassifier, FrozenLatentDecoder, VanillaAE

ART = "artifacts_mlp_gen_proof"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_CLS_MAX, ACC_TARGET, LR_CLS = 30, 0.88, 1e-3
EPOCHS_DEC_MAX, LR_DEC = 40, 1e-3

for sub in ("weights", "latents", "grids", "metrics", "data"):
    os.makedirs(os.path.join(ART, sub), exist_ok=True)

_TIMER = {}


def save_tensor(t, name):
    path = os.path.join(ART, name)
    torch.save(t.detach().cpu() if torch.is_tensor(t) else t, path)
    print("[saved]", path)
    return path


def save_json(obj, name):
    path = os.path.join(ART, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print("[saved]", path)
    return path


class stage_timer:
    """Re-seeds the stage and logs wall-clock runtime."""
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        set_seed(SEED)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        _TIMER[self.name] = _TIMER.get(self.name, 0.0) + dt
        print(f"[timer] {self.name}: {dt:.2f}s", flush=True)


def save_timing():
    save_json(_TIMER, os.path.join("metrics", "timing_log.json"))


def render_grid(imgs, title, nrow=10, fname=None):
    grid = make_grid(imgs.detach().cpu().clamp(0, 1), nrow=nrow, padding=2)
    save_image(grid, os.path.join(ART, "grids", fname))
    print("[saved]", os.path.join(ART, "grids", fname))


@torch.no_grad()
def evaluate_accuracy(model, loader):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        pred = model(xb.to(DEVICE)).argmax(dim=1).cpu()
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / total


@torch.no_grad()
def extract_latents(model, loader):
    model.eval()
    zs, ys = [], []
    for xb, yb in loader:
        zs.append(model.features(xb.to(DEVICE)).cpu())
        ys.append(yb)
    return torch.cat(zs), torch.cat(ys)


def main():
    set_seed(SEED)
    train_loader, val_loader, test_loader = build_dataloaders()

    # --------------------------- Tahap 1: MLP classifier ----------------------
    clf = MLPClassifier().to(DEVICE)
    opt_cls = torch.optim.Adam(clf.parameters(), lr=LR_CLS)
    crit_cls = nn.CrossEntropyLoss()

    cls_history = []
    with stage_timer("step1_train_mlp_classifier"):
        for epoch in range(1, EPOCHS_CLS_MAX + 1):
            clf.train()
            run_loss, seen = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt_cls.zero_grad()
                loss = crit_cls(clf(xb), yb)
                loss.backward()
                opt_cls.step()
                run_loss += loss.item() * yb.numel()
                seen += yb.numel()
            train_loss = run_loss / seen
            val_acc = evaluate_accuracy(clf, val_loader)
            test_acc = evaluate_accuracy(clf, test_loader)
            cls_history.append({"epoch": epoch, "train_loss": train_loss,
                                "val_acc": val_acc, "test_acc": test_acc})
            print(f"[step1] epoch {epoch:02d} | train_loss={train_loss:.4f} "
                  f"| val_acc={val_acc:.4f} | test_acc={test_acc:.4f}", flush=True)
            if test_acc >= ACC_TARGET:
                break

    with torch.no_grad():
        W1 = clf.fc1.weight.detach().clone()
        W2 = clf.fc2.weight.detach().clone()
        W3 = clf.fc3.weight.detach().clone()
    save_tensor({"W1": W1, "W2": W2, "W3": W3,
                 "b1": clf.fc1.bias.detach().clone(),
                 "b2": clf.fc2.bias.detach().clone(),
                 "b3": clf.fc3.bias.detach().clone()},
                os.path.join("weights", "mlp_classifier_weights.pt"))
    save_tensor(clf.state_dict(), os.path.join("weights", "mlp_classifier_statedict.pt"))

    z_train, _ = extract_latents(clf, train_loader)
    z_val, _ = extract_latents(clf, val_loader)
    z_test, _ = extract_latents(clf, test_loader)
    save_tensor(z_train, os.path.join("latents", "z_mlp_train.pt"))
    save_tensor(z_val, os.path.join("latents", "z_mlp_val.pt"))
    save_tensor(z_test, os.path.join("latents", "z_mlp_test.pt"))

    step1 = {"final_test_acc": cls_history[-1]["test_acc"],
             "final_val_acc": cls_history[-1]["val_acc"],
             "epochs_trained": len(cls_history), "acc_target": ACC_TARGET,
             "target_met": cls_history[-1]["test_acc"] >= ACC_TARGET,
             "z_dim": int(z_test.shape[1]), "history": cls_history}
    save_json(step1, os.path.join("metrics", "step1_classifier.json"))
    save_timing()

    # --------------------------- Tahap 2: weight transposition ----------------
    NUM_CLASSES = 10
    IMG_SHAPE = (1, 28, 28)
    y_c = F.one_hot(torch.arange(NUM_CLASSES), num_classes=NUM_CLASSES).float().to(DEVICE)
    with torch.no_grad():
        z_hat = torch.relu(y_c @ W3)
        h1_hat = torch.relu(z_hat @ W2)
        x_hat2 = torch.sigmoid(h1_hat @ W1)
    img2 = x_hat2.view(-1, *IMG_SHAPE).cpu()
    save_tensor(img2, os.path.join("grids", "step2_class_prototypes.pt"))
    render_grid(img2, "Step 2 class prototypes via W^T", nrow=5, fname="step2_class_prototypes.png")
    save_json({"note": "zero-extra-training transposed-weight class prototypes",
               "shape": list(img2.shape)}, os.path.join("metrics", "step2_prototypes.json"))

    # --------------------------- Tahap 3: latent transpose projection ---------
    N_SHOW = 10
    with torch.no_grad():
        x_hat3 = torch.sigmoid(torch.relu(z_test[:N_SHOW].to(DEVICE) @ W2) @ W1)
    img3 = x_hat3.view(-1, *IMG_SHAPE).cpu()
    x_true3 = next(iter(test_loader))[0][:N_SHOW]
    save_tensor(torch.cat([x_true3, img3]), os.path.join("grids", "step3_side_by_side.pt"))
    render_grid(torch.cat([x_true3, img3]), "Step 3: GT (top) vs W^T projection (bottom)",
                nrow=N_SHOW, fname="step3_side_by_side.png")
    mse_step3 = F.mse_loss(img3, x_true3).item()
    save_json({"test_mse_transpose_projection": mse_step3, "n_pairs": N_SHOW},
              os.path.join("metrics", "step3_projection.json"))

    # --------------------------- Tahap 4: frozen latent decoder ---------------
    for p in clf.parameters():
        p.requires_grad = False
    clf.eval()

    @torch.no_grad()
    def collect_flat(loader):
        return torch.cat([xb.view(xb.size(0), -1) for xb, _ in loader]), None

    X_train_flat, _ = collect_flat(train_loader)
    X_test_flat, _ = collect_flat(test_loader)

    decoder = FrozenLatentDecoder().to(DEVICE)
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=LR_DEC)
    dec_train = DataLoader(TensorDataset(z_train, X_train_flat), batch_size=128, shuffle=True)

    dec_history = []
    with stage_timer("step4_train_frozen_latent_decoder"):
        for epoch in range(1, EPOCHS_DEC_MAX + 1):
            decoder.train()
            run, seen = 0.0, 0
            for zb, xb in dec_train:
                zb, xb = zb.to(DEVICE), xb.to(DEVICE)
                opt_dec.zero_grad()
                loss = F.mse_loss(decoder(zb), xb)
                loss.backward()
                opt_dec.step()
                run += loss.item() * zb.size(0)
                seen += zb.size(0)
            train_mse = run / seen

            # test metrics
            with torch.no_grad():
                x_rec = decoder(z_test.to(DEVICE)).cpu()
            mse_vec = F.mse_loss(x_rec, X_test_flat, reduction="none").mean(dim=1)
            psnr_vec = 10.0 * torch.log10(1.0 / mse_vec)
            ssim_vec = ssim_batch(x_rec.view(-1, *IMG_SHAPE), X_test_flat.view(-1, *IMG_SHAPE))
            dec_history.append({"epoch": epoch, "train_mse": train_mse,
                                "test_mse": mse_vec.mean().item(),
                                "test_psnr_db": psnr_vec.mean().item(),
                                "test_ssim": ssim_vec.mean().item()})
            print(f"[step4] epoch {epoch:02d} | train_mse={train_mse:.6f} "
                  f"| test_mse={dec_history[-1]['test_mse']:.6f}", flush=True)

    step4 = dict(dec_history[-1])
    step4["epochs_trained"] = len(dec_history)
    step4["classifier_frozen"] = all(not p.requires_grad for p in clf.parameters())
    step4["history"] = dec_history
    save_tensor(decoder.state_dict(), os.path.join("weights", "frozen_latent_decoder_statedict.pt"))
    save_tensor(x_rec.view(-1, *IMG_SHAPE), os.path.join("grids", "step4_test_reconstructions.pt"))
    save_json(step4, os.path.join("metrics", "step4_frozen_decoder.json"))
    save_timing()
    render_grid(torch.cat([X_test_flat[:N_SHOW].view(-1, *IMG_SHAPE),
                           x_rec[:N_SHOW].view(-1, *IMG_SHAPE)]),
                "Step 4", nrow=N_SHOW, fname="step4_reconstructions.png")

    # --------------------------- Tahap 5: random latent sampling --------------
    N_SAMPLES = 25
    mu_z = z_test.mean(dim=0)
    sigma_z = z_test.std(dim=0)
    save_tensor(mu_z, os.path.join("latents", "mu_z.pt"))
    save_tensor(sigma_z, os.path.join("latents", "sigma_z.pt"))

    set_seed(SEED)
    with stage_timer("step5_random_latent_sampling"):
        with torch.no_grad():
            z_rand = mu_z + torch.randn(N_SAMPLES, z_test.shape[1]) * sigma_z
            x_rand = decoder(z_rand.to(DEVICE)).cpu()
    save_tensor(z_rand, os.path.join("latents", "z_rand_sample.pt"))
    img5 = x_rand.view(-1, *IMG_SHAPE)
    save_tensor(img5, os.path.join("grids", "step5_generated_samples.pt"))
    render_grid(img5, "Step 5: 25 samples N(mu_z, diag(sigma_z^2))", nrow=5,
                fname="step5_generated_samples.png")

    save_json({"step1_final_test_acc": step1["final_test_acc"],
               "step4_test_mse": step4["test_mse"],
               "step4_test_psnr_db": step4["test_psnr_db"],
               "step4_test_ssim": step4["test_ssim"],
               "step3_transpose_projection_mse": mse_step3,
               "n_generated_samples": N_SAMPLES, "artifacts_dir": ART},
              os.path.join("metrics", "run_summary.json"))
    save_timing()
    print("[done] MLP proof pipeline complete")


def ssim_batch(pred, target):
    from src.metrics import ssim_per_image
    outs = []
    for i in range(0, len(pred), 1000):
        outs.append(ssim_per_image(pred[i:i + 1000], target[i:i + 1000]))
    return torch.cat(outs)


if __name__ == "__main__":
    main()
