# =============================================================================
# src/metrics.py — evaluation utilities: SSIM/PSNR/MSE (identical formulas
# across all models), GAN-family diversity stats, and FID (InceptionV3).
# =============================================================================
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg


# --------------------------- SSIM (identical to the notebooks) ----------------
def _gauss(k=11, sigma=1.5):
    ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)


@torch.no_grad()
def ssim_per_image(pred, target, k=11, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
    """pred/target: (N,1,H,W) in [0,1]. Returns per-image SSIM, shape (N,)."""
    kern = _gauss(k, sigma).to(pred.device).view(1, 1, k, k)
    mu_p, mu_t = F.conv2d(pred, kern), F.conv2d(target, kern)
    var_p = F.conv2d(pred * pred, kern) - mu_p ** 2
    var_t = F.conv2d(target * target, kern) - mu_t ** 2
    cov = F.conv2d(pred * target, kern) - mu_p * mu_t
    num = (2 * mu_p * mu_t + C1) * (2 * cov + C2)
    den = (mu_p ** 2 + mu_t ** 2 + C1) * (var_p + var_t + C2)
    return ((num / den).flatten(1)).mean(1)


@torch.no_grad()
def eval_recon(encode_decode, test_loader, device):
    """Mean Test MSE / PSNR / SSIM over the test set.
    encode_decode(x:(N,1,28,28) on `device`) -> x_hat in [0,1]."""
    mse_v, psnr_v, ssim_v = [], [], []
    for xb, _ in test_loader:
        xb = xb.to(device)
        xh = encode_decode(xb).clamp(0, 1)
        mse = F.mse_loss(xh, xb, reduction="none").flatten(1).mean(1)
        mse_v.append(mse.cpu())
        psnr_v.append(10 * torch.log10(1.0 / mse).cpu())
        ssim_v.append(ssim_per_image(xh, xb).cpu())
    mse_v = torch.cat(mse_v); psnr_v = torch.cat(psnr_v); ssim_v = torch.cat(ssim_v)
    return {"test_mse": mse_v.mean().item(),
            "test_psnr_db": psnr_v.mean().item(),
            "test_ssim": ssim_v.mean().item()}


# --------------------------- GAN-family diversity stats -----------------------
def gen_diversity_stats(samples, test_pool):
    """samples: (N,1,28,28) in [0,1]; test_pool: (M,784) real images.
    Mean pairwise L2 between samples + nearest-neighbour L2 to real test data."""
    s = samples.flatten(1)
    d = torch.cdist(s, s)
    off = d[d > 0]
    nn_dist = torch.cdist(s, test_pool).min(dim=1).values
    return {"mean_pairwise_l2": off.mean().item(),
            "mean_nn_l2_to_test": nn_dist.mean().item(),
            "max_nn_l2_to_test": nn_dist.max().item()}


# --------------------------- FID (Frechet Inception Distance) -----------------
@torch.no_grad()
def fid_features(images, device, batch_size=100):
    """images: (N,1,28,28) tensor in [0,1] -> (N,2048) InceptionV3 pool3 features.
    Grayscale MNIST is replicated to 3 channels; the FID InceptionV3 resizes to 299."""
    from pytorch_fid.inception import InceptionV3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device).eval()
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)   # (b,1,28,28)
        batch = batch.repeat(1, 3, 1, 1)              # (b,3,28,28)
        f = model(batch)[0]
        feats.append(f.squeeze(-1).squeeze(-1).cpu())
    return torch.cat(feats)


def frechet_distance(feats_a, feats_b, eps=1e-6):
    """FID between two sets of Inception features (Heusel et al., 2017)."""
    mu1, mu2 = feats_a.mean(0).numpy(), feats_b.mean(0).numpy()
    s1 = torch.cov(feats_a.T).numpy()
    s2 = torch.cov(feats_b.T).numpy()
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(covmean.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + offset).dot(s2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))
