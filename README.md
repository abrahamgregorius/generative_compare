# TUGAS DL SESI 07 — Studi Komparatif Empiris Model Generatif + Pembuktian Batasan MLP Classifier

**Nama:** Abraham Gregorius Anderson Thio — **NIM:** 2802473504
**Mata Kuliah:** Deep Learning (D6421) — Sesi 07: Generative Models (Take-Home)

Implementasi PyTorch modular untuk:
1. **Pembuktian empiris 5 tahap** bahwa bobot 𝑊⁽ˡ⁾ dan fitur laten 𝑧_MLP dari MLP Classifier
   (dioptimasi Cross-Entropy) **tidak dapat** digunakan untuk rekonstruksi/generasi citra (Bagian 2 panduan).
2. **Perbandingan enam arsitektur generatif** pada split data yang identik:
   Vanilla AE, Sparse AE, VAE, Vanilla GAN (MLP), DCGAN, DDPM.

## Struktur Repositori

```
├── src/
│   ├── data.py                    # split terstandarisasi MNIST (seed 42, 54k/6k/10k, piksel [0,1])
│   ├── models.py                  # semua arsitektur (MLP classifier, decoder, 6 model generatif)
│   ├── metrics.py                 # SSIM/PSNR/MSE (formula identik), diversity stats, FID
│   ├── train_mlp_proof.py         # pipeline pembuktian 5 tahap (Tahap 1–5)
│   ├── train_generative_compare.py# pipeline pelatihan 6 model generatif
│   └── evaluate.py                # latency, FID, interpolasi laten, Figur 1–5 laporan
├── MLP_Generation_Proof.ipynb     # notebook eksekusi pembuktian 5 tahap (output tertanam)
├── Generative_Models_Compare.ipynb# notebook eksekusi 6 model generatif (output tertanam)
├── TUGAS_DL_SESI07_2802473504_AbrahamGregoriusAndersonThio_ANALISIS.md  # laporan analisis
├── artifacts_mlp_gen_proof/       # artefak Tahap 1–5 (weights, latents, grids, metrics)
├── artifacts_generative_compare/  # artefak 6 model (weights, grids, metrics)
└── requirements.txt
```

## Cara Eksekusi

```bash
# 1) environment (Python 3.10+)
pip install -r requirements.txt

# 2) pembuktian 5 tahap MLP classifier (Tahap 1–5)
python -m src.train_mlp_proof

# 3) pelatihan/evaluasi 6 model generatif
#    - jika checkpoint sudah ada di artifacts_generative_compare/weights/, checkpoint DIMUAT
#      dan metrik/grid dihitung ulang; hapus checkpoint (atau tambahkan --train) untuk melatih ulang
python -m src.train_generative_compare [--train]

# 4) evaluasi tambahan (latency per batch-64, FID 5.000 sampel, interpolasi laten, Figur 1–5)
python -m src.evaluate all
```

Catatan: notebook dapat dijalankan dengan Jupyter/VS Code; pipeline `src/` menghasilkan artefak
yang identik (seed 42) dan menjadi sumber modular untuk reproducibility penuh.

## Protokol Eksperimen Terstandarisasi

| Variabel | Nilai |
|---|---|
| Random seed | `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)` |
| Split data | 54.000 train (80%) / 6.000 val (10%) / 10.000 test (10%), permutasi tetap seed 42 |
| Rentang piksel | [0, 1] (ToTensor, tanpa normalisasi ±) |
| Batch size | 128, shuffle=True (train) / shuffle=False (val & test) |
| Dimensi laten | 64 untuk semua model laten (AE, Sparse AE, VAE) & noise GAN/DCGAN |
| Formula metrik | SSIM (kernel Gaussian 11×11, σ=1.5) & PSNR identik untuk semua model |
| Evaluasi numerik | Seluruh metrik diukur pada Test Set (10.000 citra, belum pernah dilihat model) |

## Ringkasan Hasil Utama

| Model | Paradigma | Test MSE | PSNR (dB) | SSIM | Keterangan |
|---|---|---|---|---|---|
| MLP Classifier (Tahap 1) | klasifikasi CE | — | — | — | test acc 94,50%; laten efektif 5,94/64 |
| Frozen decoder z_MLP (Tahap 4) | rekonstruksi (beku) | 0,06793 | 11,84 | 0,1195 | ≈ baseline mean-image (0,06741) |
| Vanilla AE | rekonstruksi MSE | 0,00533 | 23,54 | 0,9355 | 12,7× lebih baik dari decoder beku |
| Sparse AE (λ=1e-3) | MSE + L1 laten | 0,00537 | 23,53 | 0,9309 | 62,5% unit laten aktif |
| VAE | MSE + KL | 0,02358 | 16,68 | 0,6813 | prior N(0,I) bisa di-sampling |
| GAN (MLP) | adversarial | N/A | — | — | pairwise L2 8,40 (data asli ≈ 9,9) |
| DCGAN | adversarial conv | N/A | — | — | pairwise L2 9,67; tanpa mode collapse |
| DDPM (T=400) | diffusi eps-pred | N/A | — | — | pairwise L2 7,33; eps-MSE 0,252→0,041 |

Analisis lengkap: `TUGAS_DL_SESI07_2802473504_AbrahamGregoriusAndersonThio_ANALISIS.md`.

## Lingkungan Pengujian

- Python 3.10, PyTorch CPU (`torch==2.14.1+cpu`, torchvision 0.29.1), Windows.
- Dataset MNIST diunduh otomatis via `torchvision.datasets` (publik, tanpa registrasi).
