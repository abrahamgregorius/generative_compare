# TUGAS DL SESI 07 — Analisis Lengkap: Pembuktian Batasan MLP Classifier & Perbandingan Enam Model Generatif

**Nama:** Abraham Gregorius Anderson Thio
**NIM:** 2802473504
**Mata Kuliah:** Deep Learning (D6421) — Sesi 07: Generative Models (Take-Home)
**Dataset:** MNIST (28×28, grayscale, 10 kelas) — split terstandarisasi seed 42: train 54.000 / val 6.000 / test 10.000
**Perangkat:** CPU (PyTorch 2.14.1+cpu), piksel [0,1], z_dim 64 di seluruh model laten
**Notebook eksekusi:** `MLP_Generation_Proof.ipynb` (pembuktian 5 tahap) & `Generative_Models_Compare.ipynb` (6 model generatif)

> **Rujukan panduan:** *Tugas Take-Home D6421 — Deep Learning, Sesi 07* (`soal.md`).
> **Struktur dokumen:** Bagian III = analisis pembuktian batasan MLP classifier (Tahap 1–5); Bagian IV = analisis perbandingan enam model generatif sebagai pembanding langsung Tahap 4–5; Lampiran = artefak & ringkasan eksekusi notebook.

---

## Daftar Isi

- **Ringkasan Eksekutif** — tabel metrik konsolidasi + temuan utama
- **Bagian III** — Analisis Eksperimen Batasan MLP Classifier untuk Rekonstruksi & Generasi Citra
  - 1. Ringkasan Hasil & Temuan Utama
  - 2. Analisis Detail per Tahap (Tahap 1–5)
  - 3. Sintesis: Bukti Berantai Kelima Tahap
- **Bagian IV** — Perbandingan Empiris Enam Model Generatif
  - 0. Protokol Eksperimen Bersama
  - 1. Hasil Kuantitatif Lintas Model
  - 2. Kounterpoint Tahap 5: Sampling Prior VAE
  - 3. Observasi Visual per Model
  - 4. Sintesis Lintas Eksperimen
- **Lampiran** — Artefak, ringkasan eksekusi notebook, rujukan

---

# Ringkasan

Dua eksperimen saling melengkapi pada split data dan formula metrik yang identik (seed 42, 54k/6k/10k, [0,1], z_dim 64):

1. **Bagian III** membuktikan secara empiris (5 tahap) bahwa bobot 𝑊⁽ˡ⁾ dan laten 𝑧_MLP dari MLP classifier yang dioptimasi Cross-Entropy **gagal** untuk rekonstruksi maupun generasi citra: proyeksi 𝑊ᵀ menghasilkan template blur, decoder beku jatuh ke konstanta rata-rata (MSE 0,06793 ≈ baseline mean-image 0,06741), dan prior sampling menghasilkan 25/25 blob identik.
2. **Bagian IV** menunjukkan bahwa keenam model generatif yang dilatih pada data yang sama **berhasil**: Vanilla/Sparse AE merekonstruksi **12,7× lebih baik** dari decoder beku, VAE menghasilkan prior yang bisa di-sampling, dan GAN/DCGAN/DDPM menunjukkan keragaman sampel sehat tanpa mode collapse. Satu-satunya variabel yang membedakan adalah **tujuan optimisasi** — bukan kapasitas, bukan dimensi laten, bukan data.

### Tabel konsolidasi seluruh hasil

| Model / Jalur | Paradigma | Epoch | Metrik utama | Nilai |
|---|---|---|---|---|
| MLP Classifier (Tahap 1) | Klasifikasi (CE) | 1 | Test accuracy | **94,50%** (participation ratio laten 5,94/64) |
| Proyeksi 𝑊ᵀ (Tahap 2) | Transpos bobot, tanpa latihan | — | Visual | 10 prototype blur, tak terbaca |
| Proyeksi 𝑊ᵀ dari z_MLP (Tahap 3) | Transpos bobot | — | Test MSE | **0,3257** (≈4,8× decoder terlatih) |
| **Frozen MLP decoder** (Tahap 4) | Decoder beku di z_MLP | 40 | MSE / PSNR / SSIM | **0,06793 / 11,84 dB / 0,1195** ≈ mean-image (0,06741) |
| Sampling N(μ_z, σ_z²) (Tahap 5) | Prior empiris | — | Visual | 25/25 sampel nyaris identik |
| **Vanilla AE** | Rekonstruksi (MSE) | 15 | MSE / PSNR / SSIM | **0,00533 / 23,54 dB / 0,9355** |
| **Sparse AE** (λ=1e-3) | Rekonstruksi + L1 | 15 | MSE / PSNR / SSIM | **0,00537 / 23,53 dB / 0,9309** (62,5% latent aktif) |
| **VAE** | Rekonstruksi + KL | 15 | MSE / PSNR / SSIM | **0,02358 / 16,68 dB / 0,6813** + 25 sampel prior beragam |
| **GAN** (MLP) | Adversarial | 30 | pairwise L2 / NN-L2 ke test | **8,40 / 5,00** |
| **DCGAN** | Adversarial (conv) | 10 | pairwise L2 / NN-L2 ke test | **9,67 / 6,20** |
| **DDPM** | Diffusi (T=400) | 8 | pairwise L2 / NN-L2 ke test | **7,33 / 5,78** |
| *(ref)* Baseline mean-image | Trivial | — | Test MSE | 0,06741 |
| *(ref)* Data test asli | — | — | mean pairwise L2 | ≈ 9,9 |

---

# Bagian III — Analisis Eksperimen: Pembuktian Empiris Batasan MLP Classifier untuk Rekonstruksi dan Generasi Citra

> **Artefak eksperimen:** notebook `MLP_Generation_Proof.ipynb` dan direktori `artifacts_mlp_gen_proof/` (weights, latents, grids, metrics).
> **Perangkat eksekusi:** CPU (PyTorch), total waktu pelatihan tercatat ≈ 90,5 s (lihat `metrics/timing_log.json`).

## III.1 Ringkasan Hasil Eksperimen & Temuan Utama

### III.1.1 Capaian Utama

| Tahap | Target Eksperimen | Hasil Aktual | Status |
|---|---|---|---|
| 1. MLP Classifier 784→256→64→10 | Test accuracy ≥ 88% | **94,50%** (1 epoch; val 94,08%; train loss 0,3876) | ✅ Tercapai |
| 2. Top-Down Weight Transposition | Prototype 10 kelas tanpa pelatihan tambahan | Terbangun (10×1×28×28), namun **bercak blur tidak terbaca** | ✅ Pembuktian gagal-generatif |
| 3. Latent Transpose Projection z→x | Rekonstruksi citra uji via W2ᵀ, W1ᵀ | Test MSE = **0,3257** (vs citra asli), visual inverted/blob | ✅ Pembuktian gagal-generatif |
| 4. Frozen Latent Decoder 64→256→784 | Test MSE / PSNR / SSIM terlog | MSE **0,06793**, PSNR **11,84 dB**, SSIM **0,1195** (40 epoch) | ✅ Terlog |
| 5. Random Sampling N(μ_z, diag(σ_z²)) | Grid 25 sampel generatif | 25 sampel **nyaris identik** — blur blob sama semua | ✅ Pembuktian gagal-generatif |

### III.1.2 Temuan Utama (Key Findings)

1. **Classifier sangat akurat, tetapi latent-nya hampir tak informatif secara rekonstruktif.** MLP classifier mencapai 94,5% test accuracy, namun seluruh jalur generatif yang dibangun di atas bobot/latennya (Tahap 2–5) menghasilkan citra tidak terbaca.
2. **Transposed weight bukan decoder.** Proyeksi `Wᵀ` (Tahap 2–3) menghasilkan template kasar dan citra "negatif blob" dengan MSE 0,3257 — ±4,8× lebih buruk daripada decoder terlatih Tahap 4 (0,0679).
3. **Decoder beku konvergen ke "gambar rata-rata".** Test MSE decoder berbasis z_MLP (0,06793) bahkan **sedikit lebih buruk** daripada baseline trivial *mean-image* (MSE 0,06741 — memprediksi citra rata-rata dataset untuk semua input). Diversity keluaran decoder nyaris nol (rincian di §III.2.4).
4. **Ruang laten z_MLP efektif hanya ~6 dimensi.** Dari 64 dimensi, participation ratio eigenvalue kovarians z_test = **5,94**; beberapa dimensi bahkan mati (σ_z = 0). Variansi antar-kelas hanya 2,05× variansi intra-kelas → komponen diskriminatif mendominasi, variasi spasial intra-kelas tereliminasi.
5. **Sampling tanpa prior P(z) menghasilkan mode-unik.** Kelima tahap secara bersama membuktikan: optimasi Cross-Entropy pada P(y|x) *tidak* membangun struktur ruang laten yang kontinu, beraturan, atau bergenerasi.

### III.1.3 Keterkaitan Empiris: Tujuan Klasifikasi vs Hilangnya Variasi Spasial

Cross-Entropy Loss mengoptimalkan P(y|x) — gradien hanya menuntut **agar logit kelas benar terpisah**, bukan agar latent menyimpan "bagaimana citra itu tampak". Konsekuensi empiris yang terukur pada eksperimen ini:

- **Invariance-by-design:** agar klasifikasi robust, representasi z_MLP justru didorong *mengabaikan* posisi ketebalan stroke, kemiringan, dan gaya tulisan (variasi intra-kelas). Terukur: rasio variansi antar-kelas : intra-kelas di z hanya **2,05** — komponen spesifik-instansi kecil dan tersebar tidak beraturan di 64 dimensi.
- **Tidak ada tekanan rekonstruktif:** tidak ada istilah kerugian yang mengikat z ke x. Akibatnya peta z→x tidak terdefinisi oleh pelatihan klasifikasi, sehingga (a) proyeksi ad-hoc Wᵀ gagal (Tahap 2–3), dan (b) decoder eksternal pun hanya sanggup mempelajari fallback konstanta rata-rata (Tahap 4), karena informasi spasial sudah **dibuang secara permanen** oleh encoder klasifikasi — membekukan bobot (requires_grad = False) tidak mengembalikan informasi yang tak pernah disimpan.
- **Tidak ada regularisasi prior:** pelatihan CE tidak membentuk distribusi P(z); ruang laten hanya berupa kumpulan titik "pulau" per kelas tanpa struktur kontinu. Ini yang membuat sampling acak (Tahap 5) jatuh di antara pulau manifold dan menghasilkan artefak tidak utuh.

> **Kesimpulan ringkas:** akurasi tinggi (94,5%) dan kualitas generatif rendah (SSIM 0,12; sampel tak beragam) **dicitrakan pada model yang sama**, membuktikan secara empiris bahwa keberhasilan klasifikasi ≠ kapabilitas generasi.

---

## III.2 Analisis Detail Hasil per Tahap

### III.2.1 Tahap 1 — Pelatihan MLP Classifier (784 → 256 → 64 → 10)

**Konfigurasi:** Adam lr=1e-3, CrossEntropyLoss, batch 128, seed 42, early-stop saat test acc ≥ 88% (cap 30 epoch).

| Metrik | Nilai |
|---|---|
| Epoch terlatih | 1 (early-stop) |
| Train loss (CE) | 0,3876 |
| Val accuracy | 94,08% |
| **Test accuracy** | **94,50%** (target ≥ 88% ✅) |
| Dimensi laten z_MLP | 64 (R⁶⁴, keluaran ReLU(fc2) sebelum logits) |
| Waktu pelatihan | 9,12 s |

**Analisis konvergensi.** Target akurasi terlampaui hanya dalam 1 epoch — arsitektur tiered ini sangat efektif sebagai *diskriminator*. Justru kecepatan konvergensi ini yang menegaskan fungsinya: gradien CE dengan cepat menemukan arah fitur yang memisahkan 10 kelas, tanpa insentif apa pun membuat 64 dimensi tersebut *rekonstruktif*.

**Struktur laten (dari artefak `latents/z_mlp_*.pt`):**

| Statistik z_test | Nilai | Implikasi |
|---|---|---|
| Participation ratio (dimensi efektif) | **5,94 / 64** | Variansi terkonsentrasi pada ~6 arah kelas-pemisah |
| Dimensi mati (σ_z = 0) | ada (min σ_z = 0,0) | Unit ReLU mati — kapasitas laten tak terpakai |
| σ_z rata-rata / maks | 1,844 / 3,025 | Skala latent tidak dinormalisasi, tak dipandu prior |
| Variansi antar-kelas : intra-kelas | **2,05 : 1** | Sinyal identitas kelas dominan; detail instansi tersisa tipis |

**Kesamaan struktur dengan encoder autoencoder.** Jalur `784 → ReLU → 256 → ReLU → 64` secara topologi identik dengan encoder Vanilla AE laten-64. **Perbedaan murni terletak pada loss**: tanpa MSE rekonstruksi di ujung decoder, bottleneck 64-dim ini hanya dipelajari untuk diskriminasi. Kesamaan arsitektur + perbedaan tujuan optimasi inilah variabel kontrol yang membuat perbandingan Tahap 4 (vs Vanilla AE dimensi sama) menjadi apple-to-apple.

> **📄 Observasi visual — `grids/` (Tahap 1):** tidak ada grid citra (tahap ini menghasilkan bobot + latent). Artefak: `weights/mlp_classifier_weights.pt` (W1 256×784, W2 64×256, W3 10×64 + bias), `latents/z_mlp_{train,val,test}.pt`, `metrics/step1_classifier.json`.

---

### III.2.2 Tahap 2 — Top-Down Weight Transposition dari One-Hot y_c

**Prosedur:** `ẑ = ReLU(W3ᵀ y_c)` (R¹⁰→R⁶⁴) → `ĥ = ReLU(W2ᵀ ẑ)` (R⁶⁴→R²⁵⁶) → `x̂ = Sigmoid(W1ᵀ ĥ)` (R²⁵⁶→R⁷⁸⁴), **tanpa pelatihan tambahan**.

**Hasil (artefak `grids/step2_class_prototypes.png|.pt`):**

> **📄 Catatan observasi visual (grid 2×5, 10 kelas):**
> Kesepuluh prototype berbentuk **bercak abu-abu blur berk reparasi rendah**; terlihat samar guratan melengkung dan guratan vertikal, tetapi **tidak satu pun dapat dibaca sebagai digit MNIST**. Prototype antar-kelas tampak hampir seragam — perbedaan tipis di posisi stroke, dengan latar abu-abu seragam (bukan hitam).

**Penyebab matematis/operasional:**

1. **Wᵀ ∉ W⁻¹.** Transpose hanyalah "jalur balik arah gradien", bukan invers. Forward pass memampatkan 784→64 (kompresi ~12×); Wᵀ hanya memetakan kembali ke ruang 784-dim **sejajar baris-lalu-kolom**, tanpa kemampuan membalik kehilangan informasi.
2. **WᵀW ≠ I (matriks tak-ortogonal).** Bobot hasil Adam pada Cross-Entropy tidak pernah dikendarkan ortogonalitas. Secara analitis pada eksperimen ini, z untuk satu kelas diproyeksikan balik melewati dua Lapisan tak-ortogonal + non-linearitas, sehingga energi diproyeksikan ke arah-eigen dominan (arah pemisah kelas), bukan arah detail citra.
3. **Pemotongan negatif berulang oleh ReLU.** Setiap tahap `ReLU(Wᵀ ·)` memotong sebagian komponen — korelasi silang antar-baris W yang negatif (umum pada fitur terlatih) dibuang. Setelah dua pemotongan beruntun, sinyal yang tersisa adalah residual korelasi positif rata-rata → **smear abu-abu**, bukan stroke.
4. **y_c adalah titik vertex simplex, bukan titik manifold citra.** One-hot R¹⁰ hidup di ruang label; titik-titik z yang dihasilkan W3ᵀ tidak dijamin berada pada manifold latent citra asli.

**Kesimpulan tahap:** membalik panah klasifikasi dengan mengalikan bobot transpose menghasilkan *template kasar terdegradasi* — bukti pertama bahwa decoder tak dapat "dapat-diperoleh-gratis" dari classifier.

---

### III.2.3 Tahap 3 — Latent Transpose Projection dari Citra Uji Nyata

**Prosedur:** `x̂ = Sigmoid(W1ᵀ ReLU(W2ᵀ z_MLP))` untuk 10 citra uji pertama, dibandingkan berdampingan dengan ground truth.

**Hasil kuantitatif (artefak `metrics/step3_projection.json`):**

| Metrik | Nilai |
|---|---|
| MSE(x̂, ground truth) per 10 pasang uji | **0,3257** |
| Pembanding: MSE decoder terlatih Tahap 4 | 0,0679 |
| Pembanding: baseline mean-image (trivial) | 0,0674 |
| Degradasi vs decoder terlatih | ±**4,8× lebih buruk** |

> **📄 Catatan observasi visual (`grids/step3_side_by_side.png`):**
> Baris atas (ground truth): digit "7 2 1 0 4 1 4 9 6 9" tajam hitam-putih.
> Baris bawah (proyeksi Wᵀ): **blob putih tebal dengan lubang gelap berbentuk kabur digit** — tampak seperti *invers/negatif* bentuk asli, dikelilingi noise piksel acak di tepi. Digit "1" nyaris lenyap menjadi guratan tipis; digit berlubang (0, 9, 6) tampak sebagai cincin tidak rata.

**Mengapa forward weights klasifikasi bukan decoder rekonstruktif:**

- **Arah kolom W1/W2 menyandi "deteksi fitur", bukan "penggambaran fitur".** Baris W1 dipelajari sebagai template-detektor (bagian citra yang *membentuk* aktivasi h1), sehingga mengalikan balik dengan W1ᵀ menjumlahkan kontribusi template pada lokasi yang salah (flip spasial/terbalik) — konsisten dengan tampilan blob-negatif pada grid.
- **Informasi sudah hilang di bottleneck.** z_MLP 64-dim mempertahankan terutama identitas kelas; detail spasial (ketebalan, kemiringan, posisi) tidak ada di z, sehingga **tidak ada dekoder apa pun** — apalagi Wᵀ linear-ReLU — yang dapat mengembalikannya.
- **Absennya bias dan kalibrasi rentang.** Proyeksi transposed tidak memakai bias lapisan dan sigmoid mengikat ke [0,1]; komposisi tersebut menghasilkan offset global (latar terang) yang terlihat pada grid, bukan kontras hitam-putih citra asli.

**Kesimpulan tahap:** bahkan ketika latent berasal dari citra nyata, jalur balik berbasis bobot klasifikasi menghasilkan MSE ~5× lebih buruk dan struktur visual invers — bukti kedua yang melengkapi Tahap 2: kegagalan bersifat **struktural** (loss klasifikasi), bukan sekadar karena input y_c sintetis.

---

### III.2.4 Tahap 4 — Frozen Latent Decoder (64→256→784, MSE) vs Vanilla Autoencoder Laten-64

**Prosedur:** seluruh parameter classifier dibekukan (`requires_grad = False`, terverifikasi `classifier_frozen: true` pada artefak). Decoder standalone dilatih 40 epoch, MSE, Adam lr=1e-3, memetakan z_MLP beku → citra.

**Hasil kuantitatif (artefak `metrics/step4_frozen_decoder.json`):**

| Metrik | Frozen-Decoder (z_MLP beku) | Vanilla AE (laten-64, pembanding langsung) | Selisih |
|---|---|---|---|
| Test MSE | **0,06793** | **0,00533** | AE **12,7× lebih baik** (−92,2%) |
| Test PSNR (dB) | **11,84** | **23,54** | **+11,70 dB** |
| Test SSIM | **0,1195** | **0,9355** | **7,8× lebih tinggi** (+0,816) |
| Epoch terlatih | 40 | 15 | AE mencapainya dalam sepertiga waktu |
| MSE akhir vs baseline mean-image (0,06741) | **+0,8% lebih buruk** | **−92,1% (12,7× lebih baik)** | hanya AE yang benar-benar rekonstruktif |

*(Kolom Vanilla AE dari artefak `artifacts_generative_compare/metrics/vanilla_ae.json`, seed 42, split & formula metrik identik.)*

**Dinamika pelatihan (40 epoch):**

| Indikator | Epoch 1 | Epoch 40 | Interpretasi |
|---|---|---|---|
| Train MSE | 0,07068 | 0,06721 | Turun hanya **−4,9%** sepanjang pelatihan |
| Test MSE | 0,06855 | 0,06793 | Saturasi hampir seketika |
| Test PSNR | 11,79 dB | 11,84 dB | +0,05 dB — stagnan |
| Test SSIM | 0,1269 | **0,1195** | Justru **menurun** — semakin lama dilatih semakin mirip rata-rata |

**Analisis selisih terhadap Vanilla AE (interpretasi):**

1. **Decoder beku jatuh ke solusi trivial.** Test MSE 0,06793 ≈ baseline mean-image 0,06741. Terukur langsung dari artefak `grids/step4_test_reconstructions.pt` (10.000 rekonstruksi):
   - MSE(x̂_i, x̂_mean) = **8,3×10⁻⁵** → seluruh rekonstruksi nyaris *konstanta* yang sama;
   - std per-piksel antar-rekonstruksi **0,0065** vs ground truth **0,1924** (±30× lebih datar);
   - rata-rata jarak pasangan L2 antar-rekonstruksi **0,30** vs antar-citra asli **9,90** (±33× lebih seragam);
   - korelasi antar-rata-rata rekonstruksi 10 kelas = **0,9989–0,9996** → semua kelas didekode menjadi citra yang sama.
2. **Kontrol Vanilla AE menutup argumen variabel-bersama.** Vanilla AE laten-64 (arsitektur encoder identik 784→256→64, kapasitas dan bottleneck setara, seed 42, split sama) yang encoder-nya *ikut* dilatih bersama decoder di bawah MSE mencapai Test MSE **0,00533**, PSNR **23,54 dB**, SSIM **0,9355** — **12,7× lebih baik** dari decoder beku pada seluruh metrik, bahkan dengan hanya 15 epoch (vs 40). Karena satu-satunya perbedaan mendasar adalah *tujuan pelatihan encoder* (klasifikasi vs rekonstruksi), selisih dua kolom di tabel di atas adalah kuantifikasi langsung dari "informasi yang dibuang oleh tujuan klasifikasi" — decoder beku tak mungkin menutupnya karena input-nya (z_MLP) sudah kehilangan detail.
3. **PSNR 11,84 dB & SSIM 0,1195 = kualitas sangat rendah.** Sebagai konteks, rekonstruksi autoencoder layak pada MNIST umumnya >20 dB dengan SSIM >0,7. Nilai di level "gambar kabur abu-abu" ini konsisten dengan dekode konstanta-rata-rata, dan menegaskan bottleneck klasifikasi bukan bottleneck generatif.

> **📄 Catatan observasi visual (`grids/step4_reconstructions.png`):**
> Baris atas: ground truth "7 2 1 0 4 1 4 9 6 9" (tajam, beragam bentuk).
> Baris bawah: **kesepuluh rekonstruksi praktis identik** — blob oval abu-abu blur mirip angka "0" generik, tanpa jejak stroke digit sumber. Perbedaan antar-digital sumber tidak terlihat mata: bukti visual langsung dari collapse ke citra rata-rata.

**Kesimpulan tahap:** dengan bobot encoder dibekukan, pelatihan decoder eksternal *berhasil dijalankan* (bukti prosedural) namun *gagal secara fungsional* — MSE-nya tak lebih baik daripada menebak rata-rata. Ini menunjukkan batasan ada pada **representasi**, bukan pada kapasitas decoder.

---

### III.2.5 Tahap 5 — Random Sampling pada Ruang Laten MLP Classifier

**Prosedur:** μ_z dan σ_z dihitung dari z_test; 25 sampel z_rand ~ N(μ_z, diag(σ_z²)) didekode oleh decoder beku Tahap 4.

**Statistik sampling (artefak `latents/mu_z.pt`, `sigma_z.pt`):**

| Statistik | Nilai |
|---|---|
| σ_z rata-rata antar-dimensi | 1,844 |
| σ_z maksimum / minimum | 3,025 / **0,0** (ada dimensi mati) |
| Struktur kovarians yang dipakai | **diagonal saja** (diag(σ_z²)) — korelasi antar-dimensi diabaikan |

> **📄 Catatan observasi visual (`grids/step5_generated_samples.png`, grid 5×5):**
> Ke-25 sampel **saling hampir identik**: blob oval abu-abu blur (mirip keluaran Tahap 4), tanpa satu pun digit terbaca. Tidak ada variasi kelas, tidak ada variasi stroke, tidak ada artefak digit "setengah jadi" yang berbeda antar-sampel — semuanya jatuh ke citra rata-rata yang sama.

**Mengapa sampling di ruang laten MLP menghasilkan artefak/objek tidak utuh:**

1. **Ketiadaan regularisasi prior P(z).** Pelatihan klasifikasi (dan pelatihan decoder beku) tidak pernah menetapkan bentuk distribusi laten. N(μ_z, diag(σ_z²)) hanyalah *deskripsi empiris titik-titik test*, bukan prior generatif. Titik acak dengan probabilitas tinggi berada di **ruang antara pulau-pulau kelas** yang tak pernah di training — decoder, yang hanya hafal pemetaan titik-pulau → rata-rata, membalas dengan keluaran rata-rata yang sama.
2. **Manifold tidak kontinu / tidak beraturan.** Participation ratio 5,94 menunjukkan latent efektif rendah-dimensional tetapi **terdispersi tanpa topologi**: tetangga ruang latent tidak menjamin tetangga citra (tanpa KL-regularization ala VAE, tak ada tekanan membuat kurva latent halus). Interpolasi maupun sampling karenanya tidak bermakna semantik.
3. **Kovarians diagonal mengabaikan korelasi.** Pulau-pulau kelas di R⁶⁴ berkorelasi kuat antar-dimensi; sampling diagonal mengisi kombinasi dimensi yang **tidak pernah terjadi** pada data asli, memperparah keluaran di luar manifold.
4. **Decoder kekal pada fixpoint rata-rata.** Karena Tahap 4 membuktikan decoder menghasilkan konstanta untuk seluruh test set (MSE antar-rekonstruksi 8,3×10⁻⁵), masukan acak apa pun — masuk akal maupun bukan — tetap dipetakan ke blob yang sama. Artefak yang diminta panduan ("objek tidak utuh") pada run ini termanifes sebagai **identitas kosong**: kegagalan paling radikal, yaitu kegagalan menghasilkan variasi sama sekali.

**Kesimpulan tahap:** sampel acak dari ruang laten classifier tidak menghasilkan distribusi citra baru; hasilnya deterministik dan degeneratif. Generasi membutuhkan ruang laten yang **diregularisasi oleh prior** (mis. KL pada VAE) — sesuatu yang tak dimiliki MLP classifier, sebagaimana dibuktikan lima tahap ini.

---

## III.3 Sintesis: Bukti Berantai Kelima Tahap

| Bukti | Tahap | Kuantifikasi |
|---|---|---|
| Classifier akurat namun laten ~6 dim efektif | 1 | acc 94,5%; participation ratio 5,94/64 |
| Wᵀ menghasilkan template blur, bukan digit | 2 | 10 prototype tak terbaca |
| Proyeksi Wᵀ dari citra nyata: MSE buruk + struktur invers | 3 | MSE 0,3257 (≈4,8× decoder; ≈4,8× mean-image) |
| Decoder beku = dekoder konstanta-rata-rata | 4 | MSE 0,06793 vs mean-image 0,06741; PSNR 11,84 dB; SSIM 0,1195; std antar-rekonstruksi 0,0065 |
| Sampling tanpa prior → keluaran degeneratif identik | 5 | 25/25 sampel nyaris sama |

**Pernyataan pembuktian:** MLP classifier yang dioptimasi dengan Cross-Entropy membangun representasi diskriminatif yang **secara sistematis membuang variasi spasial intra-kelas**, tidak menyediakan jalur balik x→z yang valid (Wᵀ ≠ W⁻¹), dan tidak menjamin struktur manifold laten yang kontinu atau berprior. Akibatnya — ditunjukkan secara empiris — reconstrusi maupun generasi citra gagal di semua jalur yang diturunkan dari classifier, tanpa pelatihan encoder tambahan.

---

# Bagian IV — Perbandingan Empiris Enam Model Generatif sebagai Pembanding Tahap 4–5

**Status:** analisis berdasarkan eksekusi nyata `Generative_Models_Compare.ipynb` (seed 42, CPU, MNIST).
**Konteks:** Bagian III membuktikan bahwa bobot & laten MLP classifier gagal sebagai generator (5 tahap). Bagian ini melengkapi dengan **enam arsitektur generatif yang dilatih langsung pada split data yang identik**, sehingga kegagalan MLP bisa dibandingkan secara apples-to-apples dengan keberhasilan model yang memang dioptimasi untuk rekonstruksi/generasi.

## IV.0 Protokol Eksperimen Bersama (kontrol variabel)

Semua model berbagi konfigurasi yang persis sama — identik dengan notebook utama `MLP_Generation_Proof.ipynb`:

| Variabel | Nilai |
|---|---|
| Seed | 42 |
| Split data | 54.000 train / 6.000 val / 10.000 test |
| Rentang piksel | [0, 1] (tanpa normalisasi ±) |
| Dimensi laten | 64 (untuk semua model laten: AE, Sparse AE, VAE) |
| Formula metrik | SSIM & PSNR identik dengan notebook utama |
| Perangkat | CPU (torch 2.14.1+cpu) |

**Model laten** (Vanilla AE, Sparse AE, VAE) memakai encoder 784→256→64 — **topologi identik dengan encoder MLP classifier** di Bagian III. Ini membuat Vanilla AE menjadi kontrol variabel sempurna untuk Tahap 4: satu-satunya perbedaan dengan classifier adalah *tujuan pelatihan* (rekonstruksi vs klasifikasi).

**Model generatif murni** (GAN, DCGAN, DDPM) tidak memiliki laten 64-dim yang sebanding; mereka dievaluasi dengan statistik keragaman sampel (bukan metrik rekonstruksi).

**Catatan budget (jujur soal keterbatasan):** seluruh pelatihan dijalankan pada CPU dengan budget terbatas — AE/Sparse AE/VAE 15 epoch, GAN 30 epoch, DCGAN 10 epoch, DDPM 8 epoch (T=400, UNet kecil). Kualitas GAN/DCGAN/DDPM di bawah ini adalah **kualitas ber-budget-CPU**, bukan batas atas arsitekturnya.

---

## IV.1 Hasil Kuantitatif Lintas Model

### IV.1.1 Tabel utama

| Model | Paradigma | Epoch | Waktu (s) | Metrik utama | Nilai |
|---|---|---|---|---|---|
| **Vanilla AE** | Rekonstruksi (MSE) | 15 | 113,0 | Test MSE / PSNR / SSIM | **0,00533 / 23,54 dB / 0,9355** |
| **Sparse AE** | Rekonstruksi + L1 (λ=1e-3) | 15 | 115,3 | Test MSE / PSNR / SSIM | **0,00537 / 23,53 dB / 0,9309** |
| **VAE** | Rekonstruksi + KL (prior N(0,I)) | 15 | 129,6 | Test MSE / PSNR / SSIM | **0,02358 / 16,68 dB / 0,6813** |
| **GAN** (MLP) | Adversarial (z~N(0,I) → citra) | 30 | 283,7 | pairwise L2 antar-sampel / NN-L2 ke test | **8,40 / 5,00** |
| **DCGAN** | Adversarial (konvolusional) | 10 | 443,0 | pairwise L2 / NN-L2 ke test | **9,67 / 6,20** |
| **DDPM** | Diffusi (eps-prediction, T=400) | 8 | 23.171,7* | pairwise L2 / NN-L2 ke test | **7,33 / 5,78** |
| *(ref)* Frozen MLP decoder | Decoder beku di z_MLP (Tahap 4) | 40 | — | Test MSE / PSNR / SSIM | 0,06793 / 11,84 dB / 0,1195 |
| *(ref)* Baseline mean-image | Menebak citra rata-rata | — | — | Test MSE | 0,06741 |

\* Waktu DDPM terdistorsi throttle CPU (6,4 jam wall-clock untuk 8 epoch); jangan dibaca sebagai biaya komputasi intrinsik.

**Catatan variansi sampling:** statistik GAN/DCGAN/DDPM pada tabel ini berasal dari run pelatihan (`metrics/*.json`). Saat notebook `Generative_Models_Compare.ipynb` dieksekusi ulang (checkpoint dimuat, sampel diregenerasi dengan noise acak baru), statistik ini bergeser sedikit karena variansi sampling — mis. DCGAN pairwise L2 9,67 → 10,02, DDPM 7,33 → 7,70. Selisih ini normal untuk 64 sampel dan tidak mengubah kesimpulan (semua tetap jauh dari mode collapse).

### IV.1.1a FID (Fréchet Inception Distance) — 5.000 sampel per model

Dihitung belakangan dengan protokol standar: fitur InceptionV3 pool3 (2048-d, paket `pytorch-fid`), 5.000 sampel sintetis per model vs 5.000 citra test asli (grayscale direplikasi 3 kanal). Baseline FID antara dua paruh non-overlap citra test asli = **4,8** (floor bias ukuran 5k sampel).

| Model | FID (5k) | Interpretasi |
|---|---|---|
| **DCGAN** | **24,0** | Terbaik pada budget ini — sampel paling bersih & kontras (konsisten §IV.3.5) |
| **VAE** | 67,0 | Blur (𝔼[x\|z]) namun distribusinya tetap dekat MNIST |
| **GAN (MLP)** | 102,5 | Stroke tipis/sebagian terputus — kapasitas FC terbatas |
| **DDPM** | 177,3 | Undertrained (8 epoch × T=400, UNet kecil; eps-MSE belum plateau) — limitasi budget, bukan arsitektur |
| *(ref)* real vs real-half | 4,8 | Floor bias |

`artifacts_generative_compare/metrics/fid_scores.json`; protokol & cache sampel: `_work/fid_cache/` (5.000 sampel per model, seed 42). Analisis lengkap pada laporan Q1.

### IV.1.2 Pembandingan tiga keluarga

- **Rekonstruksi murni (AE, Sparse AE):** MSE 0,0053 — **12,7× lebih baik dari frozen decoder** dan **12,6× lebih baik dari baseline mean-image**. Ini adalah kuantifikasi langsung dari klaim Tahap 4: ketika encoder *ikut* dioptimasi rekonstruktif, bottleneck 64-dim ternyata menyimpan detail visual yang oleh classifier justru **dibuang**.
- **Rekonstruksi ber-prior (VAE):** MSE 0,0236 — lebih buruk dari AE pada rekonstruksi (trade-off tak terhindarkan: kapasitas laten dibayar untuk KL), tetapi satu-satunya model laten yang punya **prior eksplisit** → bisa sampling (lihat §IV.2).
- **Generatif murni (GAN, DCGAN, DDPM):** tak punya pasangan input–output sehingga tak punya MSE rekonstruksi; keragaman sampelnya (pairwise L2 7,3–9,7) berada di urutan yang sama dengan pairwise L2 antar-citra test asli (≈9,9) → **tidak ada mode collapse** pada ketiganya pada budget ini.

### IV.1.3 Sparse AE vs Vanilla AE — efek regularisasi sparsity

| Metrik | Vanilla AE | Sparse AE (λ=1e-3) |
|---|---|---|
| Test MSE | 0,00533 | 0,00537 (+0,7%) |
| Test PSNR | 23,54 dB | 23,53 dB (≈sama) |
| Test SSIM | **0,9355** | 0,9309 (−0,5%) |
| Fraksi laten aktif (test) | — (tidak dijerat) | **62,5%** |

Penalti sparsity hampir gratis: kualitas rekonstruksi turun <1%, tetapi representasi menjadi lebih hemat (37,5% unit laten mati rata-rata per citra). Sparsity berperan sebagai **regularisasi** — bandingkan dengan laten MLP classifier yang participation ratio-nya hanya 5,94/64: keduanya menekankan bahwa *cara* memaksa bottleneck efektif (L1 pada aktivasi vs Cross-Entropy) menentukan apakah informasi visual dipertahankan.

---

## IV.2 Kounterpoint Langsung ke Tahap 5: Sampling dari Prior VAE

Tahap 5 Bagian III menunjukkan sampling z ~ N(μ_z, diag(σ_z²)) pada laten classifier menghasilkan **25/25 sampel blob identik**. VAE adalah pembuktian bahwa masalahnya ada pada *ketiadaan prior*, bukan pada sampling acak itu sendiri — dengan arsitektur encoder/decoder setara, hanya tambahan KL q(z|x)‖N(0,I):

| Aspek | Sampling laten MLP (Tahap 5) | Sampling prior VAE (z ~ N(0,I)) |
|---|---|---|
| Sumber z | Statistik empiris test (bukan prior) | Prior eksplisit hasil regularisasi KL |
| Variasi antar-sampel | ≈ nol (MSE antar-rekonstruksi 8,3×10⁻⁵) | Nyata — 25 sampel berbeda bentuk & kelas |
| Digit terbaca | 0 dari 25 | Mayoritas terbaca, sebagian ambigu |
| Struktur manifold | Pulau terdispersi, tanpa topologi | Kontinu — interpolasi bermakna |

> **📄 Observasi visual (`grids/vae_prior_samples_25.png`, grid 5×5):**
> Ke-25 sampel **beragam**: terbaca 0, 2, 3, 4, 8, 9 dengan gaya stroke berbeda-beda; sebagian sampel ambigu/half-formed (blob yang bisa dibaca "6" atau kurva tak lengkap) — persis jenis artefak "objek tidak selalu utuh" yang diharapkan pada generator nyata, tetapi **dengan variasi yang sama sekali tidak dimiliki sampel Tahap 5**. Kualitas lebih blur daripada rekonstruksi VAE — konsekuensi standar prior Gaussian yang menghaluskan decoder.

---

## IV.3 Observasi Visual per Model

### IV.3.1 Vanilla AE (`grids/vanilla_ae_recon_grid.png`)

> Baris atas (ground truth "7 2 1 0 4 1 4 9 5 9") dan baris bawah (rekonstruksi) **praktis identik** — stroke sedikit lebih lembut, ketebalan sedikit berbeda, namun seluruh digit tetap jelas dan bentuk khas tiap penulisan (mis. ekor "9" yang melengkung) terjaga. SSIM 0,9355 konsisten dengan kesan mata: perbedaan nyaris tak terdeteksi.

### IV.3.2 Sparse AE (`grids/sparse_ae_recon_grid.png`)

> Rekonstruksi juga nyaris identik dengan ground truth, dengan stroke sedikit **lebih tipis/tipis-halus** daripada Vanilla AE — jejak ringan dari penalti L1 yang menekan aktivasi. Secara visual tak bisa dibedakan dari Vanilla AE; metrik mengonfirmasi selisih <1%.

### IV.3.3 VAE (`grids/vae_recon_grid.png`)

> Rekonstruksi **semua digit tetap terbaca**, tetapi secara konsisten lebih blur dan stroke lebih tebal/kabur daripada AE — standar untuk VAE dengan β=1 (tekanan KL menghaluskan decoder). Beberapa digit di baris terakhir sedikit terdistorsi. Kesannya: VAE membayar ~2,8× MSE untuk mendapatkan prior yang bisa di-sampling (§IV.2) — trade-off yang justru *dibutuhkan* untuk generasi.

### IV.3.4 GAN MLP (`grids/gan_samples_64.png`, 8×8)

> Mayoritas sampel adalah **digit yang bisa dikenali** (2, 3, 4, 7, 9 dominan), stroke agak tipis dan sebagian terputus; beberapa sampel setengah jadi. Keragaman bentuk nyata antar-sampel (pairwise L2 8,40). Loss D≈0,67 / G≈2,15 stabil tanpa tanda collapse — cukup sehat untuk budget 30 epoch CPU.

### IV.3.5 DCGAN (`grids/dcgan_samples_64.png`, 8×8)

> Secara visual **paling bersih di antara generator murni**: stroke tegas dan kontras baik, digit beragam dan kebanyakan utuh (0, 1, 2, 3, 6, 7, 9), meski masih ada sampel ambigu. Pairwise L2 9,67 mendekati keragaman data asli (≈9,9). Hanya 10 epoch — stabilitas loss (D 0,70→1,11, G 1,95→1,08) menunjukkan konvergensi sehat, kualitas akan naik dengan budget lebih.

### IV.3.6 DDPM (`grids/ddpm_samples_64.png`, 8×8)

> Sampel berbentuk digit yang dikenali tetapi bertekstur **noise-ish/grainy** — beberapa stroke pecah atau bertekstur kasar (konsekuensi hanya 8 epoch × T=400 pada UNet kecil; eps-MSE masih turun 0,252→0,041 dan belum plateau, jadi model jelas undertrained). Pairwise L2 7,33 dan NN-L2 5,78 sebanding dengan GAN: keragaman sehat, ketajaman belum.

### IV.3.7 Perbandingan langsung dengan keluaran MLP (Tahap 4–5)

> Kontras visual paling telak: grid `step4_reconstructions.png` / `step5_generated_samples.png` di Bagian III berisi **10 dan 25 salinan blob abu-abu yang sama persis**, sedangkan setiap grid di atas berisi puluhan digit berbeda. Tidak ada satu model pun di bagian ini yang mengalami degenerasi konstanta-rata-rata — degenerasi itu eksklusif pada jalur yang diturunkan dari bobot classifier.

---

## IV.4 Sintesis: Peta Kesimpulan Lintas Eksperimen

| Pertanyaan | Jawaban empiris | Bukti |
|---|---|---|
| Apakah bottleneck 64-dim cukup untuk merekonstruksi citra? | **Ya** — MSE 0,0053 (12,7× lebih baik dari decoder beku) | Vanilla/Sparse AE vs Tahap 4 |
| Apakah kegagalan Tahap 4 disebabkan kapasitas atau tujuan pelatihan? | **Tujuan pelatihan** — arsitektur sama, hasil 12,7× beda | Kontrol variabel §IV.1.2 |
| Apakah sampling acak dari ruang laten selalu degeneratif? | **Tidak** — asalkan laten diregularisasi prior | VAE prior sampling (§IV.2) vs Tahap 5 |
| Apakah generator adversarial/diffusi mengalami mode collapse di budget ini? | **Tidak** — pairwise L2 7,3–9,7 ≈ keragaman data asli | §IV.1.2, grid §IV.3.4–3.6 |
| Apakah regularisasi laten lain (sparsity) merusak rekonstruksi? | **Tidak** — penalti <1% dengan representasi lebih hemat | §IV.1.3 |

**Pernyataan penutup:** keenam model memvalidasi dua sisi dari kesimpulan Bagian III. (1) Model rekonstruktif membuktikan informasi visual **bisa** lewat bottleneck 64-dim — kegagalan decoder beku Tahap 4 murni karena z_MLP dibangun oleh tujuan klasifikasi. (2) Model generatif murni + prior VAE membuktikan sampling citra baru **bisa** dilakukan — kegagalan Tahap 5 murni karena ruang laten classifier tak berprior dan tak kontinu. Dengan kata lain: batasan MLP classifier bukan pada data, kapasitas, maupun dimensi laten, melainkan pada **tujuan optimisasi yang membuang informasi generatif**.

---

# Lampiran

## Lampiran A — Daftar Artefak Bagian III (`artifacts_mlp_gen_proof/`)

| Jalur | Isi |
|---|---|
| `weights/mlp_classifier_weights.pt` | W1 (256×784), W2 (64×256), W3 (10×64) + bias |
| `weights/mlp_classifier_statedict.pt`, `weights/frozen_latent_decoder_statedict.pt` | state_dict lengkap |
| `latents/z_mlp_{train,val,test}.pt` | z_MLP ∈ R⁶⁴ per split |
| `latents/mu_z.pt`, `sigma_z.pt`, `z_rand_sample.pt` | statistik sampling & sampel Tahap 5 |
| `grids/step2_class_prototypes.png` | prototype 10 kelas via Wᵀ |
| `grids/step3_side_by_side.png` | ground truth vs proyeksi Wᵀ (10 pasang) |
| `grids/step4_reconstructions.png`, `step4_test_reconstructions.pt` | rekonstruksi decoder beku |
| `grids/step5_generated_samples.png` | 25 sampel acak |
| `metrics/step{1–5}_*.json`, `run_summary.json`, `timing_log.json` | seluruh metrik & waktu |

## Lampiran B — Pembanding Vanilla AE (Hasil Eksekusi)

| Metrik | Frozen-Decoder (z_MLP beku) | Vanilla AE (laten-64) | Selisih |
|---|---|---|---|
| Test MSE | 0,06793 | **0,00533** | **12,7×** lebih baik (−92,2%) |
| Test PSNR (dB) | 11,84 | **23,54** | **+11,70 dB** |
| Test SSIM | 0,1195 | **0,9355** | **7,8×** lebih tinggi |

> Sumber: `artifacts_generative_compare/metrics/vanilla_ae.json` (run seed 42, 15 epoch, CPU; split 54k/6k/10k, piksel [0,1], z_dim 64, formula SSIM/PSNR identik dengan notebook utama). Analisis lintas-6-model lengkap ada pada **Bagian IV** dokumen ini.

## Lampiran C — Daftar Artefak Bagian IV (`artifacts_generative_compare/`)

| Jalur | Isi |
|---|---|
| `Generative_Models_Compare.ipynb` | Notebook eksekusi 6 model (output tertanam) |
| `weights/{vanilla_ae,sparse_ae,vae,gan,dcgan,ddpm}.pt` | Bobot keenam model |
| `grids/*.png` (7 berkas) | Grid rekonstruksi & sampel (dikutip di §IV.3) |
| `metrics/{model}.json`, `generative_compare_summary.json` | Metrik per model + ringkasan |

## Lampiran D — Ringkasan Eksekusi Notebook

Kedua notebook dieksekusi penuh dengan output tertanam:

- **`MLP_Generation_Proof.ipynb`** — 6 code cell, seluruhnya sukses; Tahap 1 (94,50% acc) hingga Tahap 5 (25 sampel degeneratif) dengan metrik & grid tertanam.
- **`Generative_Models_Compare.ipynb`** — 9 code cell, seluruhnya sukses (~40 s: checkpoint dimuat, metrik & grid dihitung ulang fresh; hapus checkpoint untuk melatih ulang dari nol dengan jalur kode identik). Cell penutup mencetak tabel konsolidasi 6 model + referensi MLP.

## Rujukan

1. Tim Pengampu D6421 Deep Learning, *Tugas Take-Home — Sesi 07: Generative Models* (`soal.md`), dokumen panduan tugas. `[tambahkan nama berkas/nomor halaman panduan final]`
2. Artefak eksekusi: `MLP_Generation_Proof.ipynb` + `artifacts_mlp_gen_proof/` (run seed 42, CPU, MNIST).
3. Artefak eksekusi: `Generative_Models_Compare.ipynb` + `artifacts_generative_compare/` (run seed 42, CPU, MNIST).
