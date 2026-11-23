Deep Learning (D6421) Sesi 07: Generative Models | Tugas Take-Home

# **PANDUAN TUGAS TAKE-HOME MAHASISWA**

**Studi Komparatif Empiris Model Generatif: Autoencoder (Vanilla & Sparse), GAN (Vanilla & DCGAN), VAE, dan DDPM disertai Pembuktian Batasan Teoretis Model Klasifikasi untuk Image Generation**

|**Dosen Pengampu**|Muhammad Fikri Hasani, S.Kom., M.T. & Tim Dosen Pengampu D6421|
|---|---|
|**Topik Perkuliahan**|Sesi 07: Generative Models (GANs & Diffusion) terintegrasi Sesi 06<br>(Autoencoders)|
|**Sifat & Bentuk Pengerjaan**|Tugas Mandiri / Individu (Proyek Eksperimen Komparatif)|
|**Luaran Akhir**|Laporan Ilmiah Komprehensif (PDF, 10-12 Halaman) + Repositori Publik<br>GitHub|

## **1. Latar Belakang Masalah dan Tujuan Pembelajaran**

Pemodelan data tanpa supervisi (unsupervised representation learning) merupakan fondasi dalam rekayasa deep learning modern. Sesi perkuliahan 06 membahas kompresi representasi melalui Autoencoder deterministik (Vanilla Autoencoder dan Sparse Autoencoder) serta pemodelan probabilistik Variational Autoencoder (VAE). Sesi perkuliahan 07 memperluas paradigma menuju pemodelan generatif modern: Generative Adversarial Networks (Vanilla GAN berbasis fully connected dan Deep Convolutional GAN / DCGAN) serta Denoising Diffusion Probabilistic Models (DDPM). Masing-masing keluarga arsitektur memiliki karakteristik perumusan matematis, fungsi loss objektif, serta kompromi rekayasa (trade-off) yang berbeda pada kualitas sampel, inference latency, dan kestabilan optimasi.

Tugas proyek mandiri ini mengintegrasikan materi Sesi 06 dan Sesi 07 ke dalam studi komparatif empiris berskala penuh. Mahasiswa ditugaskan untuk mengimplementasikan, melatih, mengevaluasi, serta membandingkan enam arsitektur generatif pada benchmark dataset yang sama secara terstandarisasi. Selain mengevaluasi model generatif, mahasiswa wajib merancang eksperimen pembuktian matematis dan empiris untuk menunjukkan batasan model klasifikasi terawasi (discriminative classifier) dalam menghasilkan sampel citra baru.

- **Vanilla Autoencoder**

- **Sparse Autoencoder**

- **Generative Adversarial Network (GAN)**

- **Variational Autoencoder (VAE)**

- **Deep Convolutional GAN (DCGAN)**

- **Denoising Diffusion Probabilistic Models (DDPM** )

#### **PRINSIP EVALUASI TERSTANDARISASI**

Seluruh model generatif wajib dilatih menggunakan dataset yang sama, train-val-test split yang seragam, serta penetapan random seed yang tetap (misalnya: `torch.manual_seed(42)` dan `np.random.seed(42)` ). Standardisasi ini diperlukan agar perbedaan performa empiris murni mencerminkan inductive bias masing-masing keluarga model.

## **2. Pembuktian Batasan Teoretis Model Klasifikasi untuk Image Generation**

Pada perancangan arsitektur deep learning, struktur lapisan ekstraksi fitur pada Multi-Layer Perceptron (MLP) Classifier (misalnya pemetaan dimensi 784 →256 →64) tampak identik secara struktural dengan bagian Encoder pada Autoencoder. Meskipun demikian, setelah melewati beberapa lapisan tersembunyi (hidden layers), matriks bobot 𝑊<sup>(𝑙)</sup> dan representasi fitur

Halaman 1 dari 6

Deep Learning (D6421) Sesi 07: Generative Models | Tugas Take-Home

laten (𝑧MLP) yang diperoleh untuk klasifikasi tidak dapat langsung digunakan untuk merekonstruksi maupun menyintesis citra baru (image generation).

Hal ini disebabkan oleh tiga batasan matematis utama: (1) MLP Classifier meminimalkan Cross-Entropy Loss untuk memodelkan probabilitas kondisional 𝑃(𝑦∣𝑥), sehingga jaringan membuang variasi spasial intra-kelas dan hanya menyisakan fitur pembeda kelas (information bottleneck); (2) matriks bobot persegi panjang 𝑊<sup>(𝑙)</sup> tidak bersifat ortogonal maupun invertibel (( 𝑊<sup>(𝑙)</sup> )<sup>𝑇</sup> 𝑊<sup>(𝑙)</sup> ≠ 𝐈 ) serta fungsi aktivasi ReLU(𝑢) = 𝑚𝑎𝑥(0, 𝑢) memotong nilai negatif secara ireversibel; dan (3) ruang fitur laten 𝑧MLP tidak memiliki regularisasi kerapatan prior 𝑃(𝑧). Mahasiswa wajib membuktikan ketiga batasan ini secara empiris melalui eksperimen kode.

### **2.1 Prosedur Eksperimen Mandiri: Pembuktian Ketidakmampuan Bobot dan Latent Features MLP Classifier untuk Image Generation**

Tahapan eksperimen pembuktian empiris yang wajib dijalankan oleh setiap mahasiswa:

- **Tahap 1: Pelatihan MLP Classifier Berjenjang:** Latih satu arsitektur Multi-Layer Perceptron (MLP) Classifier pada benchmark dataset pilihan dengan dimensi lapisan tersembunyi yang menyerupai Encoder Autoencoder (misalnya: 784 → 256 →64 →10) menggunakan Cross-Entropy Loss hingga mencapai akurasi test set minimal 88%. Simpan matriks bobot terlatih (𝑊1, 𝑊2, 𝑊3) dan ekstrak representasi fitur laten 𝑧MLP ∈ℝ<sup>64</sup> pada lapisan sebelum klasifikasi akhir.

- **Tahap 2: Proyeksi Balik Bobot Langsung dari Vektor Kelas (Top-Down Weight Transposition):** Gunakan matriks bobot MLP Classifier yang telah terlatih untuk memproyeksikan balik vektor one-hot kelas target 𝑦𝑐 ∈ℝ<sup>10</sup> menuju ruang piksel tanpa pelatihan tambahan melalui operasi transpos matriks: 𝑧̂ = ReLU(𝑊3𝑇𝑦𝑐) , dilanjutkan ℎ̂1 = ReLU(𝑊𝑇2𝑧)̂ ,

  - hingga 𝑥̂ = Sigmoid(𝑊1𝑇ℎ̂1). Tunjukkan secara visual bahwa citra 𝑥 yang dihasilkan berupa pola bercak kabur (superposisi ̂ template bobot kasar) karena 𝑊<sup>𝑇</sup> ≠𝑊<sup>−1</sup> .

- **Tahap 3: Proyeksi Balik dari Latent Features Citra Uji (Latent Transpose Projection):** Ambil vektor fitur laten 𝑧MLP ∈ℝ<sup>64</sup> hasil ekstraksi citra uji riil, kemudian proyeksikan langsung ke ruang piksel menggunakan transpos bobot lapisan ekstraktor fitur: 𝑥̂ = Sigmoid(𝑊1𝑇ReLU(𝑊2𝑇𝑧MLP)). Bandingkan hasilnya terhadap citra asli guna membuktikan bahwa bobot maju (forward weights) klasifikasi tidak dapat langsung dibalik menjadi dekoder rekonstruktif.

- **Tahap 4: Evaluasi Kandungan Informasi Latent Features (Frozen Latent Decoding):** Bekukan seluruh bobot MLP Classifier (requires_grad = False). Latih satu jaringan Decoder MLP terpisah (64 →256 →784) menggunakan MSE Loss untuk merekonstruksi citra asli dari fitur laten 𝑧MLP yang dibekukan. Bandingkan metrik rekonstruksinya (MSE, PSNR, SSIM) terhadap Vanilla Autoencoder berdimensi laten sama (𝑧AE ∈ℝ<sup>64</sup> ) untuk membuktikan hilangnya detail spasial intrakelas akibat ekstraksi fitur diskriminatif.

- **Tahap 5: Uji Sampling Generatif pada Ruang Laten MLP Classifier:** Hitung vektor rata-rata 𝜇𝑧 dan simpangan baku 𝜎𝑧 dari fitur laten 𝑧MLP pada test set, lalu tarik sampel acak 𝑧rand~𝒩(𝜇𝑧,diag(𝜎2𝑧 )) dan teruskan ke Decoder dari Tahap 4. Dokumentasikan bahwa citra hasil sampling gagal membentuk struktur objek baru yang utuh karena ruang laten MLP Classifier tidak membentuk generative manifold yang kontinu dan teratur.

## **3. Pilihan Benchmark Dataset dan Standar Preprocessing**

Mahasiswa wajib memilih satu dari benchmark dataset publik berikut. Dataset diunduh secara legal melalui pustaka torchvision.datasets tanpa memerlukan registrasi berbayar:

|**Benchmark Dataset**|**Dimensi & Format**|**Tingkat Kompleksitas**|**Akses via PyTorch**|
|---|---|---|---|
|**Fashion-MNIST (Sangat**|28x28 Grayscale, 10 Kategori|Menengah (kontur tekstil,|torchvision.datasets.FashionMNIST|
|**Direkomendasikan)**|Pakaian (Fashion Items),<br>70.000 citra|variasi siluet)||
|MNIST (Standar Benchmark)|28x28 Grayscale, 10 Digit<br>Angka (0-9), 70.000 citra|Dasar (latar hitam, goresan<br>stroke biner)|torchvision.datasets.MNIST|

Halaman 2 dari 6

Deep Learning (D6421) Sesi 07: Generative Models | Tugas Take-Home

|Kuzushiji-MNIST (KMNIST)|28x28 Grayscale, 10 Karakter<br>Hiragana Kuno, 70.000 citra|Menengah (kaligrafi cursive<br>tradisional)|torchvision.datasets.KMNIST|
|---|---|---|---|
|CIFAR-10|32x32 RGB (3 Kanal), 10|**Tinggi**(foto natural, variasi|torchvision.datasets.CIFAR10|
||Kategori Objek Riil, 60.000<br>citra|latar kompleks)||

#### **Aturan Standardisasi Preprocessing:**

- Dataset Split: Bagi dataset menjadi Train Set (80%), Validation Set (10%), dan Test Set (10%). Seluruh metrik numerik wajib dievaluasi pada Test Set yang belum pernah dilihat model.

- Batch Size & DataLoader: Gunakan batch_size = 128 atau 256 dengan shuffle=True pada training set dan shuffle=False pada validation/test set.

## **4. Pertanyaan Penyelidikan Ilmiah (Pertanyaan Inti Eksperimen)**

Mahasiswa wajib menyajikan analisis komparatif mendalam pada laporan akhir dengan menjawab enam pertanyaan penelitian berikut:

- **Q1: Perbandingan Kuantitatif: Reconstruction Fidelity vs Generative Quality:** Bagaimana komparasi kuantitatif kemampuan rekonstruksi antara Vanilla Autoencoder, Sparse Autoencoder, dan Variational Autoencoder (VAE) yang dievaluasi melalui metrik Test MSE, Test PSNR, dan Test SSIM? Jelaskan bagaimana penambahan penalti regularisasi sparsity 𝜆‖ℎ‖1 pada representasi laten Sparse AE memengaruhi fidelitas rekonstruksi jika dibandingkan dengan representasi padat Vanilla AE. Selanjutnya, bagaimana perbandingan metrik FID (Fréchet Inception Distance) antarmodel generatif: VAE, Vanilla GAN (berbasis MLP), DCGAN, dan DDPM saat menyintesis sampel citra baru dari prior derau acak 𝑧~𝒩(0, 𝐼)?

- **Q2: Analisis Blurriness vs Sharp Edges dan Representasi Fitur:** Mengapa perumusan pixel-wise reconstruction loss (MSE / BCE) pada Vanilla AE, Sparse AE, dan VAE menghasilkan visual rekonstruksi yang cenderung buram (blurry), sedangkan adversarial loss pada Vanilla GAN dan DCGAN menghasilkan kontur dan tekstur citra yang tajam? Jelaskan fenomena ini berdasarkan perbedaan estimasi ekspektasi kondisional 𝔼[𝑥|𝑧] versus minimisasi divergensi Jensen-Shannon 𝐷JS(𝑃data ∥𝑃𝑔). Uraikan pula perbedaan struktur fitur yang dipelajari pada hidden layer Sparse AE akibat penalti jarang terhadap representasi fitur Vanilla AE.

- **Q3: Stabilitas Training, Dinamika Arsitektur, dan Generative Trilemma:** Bagaimana perbandingan dinamika stabilitas kurva loss selama pelatihan antara Vanilla GAN (arsitektur MLP/Fully-Connected) dengan DCGAN (arsitektur konvolusional penuh)? Analisis penyebab munculnya fenomena vanishing gradient atau mode collapse pada Vanilla GAN serta prinsip arsitektur DCGAN yang memitigasi kendala tersebut. Bandingkan pula stabilitas optimasi minimax game pada GAN dengan supervised noise prediction objektif pada DDPM. Evaluasi trade-off Generative Trilemma (Sample Quality, Sampling Speed, Mode Coverage) dengan membandingkan latensi inferensi per batch (64 sampel citra) antara generator satu langkah (single-pass pada Vanilla GAN, DCGAN, dan VAE) terhadap multi-step iterative denoising pada DDPM.

- **Q4: Struktur Latent Space dan Eksperimen Interpolasi:** Bagaimana karakteristik transisi visual saat dilakukan linear latent interpolation 𝑧(𝛼) = (1 −𝛼)𝑧𝐴 + 𝛼𝑧𝐵 dengan 𝛼∈[0.0,0.2,0.4,0.6,0.8,1.0] pada VAE, DCGAN, Vanilla GAN, Sparse AE, dan Vanilla AE? Jelaskan penyebab terbentuknya artefak visual rusak pada model berbasis Autoencoder deterministik (Vanilla AE dan Sparse AE) ketika vektor laten melintasi wilayah kosong (latent space voids) jika dibandingkan dengan manifold laten kontinu yang teratur pada VAE.

- **Q5: Pembuktian Ketidakmampuan Bobot dan Fitur Laten MLP Classifier untuk Image Generation:** Berdasarkan hasil eksperimen pada Bagian 2, jelaskan alasan matematis dan empiris mengapa matriks bobot 𝑊<sup>(𝑙)</sup> serta representasi fitur laten 𝑧MLP dari MLP Classifier tidak dapat langsung digunakan untuk merekonstruksi maupun men-generate citra baru (baik melalui proyeksi balik transpos bobot 𝑊<sup>𝑇</sup> , dekoder terpisah pada fitur laten beku, maupun prior sampling). Tinjau fenomena ini dari perspektif pemodelan conditional probability 𝑃(𝑦∣𝑥) (information bottleneck yang membuang variasi intra-kelas), sifat matriks bobot non-ortogonal (𝑊<sup>𝑇</sup> 𝑊≠𝐈), serta ketiadaan regularisasi kerapatan ruang laten 𝑃(𝑧).

- **Q6: Sintesis Rekayasa dan Rekomendasi Industri:** Berdasarkan evaluasi empiris keenam model yang diuji (Vanilla AE, Sparse AE, Vanilla GAN, VAE, DCGAN, DDPM), tentukan arsitektur yang paling tepat untuk: (1) Deteksi anomali tanpa

Halaman 3 dari 6

Deep Learning (D6421) Sesi 07: Generative Models | Tugas Take-Home

supervisi pada citra inspeksi manufaktur (evaluasi antara Vanilla AE dan Sparse AE berdasarkan sensitivitas reconstruction error), (2) Filter avatar interaktif waktu-nyata pada perangkat mobile dengan anggaran latensi inferensi <30 ms (evaluasi antara Vanilla GAN, DCGAN, dan VAE), dan (3) Pembuatan aset visual beresolusi dan berfidelitas tinggi untuk industri kreatif (evaluasi DCGAN versus DDPM). Sertakan argumen teknis berdasarkan metrik empiris dan inductive bias masingmasing model.

## **5. Struktur dan Sistematika Laporan Akhir Mahasiswa**

Laporan akhir disusun dalam format PDF dengan panjang 10 hingga 12 halaman. Mahasiswa diwajibkan menyusun naskah secara tertib mengikuti struktur publikasi ilmiah berikut:

- **Bagian I: Identitas Penulis dan Tautan Repositori GitHub.** Memuat Nama Lengkap, NIM, Kelas, serta TAUTAN AKTIF repositori GitHub yang dapat diakses publik. Repositori wajib memuat skrip modular PyTorch, berkas requirements.txt, README petunjuk eksekusi, serta git commit history bertahap yang mencerminkan kemandirian pengerjaan.

- **Bagian II: Landasan Teori, Desain Arsitektur, dan Formulasi Matematika.** Diagram skematik arsitektur keenam model generatif (Vanilla AE, Sparse AE, Vanilla GAN, VAE, DCGAN, DDPM) serta satu model MLP Classifier pendamping. Tabel konfigurasi parameter (lapisan arsitektur, dimensi laten, fungsi aktivasi, optimizer, learning rate) serta formulasi matematika loss function lengkap untuk masing-masing model.

- **Bagian III: Pembuktian Teoretis dan Empiris Batasan MLP Classifier untuk Image Generation.** Uraian matematis perbedaan pemodelan 𝑃(𝑦∣𝑥) berbanding 𝑃(𝑥) serta sifat ketidakbalikan matriks bobot 𝑊<sup>𝑇</sup> ≠𝑊<sup>−1</sup> . Penyajian hasil eksperimen proyeksi balik transpos bobot, evaluasi frozen latent decoder (𝑧MLP berbanding 𝑧AE), serta hasil uji prior sampling pada ruang laten MLP Classifier.

- **Bagian IV: Tabel Komparasi Metrik Numerik (Kompilasi Kuantitatif).** Tabel perbandingan menyeluruh hasil evaluasi pada Test Set untuk keenam model yang mencakup dimensi laten, training time per epoch, inference latency per batch (64 sampel), Test MSE, Test PSNR, Test SSIM, FID Score untuk sampel sintetis, serta catatan kestabilan training.

- **Bagian V: Kompilasi Lima Visualisasi Komparatif Utama.** Penyajian lima figur visual dengan resolusi tinggi, penomoran resmi, serta analisis visual komparatif antarmodel.

- **Bagian VI: Pembahasan Kritis, Jawaban Pertanyaan Penelitian, dan Rekomendasi Industri.** Diskusi mendalam mengenai temuan empiris, analisis jawaban terperinci atas pertanyaan Q1 sampai Q6, verifikasi teori Generative Trilemma, serta rekomendasi pemilihan model untuk deployment industri nyata.

**Template Tabel Metrik Komparatif (Bagian IV Laporan):**

|**Arsitektur**<br>**Model**|**Dim Laten**|**Training Time /**<br>**Epoch (s)**|**Latency (ms /**<br>**batch 64)**|**Test MSE**|**Test PSNR (dB)**|**FID Score**|**Kestabilan**<br>**Training**|
|---|---|---|---|---|---|---|---|
|Vanilla<br>Autoencoder|32 atau 64|[diisi mhs]|[diisi mhs]|[diisi mhs]|[diisi mhs]|N/A (Latent<br>Voids)|Stabil /<br>Konvergen|
|Sparse<br>Autoencoder|32 atau 64<br>(Bottleneck) /<br>256 (Sparse)|[diisi mhs]|[diisi mhs]|[diisi mhs]|[diisi mhs]|N/A (Latent<br>Voids)|Stabil /<br>Konvergen|
|Variational<br>Autoencoder<br>(VAE)|32 atau 64|[diisi mhs]|[diisi mhs]|[diisi mhs]|[diisi mhs]|[diisi mhs]|Stabil /<br>Konvergen|
|Generative<br>Adversarial<br>Network (GAN)|64 atau 100<br>(Prior Noise)|[diisi mhs]|[diisi mhs]|N/A (Tanpa<br>Encoder)|N/A|[diisi mhs]|Sangat Rentan<br>Osilasi / Mode<br>Collapse|
|Deep<br>Convolutional<br>GAN (DCGAN)|100 (Prior Noise)|[diisi mhs]|[diisi mhs]|N/A (Tanpa<br>Encoder)|N/A|[diisi mhs]|Cukup Stabil<br>(Sensitif<br>Arsitektur)|
|Denoising<br>Diffusion|28x28 (Dimensi<br>Citra)|[diisi mhs]|[diisi mhs]|N/A (Denoising<br>Model)|N/A|[diisi mhs]|Sangat Stabil|

Halaman 4 dari 6

Deep Learning (D6421) Sesi 07: Generative Models | Tugas Take-Home

Probabilistic Models (DDPM)

#### **Rincian Lima Visualisasi Wajib pada Bagian V Laporan:**

- **Figur 1: Grid Rekonstruksi Side-by-Side:** Tampilkan minimal 8 sampel citra dari Test Set. Baris 1: Citra Asli (Ground Truth); Baris 2: Rekonstruksi Vanilla AE; Baris 3: Rekonstruksi Sparse AE; Baris 4: Rekonstruksi VAE. Sajikan analisis perbandingan ketajaman detail kontur, pelestarian fitur lokal akibat regularisasi penalti sparsity, dan efek keburaman (blurriness).

- **Figur 2: Grid Random Generative Sampling:** Sajikan minimal 16 sampel citra hasil sintesis acak berukuran kisi 4x4 untuk masing-masing model generatif: Panel A: Sampel VAE dari prior 𝑧~𝒩(0, 𝐼); Panel B: Sampel Vanilla GAN dari 𝑧~𝒩(0, 𝐼); Panel C: Sampel DCGAN dari 𝑧~𝒩(0, 𝐼); Panel D: Sampel DDPM dari proses reverse denoising 𝑥𝑇~𝒩(0, 𝐼). Evaluasi kualitas tekstur, ketajaman kontur struktural, serta keberagaman ragam mode (mode diversity) antarmodel.

- **Figur 3: Bukti Empiris Batasan Bobot dan Latent Features MLP Classifier:** Sajikan perbandingan visual empat panel: (1) Citra Asli (Ground Truth), (2) Hasil proyeksi balik langsung transpos bobot 𝑊<sup>𝑇</sup> dari vektor kelas 𝑦𝑐 dan fitur laten 𝑧MLP, (3) Hasil rekonstruksi Frozen Latent Decoder dari 𝑧MLP dibandingkan terhadap rekonstruksi Vanilla AE (𝑧AE), serta (4) Hasil random prior sampling 𝑧rand~𝒩(𝜇𝑧,diag(𝜎2𝑧 )) pada ruang laten MLP Classifier.

- **Figur 4: Linear Latent Space Interpolation:** Pilih dua titik sampel laten 𝑧𝐴 dan 𝑧𝐵 dari dua kelas yang kontras. Lakukan interpolasi linier 𝑧(𝛼) = (1 −𝛼)𝑧𝐴 + 𝛼𝑧𝐵 dengan 𝛼∈[0.0,0.2,0.4,0.6,0.8,1.0] . Dekodekan titik-titik tersebut pada Vanilla AE, Sparse AE, VAE, Vanilla GAN, dan DCGAN, lalu bandingkan kontinuitas transisi visual antarmodel serta kemunculan artefak visual.

- **Figur 5: Training Loss Curves dan DDPM Denoising Trajectory:** Plot kurva training loss terhadap epoch untuk keenam model yang dievaluasi (menampilkan kurva loss rekonstruksi, kurva loss Generator dan Discriminator pada Vanilla GAN dan DCGAN, serta loss prediksi derau pada DDPM). Pada panel model difusi, visualisasikan sekuens reverse diffusion langkah demi langkah (𝑥𝑇 →𝑥0) untuk memperlihatkan proses rekonstruksi citra secara bertahap dari derau acak.

## **6. Rubrik Penilaian dan Distribusi Bobot Evaluasi (Total 100%)**

|**Komponen Evaluasi**<br>**Bobot**||**Kriteria Kualitas Utama**|
|---|---|---|
|**Repositori GitHub &**<br>**Reproduktibilitas Kode**|**15%**|Kode PyTorch terstruktur, modular, terdokumentasi rapi,<br>requirements.txt lengkap, penetapan random seed tetap, serta<br>git commit history mencerminkan proses pengerjaan<br>bertahap.|
|**Implementasi Arsitektur &**<br>**Training Pipeline**|**25%**|Keenam arsitektur (Vanilla AE, Sparse AE, GAN, VAE,<br>DCGAN, DDPM) diimplementasikan sesuai formulasi<br>matematika yang benar, loss function tepat, dan proses<br>optimasi mencapai konvergensi tanpa galat numerik.|
|**Eksperimen Pembuktian Batasan**<br>**MLP Classifier**|**15%**|Eksperimen pembuktian batasan bobot (𝑊<sup>𝑇</sup>) dan fitur laten<br>(𝑧MLP) pada MLP Classifier dijalankan secara valid,<br>komparasi terhadap Autoencoder tersaji lengkap, serta<br>analisis teoretis pemodelan𝑃(𝑦∣𝑥)diuraikan secara<br>mendalam.|
|**Pengujian Kuantitatif & Tabel**<br>**Metrik**|**15%**|Tabel komparasi metrik terisi lengkap pada Test Set (MSE,<br>PSNR, SSIM, FID, latency), pengukuran dilakukan secara<br>objektif, serta parameter pengujian sepadan antarmodel.|
|**Kualitas Visualisasi Komparatif (5**<br>**Figur Wajib)**|**15%**|Kelima figur visual disajikan dengan ketajaman tinggi, label<br>terbaca jelas, tata letak rapi, serta menyertakan analisis<br>komparatif visual yang tajam.|

Halaman 5 dari 6

Deep Learning (D6421) Sesi 07: Generative Models | Tugas Take-Home

**Kedalaman Analisis Ilmiah & 15% Laporan**

Pembahasan menyeluruh atas seluruh pertanyaan penelitian Q1 sampai Q6, sintesis fenomena Generative Trilemma, serta rekomendasi deployment industri yang rasional dan bebas plagiarisme.

## **7. Ketentuan Pengumpulan dan Integritas Akademik**

- **Format Berkas Pengumpulan:** Berkas laporan tunggal berformat PDF dengan konvensi penamaan resmi: TUGAS_DL_SESI07_[NIM]_[NAMA_LENGKAP].pdf.

- **Tautan Repositori GitHub:** Dicantumkan pada halaman judul laporan. Repositori wajib berstatus publik agar tim dosen dan asisten laboratorium dapat meninjau kode sumber serta riwayat pengerjaan.

- **Batas Waktu Pengumpulan:** Sesuai dengan yang ditetapkan pada forum di Binusmaya (LMS).

- **Integritas Akademik & Kebijakan Plagiarisme:** Penggunaan AI coding assistant atau LLM **diperkenankan sebatas referensi sintaksis pemrograman dan penelusuran dokumentasi pustaka. Mahasiswa wajib memahami secara mandiri setiap baris kode dan narasi pembahasan yang diajukan** .

#### **PESAN TIM DOSEN PENGAMPU**

Fokuskan pembahasan laporan pada analisis mendalam mengenai hubungan kausalitas matematis antara loss function dengan karakteristik visual masing-masing model. Tunjukkan penguasaan konsep mengenai alasan mengapa model berperilaku demikian, di samping melaporkan angka metrik empiris yang diperoleh.

Halaman 6 dari 6
