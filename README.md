# Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas

Tugas Take-Home Sesi 05 (Advanced CNNs and Transfer Learning), mata kuliah Deep Learning, BINUS.
Penulis: Axel Sanjiro Yang (NIM: `2802472400`)

Proyek ini membandingkan tiga paradigma pelatihan pada klasifikasi 10 ras hewan peliharaan
(Oxford-IIIT Pet): pelatihan dari nol, feature extraction, dan progressive fine-tuning pada ResNet-18,
ditambah uji kontrol data augmentation dan pembanding EfficientNet-B0.

## Isi Repositori

| Berkas | Keterangan |
|---|---|
| `GSLC_DL_Sesi_5__Axel_Sanjiro_Yang.ipynb` | Notebook utama (eksperimen B1-B5, C1-C5), sudah dieksekusi beserta luaran sel |
| `requirements.txt` | Dependensi Python |
| `experiment_log.csv` | Ringkasan metrik per kondisi (trainable params, LR, epoch konvergen, durasi, Val Acc, Test Acc, Macro-F1) |
| `history_log.csv` | Log per epoch (train/val loss dan accuracy, learning rate per kelompok parameter) |
| `failure_analysis.csv` | 8 misklasifikasi dengan confidence tertinggi pada test set (model B5) |
| `loss_curves.png`, `augmentation_gap.png`, `confusion_matrix.png`, `failure_analysis.png`, `gradcam_before_after.png` | Visualisasi yang dihasilkan notebook |
| Laporan (`.pdf`) | Laporan analisis bagian C1-C5 |

## Dataset

- Oxford-IIIT Pet, **10 kelas pertama**: Abyssinian, American Bulldog, American Pit Bull Terrier, Basset Hound,
  Beagle, Bengal, Birman, Bombay, Boxer, British Shorthair (5 kucing, 5 anjing).
- Total 1.981 citra, stratified split 70/15/15: **1.386 train / 297 validation / 298 test**.
- Diunduh otomatis lewat `torchvision.datasets.OxfordIIITPet` ke folder `./data` (sekitar 800 MB), tidak perlu autentikasi.
- Notebook memverifikasi ketiga split saling lepas (`assert`), sehingga tidak ada data leakage.

## Kondisi Eksperimen

| Kode | Strategi | Trainable params | Learning rate |
|---|---|---|---|
| B1 | ResNet-18 dari nol (`weights=None`) | 11.181.642 | semua 1e-3 |
| B2 | Feature extraction: backbone dibekukan, hanya `fc` dilatih | 5.130 | fc 1e-3 |
| B3 | Fine-tuning `layer4` + `fc` dengan differential LR | 8.398.858 | layer4 1e-4; fc 1e-3 |
| B4 | Uji kontrol augmentasi pada konfigurasi B3 (tanpa vs. dengan augmentasi) | 8.398.858 | layer4 1e-4; fc 1e-3 |
| B5 | EfficientNet-B0 feature extraction (bonus) | 12.810 | classifier 1e-3 |

Protokol seragam: input 224x224 dengan normalisasi ImageNet, batch size 32, AdamW (weight decay 1e-2),
CrossEntropyLoss, maksimum 30 epoch, early stopping pada validation loss (patience 5, bobot epoch terbaik dipulihkan),
seed 42 (Python, NumPy, PyTorch, cuDNN deterministik). Augmentasi hanya pada train set.

## Hasil Ringkas (seed 42)

| Kode | Epoch konv. / total | Durasi (s) | Val Acc | Test Acc | Test Macro-F1 |
|---|---|---|---|---|---|
| B1 | 13 / 18 | 144,3 | 0,4815 | 0,5168 | 0,4736 |
| B2 | 16 / 21 | 137,1 | 0,9394 | 0,9329 | 0,9333 |
| B3 | 4 / 9 | 58,5 | 0,9394 | 0,9396 | 0,9399 |
| B4 tanpa augmentasi | 4 / 9 | 34,7 | 0,9495 | 0,9396 | 0,9399 |
| B4 dengan augmentasi | 4 / 9 | 60,9 | 0,9394 | 0,9396 | 0,9399 |
| B5 (EfficientNet-B0) | 21 / 26 | 180,1 | 0,9596 | 0,9530 | 0,9531 |

Catatan: test set hanya 298 citra (interval kepercayaan 95% sekitar +/-2,5 poin), sehingga selisih antara
B2, B3, dan B5 tidak signifikan secara statistik. Rincian analisis ada di laporan.

## Cara Menjalankan

### Opsi 1: Google Colab (disarankan, sesuai eksperimen asli)

1. Unggah notebook ke Google Colab.
2. Pilih **Runtime > Change runtime type > T4 GPU**.
3. Pilih **Runtime > Run all**. Paket yang dibutuhkan sudah tersedia di Colab.

### Opsi 2: Lokal

Prasyarat: Python 3.10+ dan, sangat disarankan, GPU NVIDIA dengan CUDA (di CPU pelatihan sangat lambat).

```bash
git clone <https://github.com/axelsanjiro/GSLC-DL-Sesi05>
cd <GSLC-DL-Sesi05>

python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook GSLC_DL_Sesi_5__Axel_Sanjiro_Yang.ipynb
```

Jika memakai GPU, instal build PyTorch yang sesuai versi CUDA Anda terlebih dahulu dengan mengikuti
petunjuk di https://pytorch.org/get-started/locally/.

Menjalankan seluruh sel akan:
1. mengunduh dataset dan membentuk split,
2. melatih B1-B5 (sekitar 10 menit pelatihan di T4, belum termasuk unduhan),
3. menulis ulang `experiment_log.csv` dan `history_log.csv`,
4. menghasilkan seluruh gambar (kurva loss, confusion matrix, failure analysis, Grad-CAM).

## Reproduksibilitas

- Seed tetap 42 untuk Python, NumPy, dan PyTorch; `cudnn.deterministic=True`, `cudnn.benchmark=False`;
  DataLoader memakai generator dan `worker_init_fn` ber-seed.
- Di lingkungan yang sama, B3 dan B4 (dengan augmentasi) menghasilkan metrik yang identik, sebagai pemeriksaan keterulangan.
- Hasil dapat berbeda sedikit pada GPU, versi PyTorch/CUDA, atau jumlah worker yang berbeda.
- Durasi pelatihan bersifat indikatif karena mencakup epoch patience dan pipeline data.

## Keterbatasan

Single seed, hanya 10 dari 37 kelas, pemilihan model terbaik (B5) berdasarkan Test Macro-F1, tidak ada tuning hyperparameter,
dan B5 memakai konfigurasi feature extraction (bukan B3). Pembahasan lengkap ada di bagian C4 laporan.

## Referensi dan Atribusi

- Parkhi et al. (2012). *Cats and Dogs*. CVPR (dataset Oxford-IIIT Pet).
- He et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- Tan & Le (2019). *EfficientNet*. ICML.
- Selvaraju et al. (2017). *Grad-CAM*. ICCV.
- Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization (AdamW)*. ICLR.
- Bobot pretrained dari `torchvision` (ResNet18 `IMAGENET1K_V1`, EfficientNet-B0 `IMAGENET1K_V1`).
- Kerangka kode awal dari soal Tugas Take-Home Sesi 05.

## Pernyataan Penggunaan AI

Lihat lampiran "Pernyataan Penggunaan AI" pada laporan.
