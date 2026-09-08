# Klasifikasi Penyakit Buah Kakao dengan ResNet50-CBAM

Ringkasan notebook `training-resnet50-cbam-buat-seminar-nasional-2026_FIX.ipynb` — eksperimen klasifikasi citra kakao menggunakan transfer learning ResNet50, dengan dan tanpa modul attention CBAM (Convolutional Block Attention Module), untuk keperluan seminar nasional.

## 1. Ringkasan Singkat

Notebook ini membangun dan membandingkan **dua model klasifikasi gambar**:

1. **ResNet50-CBAM** — ResNet50 (pretrained ImageNet) yang disisipi modul CBAM setelah setiap residual stage.
2. **ResNet50** — ResNet50 (pretrained ImageNet) standar tanpa CBAM, sebagai model pembanding/baseline.

Keduanya dilatih untuk mengklasifikasikan gambar buah kakao ke dalam 4 kelas kondisi/penyakit, lalu dievaluasi dan dibandingkan performanya.

## 2. Dataset

- Sumber: `ahmadwafi/dataset-penelitian-kakao-2026` (via Kaggle, path `/kaggle/input/datasets/...`).
- Format gambar: `.jpg` / `.jpeg`, di-resize ke **224×224**.
- **4 kelas**:
  - `Busuk Buah` (label 0)
  - `Helopeltis` (label 1)
  - `Kakao Sehat` (label 2)
  - `Penggerek Buah Kakao` (label 3)
- Pembagian data: **80% train / 10% validation / 10% test**, dengan stratifikasi berdasarkan label (`train_test_split` dua tahap).
- Preprocessing gambar: decode JPEG → resize → cast float32 → `resnet50.preprocess_input` (normalisasi khusus ResNet50) → label di-one-hot encode.
- Dataset dibungkus sebagai `tf.data.Dataset` (shuffle untuk train, batch size 32, prefetch).
- Semua logika dataset dienkapsulasi dalam class `CreateCocoaDataset` (load, hitung jumlah gambar per kelas, split, preprocessing, pembuatan `tf.data.Dataset`).

## 3. Arsitektur Model

### 3.1. Modul CBAM (`ConvolutionalBlockAttentionModule`)
Custom Keras layer yang mengimplementasikan dua sub-modul attention secara berurutan:
- **Channel Attention**: global average pooling + global max pooling → shared MLP (Dense-ReLU-Dense) → dijumlahkan → sigmoid → dikalikan (`Multiply`) ke feature map.
- **Spatial Attention**: rata-rata & max pooling sepanjang axis channel → concat → Conv2D (kernel 7×7, sigmoid) → dikalikan ke feature map hasil channel attention.
- Parameter dapat diatur: `reduction_ratio=16`, `spatial_kernel_size=7`.

### 3.2. ResNet50-CBAM (`ResNet50CBAM`)
- Backbone ResNet50 (`include_top=False`, `weights="imagenet"`) dipecah menjadi 4 submodel berdasarkan output tiap residual stage: `conv2_block3_out`, `conv3_block4_out`, `conv4_block6_out`, `conv5_block3_out`.
- Modul CBAM disisipkan **setelah setiap stage** (conv2 → CBAM → conv3 → CBAM → conv4 → CBAM → conv5 → CBAM).
- Head klasifikasi: `GlobalAveragePooling2D` → `Dense(256, relu)` → `Dropout(0.5)` → `Dense(4, softmax)`.
- Backbone ResNet50 di-**freeze** (non-trainable) — hanya CBAM dan head yang dilatih.

### 3.3. ResNet50 (baseline, `ResNet50` class)
- Backbone ResNet50 (frozen) langsung ke `GlobalAveragePooling2D` → `Dense(256, relu)` → `Dropout(0.5)` → `Dense(4, softmax)`, tanpa CBAM.

## 4. Konfigurasi Training

Kedua model menggunakan setup training yang sama:
- Optimizer: `Adam(learning_rate=1e-4)`
- Loss: `categorical_crossentropy`
- Metric: `accuracy`
- Epochs: 30
- Callbacks:
  - `ModelCheckpoint` (simpan model terbaik berdasar `val_accuracy`)
  - `EarlyStopping` (patience=5, restore best weights)
  - `ReduceLROnPlateau` (monitor `val_loss`, factor=0.5, patience=2, min_lr=1e-7)

## 5. Evaluasi

Untuk masing-masing model, notebook menghasilkan:
- Plot kurva **akurasi & loss** (train vs validation) per epoch.
- Prediksi pada **test set**, dikonversi dari probabilitas ke label kelas.
- **Confusion matrix** (heatmap) dengan label 4 kelas.
- **Accuracy score** dan **classification report** (precision, recall, f1-score per kelas).

## 6. Struktur Notebook

| Bagian | Isi |
|---|---|
| 1. Import Library | TensorFlow/Keras, sklearn, numpy, pandas, matplotlib, seaborn |
| 2. Load Dataset | Path dataset & konfigurasi (image size, batch size, seed) |
| 3. Class `CreateCocoaDataset` | Loading, penghitungan kelas, splitting, preprocessing pipeline |
| 4. Training Tahap 1 — ResNet50-CBAM | Definisi CBAM, definisi model, freeze backbone, compile, callback, training, plot, testing |
| 5. Training Tahap 2 — ResNet50 | Definisi model baseline, freeze backbone, compile, callback, training, testing |