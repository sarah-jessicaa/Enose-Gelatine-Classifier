# Klasifikasi Sumber Gelatin (Babi/Sapi/Ikan) dengan E-Nose

Proyek ini mengklasifikasikan sumber gelatin (Babi, Sapi, Ikan) berdasarkan data sensor E-Nose (electronic nose), menggunakan dan membandingkan dua arsitektur deep learning: **1D-CNN** dan **GRU**.

## Struktur Proyek

```
.
├── Resume Gelatin.xlsx            # Dataset mentah (data sensor, 3 sheet: Babi/Sapi/Ikan)
├── grafik waktu vs tegangan.py    # Visualisasi Raw Data vs Z-Score (untuk Bab 4 skripsi)
├── modul_1.py                     # Preprocessing, sliding window, normalisasi (Z-score)
├── modul_2.py                     # Training & evaluasi model 1D-CNN vs GRU
├── index.html                     # Demo klasifikasi interaktif (GitHub Pages)
└── requirements.txt
```

## Alur Kerja

### 0. Visualisasi Awal: `grafik waktu vs tegangan.py`
Skrip independen, tidak bergantung pada `modul_1.py` maupun `modul_2.py`. Mengambil satu sampel mentah per jenis gelatin, membandingkan sinyal **Raw Data** dengan hasil **Standardisasi Z-Score** secara berdampingan, dan menyimpannya sebagai `Grafik_Bab4_<nama sheet>.jpg`. Dijalankan lebih dulu untuk keperluan eksplorasi data dan ilustrasi pada laporan, sebelum masuk ke tahap pelatihan model.

### Modul 1: Preprocessing & Sliding Window
- Membaca 3 sheet Excel (`Resume Babi`, `Resume Sapi`, `Resume Ikan`), masing-masing berisi 8 sampel (konsentrasi 1–8%) x 8 sensor.
- Membersihkan data: nilai 0 dianggap invalid → diubah ke NaN → baris dihapus.
- Memotong setiap sampel menjadi 120 baris (6 menit, interval 3 detik).
- Koreksi baseline fraksional menggunakan baris pertama (detik ke-0) sebagai acuan.
- Split data (80/20, stratified) di level sampel fisik **sebelum** sliding window untuk mencegah data leakage.
- Sliding window diterapkan untuk memperbanyak data latih/uji.
- Normalisasi Z-score global (scaler di-fit hanya pada data latih).

### Modul 2: Training & Evaluasi
- Melatih dan membandingkan model **1D-CNN** dan **GRU** dengan data yang identik.
- Menggunakan Early Stopping (monitor `val_loss`, patience=15) untuk mencegah overfitting.
- Output otomatis: confusion matrix, kurva akurasi, kurva loss (disimpan sebagai `.png`), dan classification report.

### Demo Web: `index.html`
Model GRU hasil pelatihan diekspor ke TensorFlow.js dan disematkan langsung ke halaman ini, jadi seluruh inferensi berjalan di peramban tanpa server. Diaktifkan lewat GitHub Pages agar bisa dicoba siapa saja.

## Cara Menjalankan

```bash
pip install -r requirements.txt
python "grafik waktu vs tegangan.py"   # opsional, untuk grafik eksplorasi
python modul_2.py
```

`modul_2.py` akan otomatis memanggil `modul_1.py` untuk mengambil data.

## Parameter Utama
| Parameter | Nilai |
|---|---|
| Window size | 90 |
| Dropout rate | 0.5 |
| Max epoch | 200 (dengan Early Stopping) |
| Random seed | 42 |

## Catatan
Pastikan nama file dataset di folder **persis sama** dengan yang dipanggil di `modul_1.py` dan `modul_2.py` (variabel `nama_file_excel`), termasuk spasi/underscore, agar tidak terjadi `FileNotFoundError`.
