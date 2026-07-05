# Electronic Nose Gelatin Classifier - 1D-CNN vs GRU

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Deskripsi

Program komputer untuk autentikasi sumber gelatin (babi, sapi, ikan) menggunakan instrumen **Electronic Nose (E-Nose)** berbasis sensor **Metal-Oxide Semiconductor (MOS)**. Program ini membandingkan kinerja dua arsitektur *Deep Learning*: **1-Dimensional Convolutional Neural Network (1D-CNN)** dan **Gated Recurrent Unit (GRU)** dalam mengklasifikasikan pola kinetika gas.

Instrumen E-Nose menggunakan larik 8 sensor MOS untuk menangkap pola emisi *Volatile Organic Compounds* (VOCs) saat gelatin dipanaskan. Program ini mengatasi masalah **sensitivitas silang** (*cross-sensitivity*) yang menyebabkan tumpang tindih pola sinyal antara gelatin sapi dan ikan.

## 🎯 Fitur Utama

- ✅ **Pra-pemrosesan data otomatis**: Koreksi baseline fraksional, Z-Score normalization
- ✅ **Augmentasi data**: Sliding Window untuk memperkaya variasi data
- ✅ **Dual arsitektur**: Pelatihan dan evaluasi model 1D-CNN dan GRU
- ✅ **Visualisasi komparasi**: Raw Data vs Standardisasi Z-Score
- ✅ **Evaluasi lengkap**: Confusion Matrix, kurva pembelajaran, metrik evaluasi
- ✅ **Early Stopping**: Mencegah overfitting secara otomatis
- ✅ **Lightweight model**: Optimasi untuk implementasi mikrokontroler

## 📁 Struktur Repository
Enose-Gelatine-Classifier/
│
├── modul_1.py # Modul pra-pemrosesan data
├── modul_2.py # Modul pelatihan & evaluasi model
├── grafik_waktu_vs_tegangan.py # Visualisasi Raw Data vs Z-Score
├── Resume Gelatin.xlsx # Dataset (tidak termasuk di repo)
│
├── README.md # Dokumentasi utama
├── LICENSE # Lisensi MIT
├── requirements.txt # Dependensi Python
│
└── outputs/ # Hasil eksekusi (auto-generated)
├── ConfMatrix_1D-CNN.png
├── ConfMatrix_GRU.png
├── Kurva_1D-CNN.png
├── Kurva_GRU.png
├── Grafik_Bab4_Resume Babi.jpg
├── Grafik_Bab4_Resume Sapi.jpg
└── Grafik_Bab4_Resume Ikan.jpg ```

## 📦 Instalasi

### 1. Clone Repository

bash
git clone https://github.com/sarah-jessicaa/Enose-Gelatine-Classifier.git
cd Enose-Gelatine-Classifier

### 2. Install Dependensi
pip install -r requirements.txt
atau install manual
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn openpyxl

### 3. Persiapan Dataset
Siapkan file Excel Resume Gelatin.xlsx dengan 3 sheet:
Resume Babi
Resume Sapi
Resume Ikan
Letakkan file di direktori utama yang sama dengan skrip Python
Catatan: Dataset tidak termasuk dalam repository karena bersifat sensitif.

🚀 Cara Penggunaan
A. Visualisasi Data (Opsional - Rekomendasi)
Jalankan skrip ini terlebih dahulu untuk melihat perbandingan data mentah dan hasil normalisasi:
python grafik_waktu_vs_tegangan.py
Output: 3 file gambar (.jpg) yang menampilkan:
Raw Data (ADC) vs Standardisasi Z-Score untuk setiap jenis gelatin
Grafik side-by-side dengan 8 kanal sensor

B. Pelatihan dan Evaluasi Model
Jalankan program utama:
python modul_2.py

Proses yang akan berjalan:
Pra-pemrosesan data (modul_1.py dipanggil otomatis)
Pelatihan model 1D-CNN
Pelatihan model GRU
Evaluasi dan generasi visualisasi

Output yang dihasilkan:
ConfMatrix_1D-CNN.png - Confusion Matrix model 1D-CNN
ConfMatrix_GRU.png - Confusion Matrix model GRU
Kurva_1D-CNN.png - Kurva akurasi & loss 1D-CNN
Kurva_GRU.png - Kurva akurasi & loss GRU
Laporan klasifikasi di terminal

C. Menggunakan Visual Studio Code
Buka folder project di VS Code
Pastikan ekstensi Python dan Jupyter terinstall
Buka modul_2.py
Klik tombol Run Python File (▶) atau tekan F5

📊 Parameter Konfigurasi
Parameter dapat disesuaikan di modul_2.py:
