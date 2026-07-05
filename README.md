# Electronic Nose Gelatin Classifier - 1D-CNN vs GRU

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Deskripsi

Program komputer untuk autentikasi sumber gelatin (babi, sapi, ikan) menggunakan instrumen **Electronic Nose (E-Nose)** berbasis sensor **Metal-Oxide Semiconductor (MOS)**. Program ini membandingkan kinerja dua arsitektur *Deep Learning*: **1-Dimensional Convolutional Neural Network (1D-CNN)** dan **Gated Recurrent Unit (GRU)** dalam mengklasifikasikan pola kinetika gas.

## 🎯 Fitur Utama

- ✅ Pra-pemrosesan data kinetika gas otomatis (koreksi baseline, Z-Score normalization)
- ✅ Augmentasi data menggunakan Sliding Window
- ✅ Pelatihan dan evaluasi model 1D-CNN dan GRU
- ✅ Visualisasi komparasi Raw Data vs Standardisasi Z-Score
- ✅ Generasi Confusion Matrix dan kurva pembelajaran
- ✅ Analisis metrik evaluasi lengkap (Accuracy, Precision, Recall, F1-Score)
- ✅ Early Stopping untuk mencegah overfitting

## 📁 Struktur Repository
