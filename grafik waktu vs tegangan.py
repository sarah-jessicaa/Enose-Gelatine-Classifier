import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# =====================================================================
# GRAFIK RAW DATA DAN Z-SCORE
# =====================================================================

file_path = 'Resume Gelatin.xlsx'
sheet_names = ['Resume Babi', 'Resume Sapi', 'Resume Ikan']

nama_sensor = ['MQ-6', 'MQ-135', 'TGS-822', 'MQ-3', 'MQ-136', 'MQ-137', 'MS-1100', 'MQ-4']
warna_sensor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

for sheet in sheet_names:
    print(f"Memproses grafik untuk: {sheet}...")
    
    # 1. Load data mentah (Ambil sampel pertama: 8 kolom awal)
    df = pd.read_excel(file_path, sheet_name=sheet, header=1)
    sample_data = df.iloc[:, 0:8].copy()
    
    # 2. Bersihkan nilai 0 menjadi NaN, hapus baris kosong, dan potong tepat 120 detik (Hard-cut)
    data_mentah = sample_data.replace(0, np.nan).dropna().values[:120, :]
    
    # 3. Terapkan Standardisasi Z-Score
    scaler = StandardScaler()
    data_zscore = scaler.fit_transform(data_mentah)
    
    # 4. Proses Plotting Berdampingan
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Grafik Kiri: Raw Data ---
    for i in range(8):
        axes[0].plot(data_mentah[:, i], color=warna_sensor[i], label=nama_sensor[i])
    
    # Kustomisasi Judul: "Raw Data", Font 14, Bold
    axes[0].set_title(f'Raw Data - {sheet}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Waktu (detik)')
    axes[0].set_ylabel('Amplitudo (ADC)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # --- Grafik Kanan: Standardisasi Z-Score ---
    for i in range(8):
        axes[1].plot(data_zscore[:, i], color=warna_sensor[i], label=nama_sensor[i])
        
    # Kustomisasi Judul: Font 14, Bold
    axes[1].set_title(f'Standardisasi Z-Score - {sheet}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Waktu (detik)')
    axes[1].set_ylabel('Amplitudo Relatif (Z-Score)')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # Legend di luar bingkai grafik kanan
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', borderaxespad=0.)
    
    # 5. Simpan Gambar
    plt.tight_layout()
    nama_gambar = f"Grafik_Bab4_{sheet}.jpg"
    plt.savefig(nama_gambar, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Berhasil menyimpan: {nama_gambar}")

print("\n=== SELESAI! Silakan periksa 3 gambar grafik terbarumu. ===")