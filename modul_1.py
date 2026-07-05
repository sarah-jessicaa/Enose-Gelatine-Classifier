import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# =====================================================================
# MODUL 1 MESIN PEMBERSIH & SLIDING WINDOW E-NOSE
# =====================================================================

def load_clean_and_split_samples(file_path, sheet_names, num_samples=8, num_sensors=8):
    print(f"[*] Membaca file: {file_path}")
    label_map = {'Resume Babi': 0, 'Resume Sapi': 1, 'Resume Ikan': 2}
    semua_sampel, semua_label = [], []
    
    for sheet in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet, header=1)
        label_hewan = label_map[sheet]
        
        for i in range(num_samples):
            # 1. Isolasi 8 kolom per konsentrasi (1-8%)
            start_col = i * num_sensors
            end_col = start_col + num_sensors
            sample_data = df.iloc[:, start_col:end_col].copy()
            
            # 2. Ganti 0 jadi NaN, lalu hapus baris yang ada NaN-nya
            sample_data = sample_data.replace(0, np.nan).dropna().values
            
            # 3. HARD-CUT 120 BARIS: 6 Menit = 360 Detik : 3 = 120
            data_120_baris = sample_data[:120, :]
            
            # 4. KOREKSI BASELINE (REVISI): Ambil HANYA 1 baris pertama (detik ke-0)
            x_base = data_120_baris[0, :].copy() 
            x_base[x_base == 0] = 1e-10 
            
            # 5. Rumus Fraksional
            data_corrected = (data_120_baris - x_base) / x_base
            
            semua_sampel.append(data_corrected)
            semua_label.append(label_hewan)

    # 6. Memecah 24 sampel fisik ke Latih dan Uji SEBELUM di-Sliding Window
    X_train_samp, X_test_samp, y_train_samp, y_test_samp = train_test_split(
        semua_sampel, semua_label, test_size=0.2, stratify=semua_label, random_state=42
    )
    
    print(f"[*] Total Sampel Latih (Fisik): {len(X_train_samp)}")
    print(f"[*] Total Sampel Uji (Fisik): {len(X_test_samp)}")
    return X_train_samp, X_test_samp, y_train_samp, y_test_samp

def generate_sliding_window(samples, labels, window_size, stride=1):
    X, y = [], []
    for i, data in enumerate(samples):
        label = labels[i]
        for start in range(0, len(data) - window_size + 1, stride):
            window = data[start : start + window_size, :]
            X.append(window)
            y.append(label)
    return np.array(X), np.array(y)

def apply_global_zscore(X_train, X_test):
    print("[*] Melakukan Z-Score Normalization...")
    num_train, win_size, num_features = X_train.shape
    num_test = X_test.shape[0]
    
    X_train_2d = X_train.reshape(-1, num_features)
    X_test_2d = X_test.reshape(-1, num_features)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_2d) 
    X_test_scaled = scaler.transform(X_test_2d)       
    
    X_train_3D = X_train_scaled.reshape(num_train, win_size, num_features)
    X_test_3D = X_test_scaled.reshape(num_test, win_size, num_features)
    return X_train_3D, X_test_3D

def eksekusi_modul_1(file_path, window_size, stride=1):
    print(f"\n=== [START] MODUL 1 (Window={window_size}) ===")
    sheet_names = ['Resume Babi', 'Resume Sapi', 'Resume Ikan']
    
    X_tr_samp, X_te_samp, y_tr_samp, y_te_samp = load_clean_and_split_samples(file_path, sheet_names)
    X_train_raw, y_train = generate_sliding_window(X_tr_samp, y_tr_samp, window_size, stride)
    X_test_raw, y_test = generate_sliding_window(X_te_samp, y_te_samp, window_size, stride)
    X_train, X_test = apply_global_zscore(X_train_raw, X_test_raw)
    
    print(f"[*] HASIL AKHIR MATRIKS 3D: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print("=== [SELESAI] ===")
    return X_train, y_train, X_test, y_test

# =====================================================================
# SAKLAR UTAMA (MAIN EXECUTION)
# =====================================================================
if __name__ == "__main__":
    # GANTI SESUAI NAMA FILE
    nama_file_excel = 'Resume Gelatin.xlsx' 
    
    if not os.path.exists(nama_file_excel):
        print(f"[ERROR] File '{nama_file_excel}' tidak ditemukan di folder:")
        print(os.getcwd())
    else:
        # Menjalankan mesin dengan ukuran window uji coba = 90
        X_train, y_train, X_test, y_test = eksekusi_modul_1(nama_file_excel, window_size=90)