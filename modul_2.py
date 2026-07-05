import tensorflow as tf
import os
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import GRU, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import gc 
from modul_1 import eksekusi_modul_1 

# =====================================================================
# KUNCI REPRODUCIBILITY (Agar hasil tetap konsisten)
# =====================================================================
seed_angka = 42 
os.environ['PYTHONHASHSEED'] = str(seed_angka)
random.seed(seed_angka)
np.random.seed(seed_angka)
tf.random.set_seed(seed_angka)

# =====================================================================
# 1. PARAMETER 
# =====================================================================
nama_file_excel = 'Resume Gelatin.xlsx'
ukuran_jendela = 90
tingkat_dropout = 0.5
maksimal_epoch = 200 # set tinggi, agar Early Stopping yang memutuskan kapan berhenti

print("="*70)
print(f"[INFO] MEMULAI 1D-CNN vs GRU (Window={ukuran_jendela}, DO={tingkat_dropout})")
print("="*70)

# Ekstrak Data HANYA SEKALI untuk memastikan CNN dan GRU memakan data yang 100% SAMA
X_train, y_train, X_test, y_test = eksekusi_modul_1(file_path=nama_file_excel, window_size=ukuran_jendela)
bentuk_data = (ukuran_jendela, 8) 

# =====================================================================
# 2. DEFINISI ARSITEKTUR
# =====================================================================
def buat_model_cnn(input_shape, dropout_rate):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def buat_model_gru(input_shape, dropout_rate):
    model = Sequential([
        GRU(units=32, return_sequences=False, input_shape=input_shape),
        LayerNormalization(),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# =====================================================================
# 3. EARLY STOPPING
# =====================================================================
rem_otomatis = EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True, 
    verbose=1
)

# =====================================================================
# 4. PROCESS
# =====================================================================
model_dict = {
    '1D-CNN': buat_model_cnn(bentuk_data, tingkat_dropout),
    'GRU': buat_model_gru(bentuk_data, tingkat_dropout)
}

# --- PENGATURAN UNTUK MEMPERBESAR SELURUH TEKS GRAFIK ---
plt.rcParams.update({
    'font.size': 14,             # Ukuran font dasar
    'axes.labelsize': 16,        # Ukuran font label sumbu X dan Y
    'xtick.labelsize': 14,       # Ukuran angka di sumbu X
    'ytick.labelsize': 14,       # Ukuran angka di sumbu Y
    'axes.titlesize': 18,        # Ukuran judul grafik
    'legend.fontsize': 14        # Ukuran teks di kotak legenda
})

for nama_model, model in model_dict.items():
    print(f"\n>>> [PROSES] Melatih Model {nama_model} dengan Early Stopping... <<<")
    
    # Proses Belajar
    history = model.fit(
        X_train, y_train,
        epochs=maksimal_epoch,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[rem_otomatis],
        verbose=0 
    )
    
    epoch_berhenti = len(history.history['loss'])
    print(f"[*] {nama_model} berhenti belajar di Epoch ke-{epoch_berhenti}.")
    print(f"[*] Memori dikembalikan ke titik epoch terbaik secara otomatis.")
    
    # =====================================================================
    # 5. CETAK HASIL
    # =====================================================================
    prediksi_prob = model.predict(X_test, verbose=0)
    prediksi_kelas = np.argmax(prediksi_prob, axis=1)
    
    # --- A. Confusion Matrix ---
    cm = confusion_matrix(y_test, prediksi_kelas)
    plt.figure(figsize=(8, 6))
    warna = 'Blues' if nama_model == '1D-CNN' else 'Reds'
    
    # annot_kws={"size": 18} digunakan untuk memperbesar angka di dalam kotak
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap=warna, 
                     xticklabels=['Babi', 'Sapi', 'Ikan'], 
                     yticklabels=['Babi', 'Sapi', 'Ikan'],
                     annot_kws={"size": 18, "weight": "bold"})
    
    ax.set_ylabel('Actual Class', fontweight='bold')
    ax.set_xlabel('Predicted Class', fontweight='bold')
    plt.title(f'Confusion Matrix {nama_model}', pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"ConfMatrix_{nama_model}.png", dpi=300)
    plt.close()
    
    pencatatan = history.history
    epoch_riil = range(1, len(pencatatan['accuracy']) + 1)
    
    # --- B. Kurva Akurasi ---
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_riil, pencatatan['accuracy'], label='Training', color='blue', linewidth=3.0)
    plt.plot(epoch_riil, pencatatan['val_accuracy'], label='Validation', color='orange', linewidth=3.0)
    plt.title(f'Akurasi {nama_model}', fontweight='bold', pad=15)
    plt.xlabel('Epoch (Putaran Belajar)', fontweight='bold') 
    plt.ylabel('Tingkat Akurasi', fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) 
    plt.tight_layout()
    plt.savefig(f"Akurasi_{nama_model}.png", dpi=300)
    plt.close()
    
    # --- C. Kurva Kerugian / Loss ---
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_riil, pencatatan['loss'], label='Training Loss', color='blue', linewidth=3.0)
    plt.plot(epoch_riil, pencatatan['val_loss'], label='Validation Loss', color='orange', linewidth=3.0)
    plt.title(f'Kerugian {nama_model}', fontweight='bold', pad=15)
    plt.xlabel('Epoch (Putaran Belajar)', fontweight='bold')
    plt.ylabel('Nilai Error (Loss)', fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(f"Loss_{nama_model}.png", dpi=300)
    plt.close()

    # --- D. Laporan Klasifikasi Lengkap ---
    laporan = classification_report(y_test, prediksi_kelas, target_names=['Babi', 'Sapi', 'Ikan'])
    print(f"\n--- LAPORAN KLASIFIKASI {nama_model} ---")
    print(laporan)
    
    # Bersihkan Memori
    tf.keras.backend.clear_session()
    del model
    gc.collect()

print("\n" + "="*70)
print("[INFO] SELESAI!")
print("="*70)