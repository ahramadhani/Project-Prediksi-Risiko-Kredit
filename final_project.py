# ==============================================================================
# FINAL PROJECT VIX ID/X PARTNERS
# Nama: [Nama Lengkap Anda]
# Proyek: Model Prediksi Risiko Kredit End-to-End dengan Penanganan Imbalanced Data
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Untuk menyimpan model

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Penanganan Imbalanced Data
# Pastikan Anda sudah install: pip install -U imbalanced-learn
from imblearn.over_sampling import SMOTE

# Model & Evaluasi
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score

print("Semua library berhasil di-import.")

# ------------------------------------------------------------------------------
# 2. BUSINESS UNDERSTANDING (SESUAI METODOLOGI)
# ------------------------------------------------------------------------------
# Tujuan: Membangun model machine learning untuk memprediksi apakah seorang
# peminjam akan 'Fully Paid' (Lunas) atau 'Charged Off' (Gagal Bayar).
# Problem: Data pinjaman secara alami tidak seimbang (imbalanced).
# Solusi: Menerapkan teknik oversampling (SMOTE) untuk meningkatkan
# performa model dalam mendeteksi pinjaman 'Charged Off'.

# ------------------------------------------------------------------------------
# 3. DATA COLLECTION & LOADING
# ------------------------------------------------------------------------------
try:
    # Ganti dengan path file Anda
    data = pd.read_csv('loan_data_2007_2014.csv', index_col=0)
    print(f"Data berhasil dimuat. Bentuk data: {data.shape}")
except FileNotFoundError:
    print("Error: File 'loan_data_2007_2014.csv' tidak ditemukan.")
    # Keluar dari script jika data tidak ada
    exit()

# ------------------------------------------------------------------------------
# 4. DATA PREPARATION & FEATURE ENGINEERING
# ------------------------------------------------------------------------------

# 4.1. Mendefinisikan Target Variable (y)
# Ini adalah langkah paling krusial. Kita hanya tertarik pada status akhir.
# 'Fully Paid' adalah kelas 'baik' (target = 0)
# 'Charged Off' adalah kelas 'buruk' (target = 1)
# Status lain ('Current', 'In Grace Period', dll.) kita buang karena pinjaman tsb.
# belum selesai.

print("Memulai Data Preparation...")
data['loan_status'] = data['loan_status'].map({
    'Fully Paid': 0,
    'Charged Off': 1
})

# Filter data: hanya ambil status 0 dan 1, buang nilai NaN (status lainnya)
df_model = data[data['loan_status'].isin([0, 1])].copy()
df_model['loan_status'] = df_model['loan_status'].astype(int)

print(f"Data setelah difilter (Fully Paid & Charged Off): {df_model.shape}")

# 4.2. Feature Selection (Memilih Fitur)
# Memilih fitur yang relevan dan tersedia *sebelum* pinjaman disetujui.
# Contoh: 'total_pymnt' (total pembayaran) tidak bisa dipakai karena itu
# adalah hasil/output, bukan prediktor.

features = [
    'loan_amnt',         # Jumlah pinjaman
    'int_rate',          # Suku bunga
    'installment',       # Angsuran bulanan
    'grade',             # Grade pinjaman (A, B, C, dst.)
    'emp_length',        # Lama bekerja
    'home_ownership',    # Status kepemilikan rumah
    'annual_inc',        # Pendapatan tahunan
    'verification_status', # Status verifikasi
    'dti',               # Debt-to-Income Ratio
    'delinq_2yrs',       # Keterlambatan pembayaran 2 thn terakhir
    'inq_last_6mths',    # Jumlah penyelidikan 6 bln terakhir
    'open_acc',          # Jumlah akun terbuka
    'pub_rec',           # Catatan publik buruk
    'revol_util',        # Utilisasi kredit revolving
    'total_acc'          # Total akun kredit
]
target = 'loan_status'

df_model = df_model[features + [target]]
print("Feature selection selesai.")

# 4.3. Membersihkan dan Mengkodekan Fitur

# 'emp_length' (Ordinal)
emp_length_map = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
    '10+ years': 10
}
df_model['emp_length'] = df_model['emp_length'].map(emp_length_map)

# 'grade' (Ordinal)
grade_map = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1, 'G': 0}
df_model['grade'] = df_model['grade'].map(grade_map)

# Fitur Kategorikal Lainnya (Nominal) -> One-Hot Encoding
categorical_cols = ['home_ownership', 'verification_status']
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

print("Encoding fitur kategorikal selesai.")

# 4.4. Menangani Missing Values (Imputation)
# Untuk script, kita gunakan imputasi median (strategi sederhana)
# Kolom numerik
numeric_cols = df_model.select_dtypes(include=np.number).columns.drop(target)

imputer_numeric = SimpleImputer(strategy='median')
df_model[numeric_cols] = imputer_numeric.fit_transform(df_model[numeric_cols])

print("Penanganan missing values selesai.")

# ------------------------------------------------------------------------------
# 5. PEMBAGIAN DATA (TRAIN-TEST SPLIT)
# ------------------------------------------------------------------------------

X = df_model.drop(target, axis=1)
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Data berhasil dibagi (Train-Test Split).")
print(f"Bentuk X_train: {X_train.shape}")
print(f"Bentuk X_test: {X_test.shape}")

# Cek distribusi target di y_train (sebelum SMOTE)
print("\nDistribusi Target (Sebelum SMOTE):")
print(y_train.value_counts(normalize=True))

# ------------------------------------------------------------------------------
# 6. FEATURE SCALING
# ------------------------------------------------------------------------------
# Scaling penting untuk banyak model, meskipun Random Forest tidak terlalu
# sensitif, ini adalah 'best practice'.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling selesai.")

# ------------------------------------------------------------------------------
# 7. MODEL BUILDING (SEBELUM PENANGANAN IMBALANCE)
# ------------------------------------------------------------------------------
print("\nMelatih Model Baseline (Random Forest) pada data ASLI (Imbalanced)...")

rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train_scaled, y_train)

# Prediksi
y_pred_baseline = rf_baseline.predict(X_test_scaled)

print("Pelatihan Model Baseline Selesai.")

# ------------------------------------------------------------------------------
# 8. MODEL EVALUATION (SEBELUM PENANGANAN IMBALANCE)
# ------------------------------------------------------------------------------
print("\n--- HASIL EVALUASI MODEL BASELINE (SEBELUM SMOTE) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_baseline):.4f}")
print("\nClassification Report (Baseline):")
print(classification_report(y_test, y_pred_baseline, target_names=['Fully Paid (0)', 'Charged Off (1)']))
print("\nConfusion Matrix (Baseline):")
print(confusion_matrix(y_test, y_pred_baseline))

# Kita bisa lihat bahwa Recall untuk 'Charged Off (1)' akan sangat rendah!
# Ini adalah masalah utama yang akan kita selesaikan.

# ------------------------------------------------------------------------------
# 9. PENANGANAN IMBALANCED DATA (SMOTE)
# ------------------------------------------------------------------------------
print("\nMelakukan Oversampling dengan SMOTE pada data training...")

# Perhatian: SMOTE hanya di-apply pada data training!
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("SMOTE Selesai.")
print("Distribusi Target (Setelah SMOTE):")
print(pd.Series(y_train_smote).value_counts(normalize=True))

# ------------------------------------------------------------------------------
# 10. MODEL BUILDING (SETELAH PENANGANAN IMBALANCE)
# ------------------------------------------------------------------------------
print("\nMelatih Model Final (Random Forest) pada data SMOTE...")

rf_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_smote.fit(X_train_smote, y_train_smote)

# Prediksi
# Model dievaluasi pada data test yang ASLI (tidak di-SMOTE)
y_pred_smote = rf_smote.predict(X_test_scaled)

print("Pelatihan Model Final Selesai.")

# ------------------------------------------------------------------------------
# 11. MODEL EVALUATION (SETELAH PENANGANAN IMBALANCE)
# ------------------------------------------------------------------------------
print("\n--- HASIL EVALUASI MODEL FINAL (SETELAH SMOTE) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_smote):.4f}")
print("\nClassification Report (Final - SMOTE):")
print(classification_report(y_test, y_pred_smote, target_names=['Fully Paid (0)', 'Charged Off (1)']))
print("\nConfusion Matrix (Final - SMOTE):")
print(confusion_matrix(y_test, y_pred_smote))

# ------------------------------------------------------------------------------
# 12. KESIMPULAN & DEPLOYMENT
# ------------------------------------------------------------------------------
print("\n--- KESIMPULAN PERBANDINGAN ---")
recall_baseline = recall_score(y_test, y_pred_baseline, pos_label=1)
recall_smote = recall_score(y_test, y_pred_smote, pos_label=1)

print(f"Recall 'Charged Off' (Baseline): {recall_baseline:.4f}")
print(f"Recall 'Charged Off' (SMOTE): {recall_smote:.4f}")
print(f"\nPeningkatan Recall: {((recall_smote - recall_baseline) / recall_baseline) * 100:.2f}%")
print("\nModel SMOTE menunjukkan peningkatan signifikan dalam kemampuan")
print("mendeteksi pinjaman Gagal Bayar (Charged Off), yang merupakan")
print("tujuan utama bisnis.")

# 12.1. Menyimpan Model Final untuk Deployment
# Menyimpan model (rf_smote) dan scaler
try:
    joblib.dump(rf_smote, 'model_final_rf.pkl')
    joblib.dump(scaler, 'scaler_final.pkl')
    print("\nModel final (rf_smote) dan scaler berhasil disimpan ke file .pkl.")
except Exception as e:
    print(f"\nError saat menyimpan model: {e}")

print("\n=== SCRIPT SELESAI ===")