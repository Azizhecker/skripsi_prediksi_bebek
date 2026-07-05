import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("Membaca dataset...")
# 1. Memuat Dataset
try:
    df = pd.read_csv('datapeternakbebak.csv')
    print(f"Dataset berhasil dimuat! Total data: {len(df)} baris.")
except FileNotFoundError:
    print("Error: File 'dataset_bebek_2023_2025.csv' tidak ditemukan. Pastikan file berada di folder yang sama.")
    exit()

# 2. Menentukan Fitur (X) dan Target (y)
X = df[['suhu', 'kelembapan', 'curah_hujan', 'usia_bebek']]
y = df['produksi_telur']

# 3. Membagi Data (80% Latih, 20% Uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Melatih Model Random Forest
print("Melatih model Random Forest...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluasi Model (Sesuai Bab 1.4 Skripsi Anda)
prediksi_test = model.predict(X_test)
mae = mean_absolute_error(y_test, prediksi_test)
rmse = np.sqrt(mean_squared_error(y_test, prediksi_test))
r2 = r2_score(y_test, prediksi_test)

print("\n=== HASIL EVALUASI MODEL ===")
print(f"Mean Absolute Error (MAE) : {mae:.2f}")
print(f"Root Mean Squared Error (RMSE) : {rmse:.2f}")
print(f"R-squared (R2) : {r2:.2f}")
print("============================\n")

# 6. Menyimpan Model
joblib.dump(model, 'model_rf.pkl')
print("Model berhasil disimpan sebagai 'model_rf.pkl' dan siap digunakan di website!")