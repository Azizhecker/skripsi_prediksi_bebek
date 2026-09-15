import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# KONFIGURASI
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset_bebek_skripsi_508.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model_rf.pkl"
)

RESULT_PATH = os.path.join(
    BASE_DIR,
    "hasil_model.json"
)

print("=" * 60)
print("PELATIHAN RANDOM FOREST - PREDIKSI TELUR BEBEK")
print("=" * 60)


# =========================================================
# 1. MEMBACA DATASET
# =========================================================

if not os.path.exists(DATASET_PATH):
    print("\nERROR:")
    print("Dataset tidak ditemukan:")
    print(DATASET_PATH)
    print("\nPastikan dataset_bebek_skripsi_508.csv berada")
    print("di folder yang sama dengan train_model.py")
    exit()

df = pd.read_csv(DATASET_PATH)

print("\nDataset berhasil dibaca.")
print("Jumlah data :", len(df))
print("Jumlah kolom:", len(df.columns))


# =========================================================
# 2. CEK KOLOM
# =========================================================

kolom_wajib = [
    "tanggal",
    "jenis_bebek",
    "populasi_ratu",
    "populasi_air",
    "populasi_total",
    "kepadatan_kandang",
    "pakan_kg",
    "kondisi_cuaca",
    "suhu",
    "kelembapan",
    "curah_hujan",
    "usia_ratu_bulan",
    "usia_air_bulan",
    "produksi_telur"
]

kolom_hilang = [
    kolom for kolom in kolom_wajib
    if kolom not in df.columns
]

if kolom_hilang:
    print("\nERROR! Kolom berikut tidak ditemukan:")
    for kolom in kolom_hilang:
        print("-", kolom)

    print("\nKolom dataset yang ditemukan:")
    print(df.columns.tolist())
    exit()


# =========================================================
# 3. PEMBERSIHAN DATA
# =========================================================

df["tanggal"] = pd.to_datetime(
    df["tanggal"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "suhu",
        "kelembapan",
        "curah_hujan",
        "usia_ratu_bulan",
        "usia_air_bulan",
        "produksi_telur"
    ]
)

df["kondisi_cuaca"] = (
    df["kondisi_cuaca"]
    .astype(str)
    .str.strip()
)


# =========================================================
# 4. FITUR DAN TARGET
# =========================================================

fitur_numerik = [
    "populasi_ratu",
    "populasi_air",
    "populasi_total",
    "kepadatan_kandang",
    "pakan_kg",
    "suhu",
    "kelembapan",
    "curah_hujan",
    "usia_ratu_bulan",
    "usia_air_bulan"
]

fitur_kategori = [
    "kondisi_cuaca"
]

fitur = fitur_numerik + fitur_kategori

target = "produksi_telur"

X = df[fitur].copy()
y = df[target].copy()


# =========================================================
# 5. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nPembagian data:")
print("Data training:", len(X_train))
print("Data testing :", len(X_test))


# =========================================================
# 6. PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "kategori",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            fitur_kategori
        )
    ],
    remainder="passthrough"
)


# =========================================================
# 7. RANDOM FOREST
# =========================================================

rf = RandomForestRegressor(
    random_state=42
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf)
    ]
)


# =========================================================
# 8. GRID SEARCH
# =========================================================

parameter_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

print("\nMulai GridSearchCV...")
print("Proses mungkin membutuhkan waktu beberapa saat.")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)


# =========================================================
# 9. MODEL TERBAIK
# =========================================================

best_model = grid_search.best_estimator_

print("\nModel terbaik:")
print(grid_search.best_params_)


# =========================================================
# 10. PREDIKSI DATA TESTING
# =========================================================

prediksi = best_model.predict(X_test)

prediksi = np.maximum(
    prediksi,
    0
)


# =========================================================
# 11. EVALUASI
# =========================================================

mae = mean_absolute_error(
    y_test,
    prediksi
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        prediksi
    )
)

r2 = r2_score(
    y_test,
    prediksi
)


# MAPE
y_test_array = np.array(y_test)
prediksi_array = np.array(prediksi)

mask = y_test_array != 0

if mask.sum() > 0:
    mape = np.mean(
        np.abs(
            (
                y_test_array[mask]
                - prediksi_array[mask]
            )
            / y_test_array[mask]
        )
    ) * 100
else:
    mape = 0


print("\n" + "=" * 60)
print("HASIL EVALUASI")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.4f}%")
print(f"R²   : {r2:.4f}")


# =========================================================
# 12. HASIL PREDIKSI DATA TESTING
# =========================================================

hasil_testing = X_test.copy()

hasil_testing["aktual"] = y_test.values
hasil_testing["prediksi"] = np.round(
    prediksi,
    2
)

hasil_testing["error"] = (
    hasil_testing["aktual"]
    - hasil_testing["prediksi"]
)

hasil_testing["absolute_error"] = (
    hasil_testing["error"].abs()
)

hasil_testing["squared_error"] = (
    hasil_testing["error"] ** 2
)

hasil_testing["APE"] = np.where(
    hasil_testing["aktual"] != 0,
    (
        hasil_testing["absolute_error"]
        / hasil_testing["aktual"]
    ) * 100,
    0
)

hasil_testing = hasil_testing.reset_index(
    drop=True
)


# =========================================================
# 13. FEATURE IMPORTANCE
# =========================================================

preprocessor_fitted = best_model.named_steps[
    "preprocessor"
]

rf_fitted = best_model.named_steps[
    "model"
]

feature_names = (
    preprocessor_fitted
    .get_feature_names_out()
)

importance = rf_fitted.feature_importances_

feature_importance = pd.DataFrame({
    "fitur": feature_names,
    "importance": importance
})

feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


# =========================================================
# 14. SIMPAN MODEL
# =========================================================

model_package = {
    "model": best_model,
    "fitur": fitur,
    "fitur_numerik": fitur_numerik,
    "fitur_kategori": fitur_kategori,
    "target": target,
    "best_params": grid_search.best_params_,
    "metrics": {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2)
    },
    "feature_importance": feature_importance.to_dict(
        orient="records"
    )
}

joblib.dump(
    model_package,
    MODEL_PATH
)


# =========================================================
# 15. SIMPAN HASIL JSON
# =========================================================

hasil_model = {
    "jumlah_data": int(len(df)),
    "data_training": int(len(X_train)),
    "data_testing": int(len(X_test)),
    "mae": float(mae),
    "rmse": float(rmse),
    "mape": float(mape),
    "r2": float(r2),
    "best_params": grid_search.best_params_
}

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        hasil_model,
        f,
        indent=4
    )


# =========================================================
# 16. SIMPAN HASIL PREDIKSI
# =========================================================

hasil_testing.to_csv(
    os.path.join(
        BASE_DIR,
        "hasil_testing.csv"
    ),
    index=False
)

feature_importance.to_csv(
    os.path.join(
        BASE_DIR,
        "feature_importance.csv"
    ),
    index=False
)


print("\nFile berhasil dibuat:")
print("- model_rf.pkl")
print("- hasil_model.json")
print("- hasil_testing.csv")
print("- feature_importance.csv")

print("\nPELATIHAN SELESAI.")
print("=" * 60)