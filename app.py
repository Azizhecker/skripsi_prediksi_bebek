from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Memuat model Random Forest
MODEL_PATH = 'model_rf.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', hasil_prediksi=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', error="Model belum dilatih.")

    try:
        # Menangkap input lama
        suhu = float(request.form['suhu'])
        kelembapan = float(request.form['kelembapan'])
        curah_hujan = float(request.form['curah_hujan'])
        usia_bebek = float(request.form['usia_bebek'])
        
        # Menangkap input baru
        jenis_bebek_teks = request.form['jenis_bebek']
        populasi_ekor = int(request.form['populasi_ekor'])
        pakan_kg = float(request.form['pakan_kg'])
        
        # Mapping jenis bebek (HARUS SAMA DENGAN TRAIN_MODEL.PY)
        jenis_bebek_num = 0 if jenis_bebek_teks == 'Bebek Air' else 1
        
        # Urutan input HARUS SAMA PERSIS dengan saat train_model.py
        input_data = np.array([[suhu, kelembapan, curah_hujan, usia_bebek, jenis_bebek_num, populasi_ekor, pakan_kg]])

        prediksi = model.predict(input_data)
        hasil = int(round(prediksi[0]))
        
        if hasil < 0: hasil = 0

        return render_template('index.html', 
                               hasil_prediksi=hasil, 
                               suhu=suhu, kelembapan=kelembapan, 
                               curah_hujan=curah_hujan, usia_bebek=usia_bebek,
                               jenis_bebek=jenis_bebek_teks, 
                               populasi_ekor=populasi_ekor, pakan_kg=pakan_kg) 

    except Exception as e:
        return render_template('index.html', error=f"Terjadi kesalahan teknis: {str(e)}")

@app.route('/cara-kerja', methods=['GET'])
def cara_kerja():
    return render_template('cara_kerja.html')

@app.route('/evaluasi', methods=['GET'])
def evaluasi():
    return render_template('evaluasi.html')

@app.route('/kesimpulan', methods=['GET'])
def kesimpulan():
    return render_template('kesimpulan.html')

@app.route('/grafik', methods=['GET'])
def grafik():
    labels = []
    data_aktual = []
    data_prediksi = []
    
    # --- TAMBAHKAN BARIS INI (Pendefinisian variabel yang hilang) ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_dataset = os.path.join(BASE_DIR, 'dataset_bebek_realistis.csv') 
    # ----------------------------------------------------------------
    
    try:
        if os.path.exists(file_dataset):
            df = pd.read_csv(file_dataset)
            
            # Ambil 365 data terakhir
            df_terbaru = df.tail(365).copy() # Gunakan .copy() agar tidak kena peringatan SettingWithCopy
            
            labels = df_terbaru['tanggal'].tolist()
            data_aktual = df_terbaru['produksi_telur'].tolist()
            
            if model is not None:
                # Encoding jenis bebek
                df_terbaru['jenis_bebek_num'] = df_terbaru['jenis_bebek'].map({'Bebek Air': 0, 'Bebek Lati': 1})
                
                # Gunakan 7 fitur yang sama dengan train_model.py
                fitur_grafik = df_terbaru[['suhu', 'kelembapan', 'curah_hujan', 'usia_bebek', 'jenis_bebek_num', 'populasi_ekor', 'pakan_kg']]
                
                prediksi = model.predict(fitur_grafik)
                data_prediksi = [int(round(p)) for p in prediksi]
                
    except Exception as e:
        print(f"Gagal memuat data grafik: {str(e)}")
        
    return render_template('grafik.html', labels=labels, data_aktual=data_aktual, data_prediksi=data_prediksi)

if __name__ == '__main__':
    app.run(debug=True)