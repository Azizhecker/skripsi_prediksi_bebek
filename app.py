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
        return render_template('index.html', error="Model belum dilatih. Jalankan train_model.py terlebih dahulu.")

    try:
        suhu = float(request.form['suhu'])
        kelembapan = float(request.form['kelembapan'])
        curah_hujan = float(request.form['curah_hujan'])
        usia_bebek = float(request.form['usia_bebek'])

        input_data = np.array([[suhu, kelembapan, curah_hujan, usia_bebek]])

        prediksi = model.predict(input_data)
        hasil = int(round(prediksi[0]))
        
        # Mencegah hasil prediksi minus
        if hasil < 0:
            hasil = 0

        return render_template('index.html', 
                               hasil_prediksi=hasil, 
                               suhu=suhu, 
                               kelembapan=kelembapan, 
                               curah_hujan=curah_hujan, 
                               usia_bebek=usia_bebek) 

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
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Menggunakan nama file CSV terakhir yang Anda unggah
    file_dataset = os.path.join(BASE_DIR, 'datapeternakbebak.csv') 
    
    try:
        if os.path.exists(file_dataset):
            df = pd.read_csv(file_dataset)
            
            # Ambil semua data (maksimal 365 hari terakhir agar browser tidak berat)
            df_terbaru = df.tail(365)
            labels = df_terbaru['tanggal'].tolist()
            data_aktual = df_terbaru['produksi_telur'].tolist()
            
            # Jika model sudah dilatih, buat prediksinya untuk grafik kedua
            if model is not None:
                X_grafik = df_terbaru[['suhu', 'kelembapan', 'curah_hujan', 'usia_bebek']]
                prediksi = model.predict(X_grafik)
                data_prediksi = [int(round(p)) for p in prediksi]
                
    except Exception as e:
        print(f"Gagal memuat data grafik: {str(e)}")
        
    return render_template('grafik.html', labels=labels, data_aktual=data_aktual, data_prediksi=data_prediksi)

if __name__ == '__main__':
    app.run(debug=True)