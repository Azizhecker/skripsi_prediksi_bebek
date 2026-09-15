import os
import json
import glob
import joblib
import numpy as np
import pandas as pd
import pymysql

from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import check_password_hash


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)
app.secret_key = "skripsi_prediksi_bebek_2026"


# ============================================================
# DATABASE MYSQL
# ============================================================

MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DATABASE = "prediksi_bebek"


# ============================================================
# PATH PROJECT
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")


# ============================================================
# CARI FILE DATASET
# ============================================================

DATASET_CANDIDATES = [
    "dataset_bebek_skripsi_508.csv",
    "dataset_bebek_508.csv",
    "dataset_bebek.csv",
]

DATASET_PATH = None

for filename in DATASET_CANDIDATES:
    path = os.path.join(BASE_DIR, filename)

    if os.path.exists(path):
        DATASET_PATH = path
        break


# ============================================================
# CARI MODEL
# ============================================================

MODEL_CANDIDATES = [
    # Nama model yang umum digunakan
    "model_random_forest_bebek.pkl",
    "model_rf.pkl",
    "model_random_forest.pkl",
    "model_random_forest.joblib",
    "model_rf.joblib",
]

MODEL_PATH = None

for filename in MODEL_CANDIDATES:
    path = os.path.join(BASE_DIR, filename)

    if os.path.exists(path):
        MODEL_PATH = path
        break


# ============================================================
# CARI FEATURE IMPORTANCE
# ============================================================

FEATURE_CANDIDATES = [
    "feature_importance.csv",
    "feature_importance_rf.csv",
    "hasil_feature_importance.csv",
]

FEATURE_PATH = None

for filename in FEATURE_CANDIDATES:
    path = os.path.join(BASE_DIR, filename)

    if os.path.exists(path):
        FEATURE_PATH = path
        break


# ============================================================
# CARI HASIL MODEL
# ============================================================

RESULT_CANDIDATES = [
    "hasil_model.json",
    "performa_model.json",
    "metrics.json",
    "hasil_evaluasi.json",
]

RESULT_PATH = None

for filename in RESULT_CANDIDATES:
    path = os.path.join(BASE_DIR, filename)

    if os.path.exists(path):
        RESULT_PATH = path
        break


# ============================================================
# FITUR MODEL
# ============================================================

NUMERIC_FEATURES = [
    "populasi_ratu",
    "populasi_air",
    "populasi_total",
    "kepadatan_kandang",
    "pakan_kg",
    "suhu",
    "kelembapan",
    "curah_hujan",
    "usia_ratu_bulan",
    "usia_air_bulan",
]

CATEGORICAL_FEATURES = [
    "kondisi_cuaca",
]

MODEL_FEATURES = (
    NUMERIC_FEATURES +
    CATEGORICAL_FEATURES
)


# ============================================================
# DATASET
# ============================================================

df_dataset = pd.DataFrame()

if DATASET_PATH:

    try:
        df_dataset = pd.read_csv(
            DATASET_PATH
        )

        print(
            "Dataset berhasil:",
            DATASET_PATH
        )

        print(
            "Jumlah data:",
            len(df_dataset)
        )

    except Exception as e:

        print(
            "Gagal membaca dataset:",
            e
        )

else:

    print(
        "DATASET TIDAK DITEMUKAN"
    )


# ============================================================
# MODEL
# ============================================================

model_package = None


def load_model():

    global MODEL_PATH

    # --------------------------------------------------------
    # CARI SEMUA PKL/JOBLIB
    # --------------------------------------------------------

    if MODEL_PATH is None:

        files = []

        files.extend(
            glob.glob(
                os.path.join(
                    BASE_DIR,
                    "*.pkl"
                )
            )
        )

        files.extend(
            glob.glob(
                os.path.join(
                    BASE_DIR,
                    "*.joblib"
                )
            )
        )

        # Cari yang namanya berkaitan dengan model
        for path in files:

            name = os.path.basename(
                path
            ).lower()

            if (
                "model" in name
                or "random" in name
                or "rf" in name
            ):

                MODEL_PATH = path
                break

    if MODEL_PATH is None:
        return None

    try:

        loaded = joblib.load(
            MODEL_PATH
        )

        print(
            "MODEL BERHASIL DIMUAT:",
            MODEL_PATH
        )

        return loaded

    except Exception as e:

        print(
            "GAGAL LOAD MODEL:",
            e
        )

        return None


model_package = load_model()


# ============================================================
# AMBIL MODEL PIPELINE
# ============================================================

def get_model():

    if model_package is None:
        return None

    if isinstance(
        model_package,
        dict
    ):

        # Format:
        # {"model": pipeline}

        if "model" in model_package:
            return model_package["model"]

        if "pipeline" in model_package:
            return model_package["pipeline"]

        if "best_model" in model_package:
            return model_package["best_model"]

        return None

    return model_package


# ============================================================
# METRICS
# ============================================================

def get_metrics():

    metrics = {
        "MAE": 0,
        "RMSE": 0,
        "MAPE": 0,
        "R2": 0,
    }

    # --------------------------------------------------------
    # DARI MODEL PACKAGE
    # --------------------------------------------------------

    try:

        if isinstance(
            model_package,
            dict
        ):

            saved_metrics = model_package.get(
                "metrics",
                {}
            )

            if isinstance(
                saved_metrics,
                dict
            ):

                metrics["MAE"] = float(
                    saved_metrics.get(
                        "MAE",
                        saved_metrics.get(
                            "mae",
                            0
                        )
                    )
                )

                metrics["RMSE"] = float(
                    saved_metrics.get(
                        "RMSE",
                        saved_metrics.get(
                            "rmse",
                            0
                        )
                    )
                )

                metrics["MAPE"] = float(
                    saved_metrics.get(
                        "MAPE",
                        saved_metrics.get(
                            "mape",
                            0
                        )
                    )
                )

                metrics["R2"] = float(
                    saved_metrics.get(
                        "R2",
                        saved_metrics.get(
                            "r2",
                            0
                        )
                    )
                )

    except Exception as e:

        print(
            "Metrics model error:",
            e
        )


    # --------------------------------------------------------
    # DARI JSON
    # --------------------------------------------------------

    if RESULT_PATH:

        try:

            with open(
                RESULT_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                result = json.load(
                    file
                )

            saved = result.get(
                "metrics",
                result
            )

            if isinstance(
                saved,
                dict
            ):

                metrics["MAE"] = float(
                    saved.get(
                        "MAE",
                        saved.get(
                            "mae",
                            metrics["MAE"]
                        )
                    )
                )

                metrics["RMSE"] = float(
                    saved.get(
                        "RMSE",
                        saved.get(
                            "rmse",
                            metrics["RMSE"]
                        )
                    )
                )

                metrics["MAPE"] = float(
                    saved.get(
                        "MAPE",
                        saved.get(
                            "mape",
                            metrics["MAPE"]
                        )
                    )
                )

                metrics["R2"] = float(
                    saved.get(
                        "R2",
                        saved.get(
                            "r2",
                            metrics["R2"]
                        )
                    )
                )

        except Exception as e:

            print(
                "Metrics JSON error:",
                e
            )

    return metrics


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance():

    result = []


    # --------------------------------------------------------
    # 1. DARI MODEL PACKAGE
    # --------------------------------------------------------

    try:

        if isinstance(
            model_package,
            dict
        ):

            saved = model_package.get(
                "feature_importance",
                []
            )

            if isinstance(
                saved,
                list
            ):

                for item in saved:

                    if isinstance(
                        item,
                        dict
                    ):

                        feature = (
                            item.get(
                                "feature"
                            )
                            or
                            item.get(
                                "fitur"
                            )
                            or
                            item.get(
                                "name"
                            )
                        )

                        importance = (
                            item.get(
                                "importance",
                                item.get(
                                    "nilai",
                                    0
                                )
                            )
                        )

                        if feature:

                            result.append(
                                {
                                    "feature":
                                        str(feature),

                                    "importance":
                                        float(
                                            importance
                                        )
                                }
                            )

    except Exception as e:

        print(
            "Feature model error:",
            e
        )


    if result:

        result.sort(
            key=lambda x:
            x["importance"],
            reverse=True
        )

        return result


    # --------------------------------------------------------
    # 2. DARI CSV
    # --------------------------------------------------------

    if FEATURE_PATH:

        try:

            feature_df = pd.read_csv(
                FEATURE_PATH
            )

            feature_column = None
            importance_column = None

            for column in feature_df.columns:

                lower = column.lower()

                if lower in [
                    "feature",
                    "fitur",
                    "features",
                    "nama_fitur",
                ]:

                    feature_column = column

                if lower in [
                    "importance",
                    "nilai_importance",
                    "feature_importance",
                    "nilai",
                ]:

                    importance_column = column

            if (
                feature_column
                and
                importance_column
            ):

                for _, row in feature_df.iterrows():

                    result.append(
                        {
                            "feature":
                                str(
                                    row[
                                        feature_column
                                    ]
                                ),

                            "importance":
                                float(
                                    row[
                                        importance_column
                                    ]
                                )
                        }
                    )

        except Exception as e:

            print(
                "Feature CSV error:",
                e
            )


    if result:

        result.sort(
            key=lambda x:
            x["importance"],
            reverse=True
        )

        return result


    # --------------------------------------------------------
    # 3. DARI PIPELINE RANDOM FOREST
    # --------------------------------------------------------

    try:

        model = get_model()

        if model is None:
            return []

        rf = None
        preprocessor = None

        if hasattr(
            model,
            "named_steps"
        ):

            rf = model.named_steps.get(
                "model"
            )

            preprocessor = model.named_steps.get(
                "preprocessor"
            )

        if rf is None:

            if hasattr(
                model,
                "feature_importances_"
            ):

                rf = model

        if rf is None:
            return []

        if not hasattr(
            rf,
            "feature_importances_"
        ):

            return []

        importances = (
            rf.feature_importances_
        )

        if preprocessor is not None:

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

        else:

            feature_names = (
                MODEL_FEATURES
            )

        for name, value in zip(
            feature_names,
            importances
        ):

            clean_name = str(
                name
            )

            clean_name = clean_name.replace(
                "num__",
                ""
            )

            clean_name = clean_name.replace(
                "cat__",
                ""
            )

            result.append(
                {
                    "feature":
                        clean_name,

                    "importance":
                        float(value)
                }
            )

    except Exception as e:

        print(
            "Feature pipeline error:",
            e
        )


    result.sort(
        key=lambda x:
        x["importance"],
        reverse=True
    )

    return result


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db():

    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )


# ============================================================
# LOGIN REQUIRED
# ============================================================

def login_required(function):

    @wraps(function)
    def decorated_function(
        *args,
        **kwargs
    ):

        if "username" not in session:

            return redirect(
                url_for("login")
            )

        return function(
            *args,
            **kwargs
        )

    return decorated_function


# ============================================================
# ADMIN REQUIRED
# ============================================================

def admin_required(function):

    @wraps(function)
    def decorated_function(
        *args,
        **kwargs
    ):

        if "username" not in session:

            return redirect(
                url_for("login")
            )

        if session.get(
            "role"
        ) != "admin":

            return redirect(
                url_for("dashboard")
            )

        return function(
            *args,
            **kwargs
        )

    return decorated_function


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def login():

    if "username" in session:

        return redirect(
            url_for("dashboard")
        )

    error = None

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if not username or not password:

            error = (
                "Username dan password wajib diisi."
            )

        else:

            try:

                connection = get_db()

                with connection.cursor() as cursor:

                    cursor.execute(
                        """
                        SELECT
                            id,
                            username,
                            password,
                            nama,
                            role,
                            status
                        FROM users
                        WHERE username = %s
                        LIMIT 1
                        """,
                        (username,)
                    )

                    user = cursor.fetchone()

                connection.close()

                if user is None:

                    error = (
                        "Username atau password salah."
                    )

                elif user["status"] != "aktif":

                    error = (
                        "Akun tidak aktif."
                    )

                elif check_password_hash(
                    user["password"],
                    password
                ):

                    session.clear()

                    session["user_id"] = user["id"]
                    session["username"] = user["username"]
                    session["nama"] = user["nama"]
                    session["role"] = user["role"]

                    return redirect(
                        url_for("dashboard")
                    )

                else:

                    error = (
                        "Username atau password salah."
                    )

            except Exception as e:

                print(
                    "LOGIN ERROR:",
                    e
                )

                error = (
                    "Database MySQL belum terhubung."
                )

    return render_template(
        "login.html",
        error=error
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():

    jumlah_data = 0

    rata_rata_produksi = 0

    produksi_minimum = 0

    produksi_maksimum = 0

    populasi_awal = 0

    populasi_akhir = 0

    metrics = get_metrics()

    mae = metrics["MAE"]
    rmse = metrics["RMSE"]
    mape = metrics["MAPE"]
    r2 = metrics["R2"]


    if not df_dataset.empty:

        try:

            jumlah_data = len(
                df_dataset
            )

            if "produksi_telur" in df_dataset:

                produksi = pd.to_numeric(
                    df_dataset[
                        "produksi_telur"
                    ],
                    errors="coerce"
                ).dropna()

                if len(produksi):

                    rata_rata_produksi = float(
                        produksi.mean()
                    )

                    produksi_minimum = float(
                        produksi.min()
                    )

                    produksi_maksimum = float(
                        produksi.max()
                    )


            if "populasi_total" in df_dataset:

                populasi = pd.to_numeric(
                    df_dataset[
                        "populasi_total"
                    ],
                    errors="coerce"
                ).dropna()

                if len(populasi):

                    populasi_awal = int(
                        populasi.iloc[0]
                    )

                    populasi_akhir = int(
                        populasi.iloc[-1]
                    )

        except Exception as e:

            print(
                "DASHBOARD ERROR:",
                e
            )


    return render_template(
        "dashboard.html",

        jumlah_data=jumlah_data,

        rata_rata_produksi=
            rata_rata_produksi,

        produksi_minimum=
            produksi_minimum,

        produksi_maksimum=
            produksi_maksimum,

        populasi_awal=
            populasi_awal,

        populasi_akhir=
            populasi_akhir,

        metrics=metrics,

        mae=mae,

        rmse=rmse,

        mape=mape,

        r2=r2,

        username=session.get(
            "username",
            ""
        ),

        nama=session.get(
            "nama",
            ""
        ),

        role=session.get(
            "role",
            "user"
        )
    )


# ============================================================
# PREDIKSI
# ============================================================

@app.route(
    "/prediksi",
    methods=["GET", "POST"]
)
@login_required
def prediksi():

    # WAJIB ADA AGAR TEMPLATE TIDAK ERROR
    input_data = {}

    data_input = {}

    hasil_prediksi = None

    error = None


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    default_data = {

        "populasi_ratu": 1600,

        "populasi_air": 400,

        "populasi_total": 2000,

        "kepadatan_kandang": 4,

        "pakan_kg": 22,

        "kondisi_cuaca": "Normal",

        "suhu": 29.5,

        "kelembapan": 66,

        "curah_hujan": 5,

        "usia_ratu_bulan": 18,

        "usia_air_bulan": 15
    }


    input_data = default_data.copy()


    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    if request.method == "POST":

        try:

            populasi_ratu = float(
                request.form.get(
                    "populasi_ratu",
                    1600
                )
            )

            populasi_air = float(
                request.form.get(
                    "populasi_air",
                    400
                )
            )

            populasi_total = (
                populasi_ratu +
                populasi_air
            )

            kepadatan_kandang = (
                populasi_total /
                500
            )

            pakan_kg = float(
                request.form.get(
                    "pakan_kg",
                    22
                )
            )

            kondisi_cuaca = request.form.get(
                "kondisi_cuaca",
                "Normal"
            )

            suhu = float(
                request.form.get(
                    "suhu",
                    29.5
                )
            )

            kelembapan = float(
                request.form.get(
                    "kelembapan",
                    66
                )
            )

            curah_hujan = float(
                request.form.get(
                    "curah_hujan",
                    5
                )
            )

            usia_ratu_bulan = float(
                request.form.get(
                    "usia_ratu_bulan",
                    18
                )
            )

            usia_air_bulan = float(
                request.form.get(
                    "usia_air_bulan",
                    15
                )
            )


            input_data = {

                "populasi_ratu":
                    populasi_ratu,

                "populasi_air":
                    populasi_air,

                "populasi_total":
                    populasi_total,

                "kepadatan_kandang":
                    kepadatan_kandang,

                "pakan_kg":
                    pakan_kg,

                "kondisi_cuaca":
                    kondisi_cuaca,

                "suhu":
                    suhu,

                "kelembapan":
                    kelembapan,

                "curah_hujan":
                    curah_hujan,

                "usia_ratu_bulan":
                    usia_ratu_bulan,

                "usia_air_bulan":
                    usia_air_bulan
            }

            data_input = input_data.copy()


            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            model = get_model()

            if model is None:

                raise Exception(
                    "Model Random Forest tidak ditemukan. "
                    "Pastikan model_rf.pkl berada di folder project."
                )


            # ------------------------------------------------
            # DATAFRAME
            # ------------------------------------------------

            input_df = pd.DataFrame(
                [input_data]
            )


            # ------------------------------------------------
            # PREDIKSI
            # ------------------------------------------------

            prediction = model.predict(
                input_df[
                    MODEL_FEATURES
                ]
            )

            hasil_prediksi = int(
                round(
                    max(
                        0,
                        float(
                            prediction[0]
                        )
                    )
                )
            )


        except Exception as e:

            print(
                "PREDIKSI ERROR:",
                e
            )

            error = str(e)


    return render_template(
        "prediksi.html",

        input_data=input_data,

        data_input=data_input,

        hasil_prediksi=hasil_prediksi,

        error=error,

        # nilai langsung jika template lama memakainya
        populasi_ratu=input_data.get(
            "populasi_ratu",
            1600
        ),

        populasi_air=input_data.get(
            "populasi_air",
            400
        ),

        populasi_total=input_data.get(
            "populasi_total",
            2000
        ),

        kepadatan_kandang=input_data.get(
            "kepadatan_kandang",
            4
        ),

        pakan_kg=input_data.get(
            "pakan_kg",
            22
        ),

        kondisi_cuaca=input_data.get(
            "kondisi_cuaca",
            "Normal"
        ),

        suhu=input_data.get(
            "suhu",
            29.5
        ),

        kelembapan=input_data.get(
            "kelembapan",
            66
        ),

        curah_hujan=input_data.get(
            "curah_hujan",
            5
        ),

        usia_ratu_bulan=input_data.get(
            "usia_ratu_bulan",
            18
        ),

        usia_air_bulan=input_data.get(
            "usia_air_bulan",
            15
        )
    )


# ============================================================
# GRAFIK
# ============================================================

@app.route("/grafik")
@login_required
def grafik():

    labels = []

    data_aktual = []

    data_prediksi = []


    try:

        if not df_dataset.empty:

            df = df_dataset.copy()

            if "tanggal" in df.columns:

                df["tanggal"] = pd.to_datetime(
                    df["tanggal"],
                    errors="coerce"
                )

                df = df.dropna(
                    subset=["tanggal"]
                )

                df = df.sort_values(
                    "tanggal"
                )

                df = df.tail(365)

                labels = [
                    tanggal.strftime(
                        "%d-%m-%Y"
                    )
                    for tanggal
                    in df["tanggal"]
                ]


            if "produksi_telur" in df.columns:

                data_aktual = (
                    pd.to_numeric(
                        df["produksi_telur"],
                        errors="coerce"
                    )
                    .fillna(0)
                    .round(0)
                    .astype(int)
                    .tolist()
                )


            model = get_model()

            if model is not None:

                model_df = df[
                    MODEL_FEATURES
                ].copy()

                prediction = model.predict(
                    model_df
                )

                data_prediksi = [

                    int(
                        round(
                            max(
                                0,
                                float(value)
                            )
                        )
                    )

                    for value
                    in prediction
                ]


    except Exception as e:

        print(
            "GRAFIK ERROR:",
            e
        )


    return render_template(
        "grafik.html",

        labels=labels,

        data_aktual=data_aktual,

        data_prediksi=data_prediksi,

        jumlah_data=len(
            labels
        )
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

@app.route(
    "/feature-importance"
)
@login_required
def feature_importance():

    data = get_feature_importance()

    return render_template(
        "feature_importance.html",

        data=data,

        feature_importance=data
    )


# ============================================================
# EVALUASI
# ============================================================

@app.route("/evaluasi")
@admin_required
def evaluasi():

    metrics = get_metrics()

    best_params = {}

    try:

        if isinstance(
            model_package,
            dict
        ):

            best_params = model_package.get(
                "best_params",
                model_package.get(
                    "best_params_",
                    {}
                )
            )

    except Exception:
        best_params = {}


    return render_template(
        "evaluasi.html",

        metrics=metrics,

        mae=metrics["MAE"],

        rmse=metrics["RMSE"],

        mape=metrics["MAPE"],

        r2=metrics["R2"],

        best_params=best_params
    )


# ============================================================
# DATA MANAGEMENT
# ============================================================

@app.route("/data")
@admin_required
def data():

    records = []

    columns = []

    jumlah_data = 0


    try:

        if not df_dataset.empty:

            df = df_dataset.copy()

            df = df.fillna("")

            columns = df.columns.tolist()

            records = df.to_dict(
                orient="records"
            )

            jumlah_data = len(
                records
            )


    except Exception as e:

        print(
            "DATA ERROR:",
            e
        )


    return render_template(
        "data.html",

        data=records,
        data_rows=records,  # dipakai oleh template data.html
        records=records,
        columns=columns,
        jumlah_data=jumlah_data
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.route("/model")
@admin_required
def model_info():

    metrics = get_metrics()

    best_params = {}

    model_available = (
        get_model() is not None
    )


    try:

        if isinstance(
            model_package,
            dict
        ):

            best_params = model_package.get(
                "best_params",
                model_package.get(
                    "best_params_",
                    {}
                )
            )

    except Exception:

        best_params = {}


    return render_template(
        "model.html",

        metrics=metrics,
        best_params=best_params,

        # Template model.html menggunakan nama model_loaded
        model_loaded=model_available,
        model_available=model_available,

        model_path=MODEL_PATH,

        dataset_path=DATASET_PATH,

        jumlah_data=len(
            df_dataset
        )
    )


# ============================================================
# CARA KERJA
# ============================================================

@app.route("/cara-kerja")
@login_required
def cara_kerja():

    return render_template(
        "cara_kerja.html"
    )


# ============================================================
# KESIMPULAN
# ============================================================

@app.route("/kesimpulan")
@login_required
def kesimpulan():

    metrics = get_metrics()

    return render_template(
        "kesimpulan.html",

        metrics=metrics,

        mae=metrics["MAE"],

        rmse=metrics["RMSE"],

        mape=metrics["MAPE"],

        r2=metrics["R2"]
    )


# ============================================================
# HASIL PREDIKSI
# ============================================================

@app.route("/hasil-prediksi")
@login_required
def hasil_prediksi():

    records = []

    columns = []


    # --------------------------------------------------------
    # Coba dari testing CSV
    # --------------------------------------------------------

    testing_candidates = [

        "hasil_testing.csv",

        "hasil_prediksi.csv",

        "hasil_prediksi_testing.csv",

        "prediksi_testing.csv",

    ]

    testing_path = None

    for filename in testing_candidates:

        path = os.path.join(
            BASE_DIR,
            filename
        )

        if os.path.exists(path):

            testing_path = path

            break


    if testing_path:

        try:

            df = pd.read_csv(
                testing_path
            )

            df = df.fillna("")

            columns = df.columns.tolist()

            records = df.to_dict(
                orient="records"
            )

        except Exception as e:

            print(
                "HASIL PREDIKSI ERROR:",
                e
            )


    return render_template(
        "hasil_prediksi.html",

        data=records,

        records=records,

        columns=columns
    )


# ============================================================
# ERROR 404
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return redirect(
        url_for("dashboard")
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("")
    print("=" * 70)
    print(" SISTEM PREDIKSI PRODUKSI TELUR BEBEK")
    print("=" * 70)

    print(
        "Project :",
        BASE_DIR
    )

    print(
        "Dataset :",
        DATASET_PATH
    )

    print(
        "Model   :",
        MODEL_PATH
    )

    print(
        "Feature :",
        FEATURE_PATH
    )

    print(
        "Database:",
        MYSQL_DATABASE
    )

    print(
        "Model loaded:",
        get_model() is not None
    )

    print(
        "Data loaded :",
        len(df_dataset)
    )

    print("=" * 70)
    print("")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )