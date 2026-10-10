# -*- coding: utf-8 -*-
"""
Eksplorasi Model Forecasting Menggunakan Linear Regression dan Neural Network
Mata Kuliah : IFB-307 Data Mining & Information Retrieval
Kelompok    : 9
Dataset     : Kaggle "Store Sales - Time Series Forecasting" (Corporacion Favorita)

Cara menjalankan:
    1. Ekstrak store-sales-time-series-forecasting.zip ke folder "data/"
       (minimal berisi train.csv dan test.csv)
    2. pip install pandas numpy scikit-learn matplotlib
    3. python forecasting_penjualan.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 0. KONFIGURASI
# ----------------------------------------------------------------------
DATA_DIR = "data"
OUT_DIR = "hasil"
TEST_DAYS = 28          # 28 hari terakhir dipakai sebagai data uji
HORIZON = 16            # horizon peramalan ke depan (16-31 Agustus 2017)
SEED = 42
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (dalam persen)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluasi(nama, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {"Model": nama, "MAE": mae, "MSE": mse,
            "RMSE": np.sqrt(mse), "MAPE (%)": mape(y_true, y_pred)}


# ----------------------------------------------------------------------
# 1. MEMUAT DATA DAN PRA-PEMROSESAN
# ----------------------------------------------------------------------
print("=" * 70)
print("1. DATASET")
print("=" * 70)
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["date"])
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), parse_dates=["date"])
print("Jumlah baris train :", len(train))
print("Rentang tanggal    :", train["date"].min().date(), "s.d.", train["date"].max().date())
print("Jumlah toko        :", train["store_nbr"].nunique())
print("Jumlah kategori    :", train["family"].nunique())
print("Missing value      :", int(train.isna().sum().sum()))
print(train.head())

# Agregasi: total penjualan & total item promosi per hari (seluruh toko)
harian = train.groupby("date").agg(sales=("sales", "sum"),
                                   promo=("onpromotion", "sum"))
# Lengkapi tanggal yang hilang (25 Desember: toko tutup -> penjualan 0)
kalender = pd.date_range(harian.index.min(), harian.index.max(), freq="D")
harian = harian.reindex(kalender)
harian["sales"] = harian["sales"].fillna(0.0)
harian["promo"] = harian["promo"].fillna(0.0)
harian.index.name = "date"
print("\nJumlah hari (setelah melengkapi kalender):", len(harian))
print(harian["sales"].describe().round(2))

# ----------------------------------------------------------------------
# 2. REKAYASA FITUR (FEATURE ENGINEERING)
# ----------------------------------------------------------------------
LAGS = [1, 7, 14]


def buat_fitur(df):
    """Membentuk fitur lag, promosi, dan hari dalam seminggu."""
    f = pd.DataFrame(index=df.index)
    for k in LAGS:
        f[f"lag_{k}"] = df["sales"].shift(k)
    f["promo"] = df["promo"]
    dow = pd.get_dummies(df.index.dayofweek, prefix="dow").astype(float)
    dow.index = df.index
    for d in range(7):
        f[f"dow_{d}"] = dow.get(f"dow_{d}", 0.0)
    return f


fitur = buat_fitur(harian)
data = pd.concat([fitur, harian["sales"].rename("y")], axis=1).dropna()
data = data[data.index >= "2014-01-01"]      # buang 2013 (toko baru buka, pola belum stabil)

X_cols = [c for c in data.columns if c != "y"]
split_date = data.index[-TEST_DAYS]
train_df = data[data.index < split_date]
test_df = data[data.index >= split_date]
print(f"\nData latih : {len(train_df)} hari ({train_df.index.min().date()} s.d. {train_df.index.max().date()})")
print(f"Data uji   : {len(test_df)} hari ({test_df.index.min().date()} s.d. {test_df.index.max().date()})")

X_tr, y_tr = train_df[X_cols].values, train_df["y"].values
X_te, y_te = test_df[X_cols].values, test_df["y"].values

# Standardisasi (fit hanya pada data latih agar tidak terjadi kebocoran data)
sx, sy = StandardScaler(), StandardScaler()
X_tr_s = sx.fit_transform(X_tr)
X_te_s = sx.transform(X_te)
y_tr_s = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()

# ----------------------------------------------------------------------
# 3. MODEL 1: LINEAR REGRESSION
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. LINEAR REGRESSION")
print("=" * 70)
lr = LinearRegression()
lr.fit(X_tr, y_tr)
pred_lr_tr = lr.predict(X_tr)
pred_lr = lr.predict(X_te)
print("Intercept :", round(lr.intercept_, 4))
koef = pd.Series(lr.coef_, index=X_cols).round(4)
print("Koefisien :")
print(koef.to_string())

# ----------------------------------------------------------------------
# 4. MODEL 2: NEURAL NETWORK (Multi-Layer Perceptron)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. NEURAL NETWORK (MLP)")
print("=" * 70)
nn = MLPRegressor(hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
                  learning_rate_init=0.001, max_iter=2000, early_stopping=True,
                  validation_fraction=0.1, n_iter_no_change=30, random_state=SEED)
nn.fit(X_tr_s, y_tr_s)
pred_nn_tr = sy.inverse_transform(nn.predict(X_tr_s).reshape(-1, 1)).ravel()
pred_nn = sy.inverse_transform(nn.predict(X_te_s).reshape(-1, 1)).ravel()
print("Arsitektur        :", f"{len(X_cols)} input -> 16 -> 8 -> 1 (ReLU, Adam)")
print("Jumlah iterasi    :", nn.n_iter_)
print("Loss akhir (train):", round(nn.loss_, 6))

# ----------------------------------------------------------------------
# 5. EVALUASI
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. EVALUASI MODEL")
print("=" * 70)
hasil_uji = pd.DataFrame([evaluasi("Linear Regression", y_te, pred_lr),
                          evaluasi("Neural Network", y_te, pred_nn)]).set_index("Model")
hasil_latih = pd.DataFrame([evaluasi("Linear Regression", y_tr, pred_lr_tr),
                            evaluasi("Neural Network", y_tr, pred_nn_tr)]).set_index("Model")
print("--- Data Uji ---")
print(hasil_uji.round(2).to_string())
print("\n--- Data Latih ---")
print(hasil_latih.round(2).to_string())
terbaik = hasil_uji["RMSE"].idxmin()
print(f"\nModel dengan RMSE terkecil pada data uji: {terbaik}")
hasil_uji.round(4).to_csv(os.path.join(OUT_DIR, "metrik_data_uji.csv"))

# Perbandingan aktual vs prediksi pada data uji
tabel_uji = pd.DataFrame({"Aktual": y_te, "Pred_LR": pred_lr, "Pred_NN": pred_nn},
                         index=test_df.index).round(2)
print("\nCuplikan aktual vs prediksi (data uji):")
print(tabel_uji.head(10).to_string())
tabel_uji.to_csv(os.path.join(OUT_DIR, "prediksi_data_uji.csv"))

# ----------------------------------------------------------------------
# 6. PERAMALAN 16 HARI KE DEPAN (REKURSIF)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. PERAMALAN 16 HARI KE DEPAN (16-31 AGUSTUS 2017)")
print("=" * 70)
promo_test = test.groupby("date")["onpromotion"].sum()
riwayat_lr = harian["sales"].copy()
riwayat_nn = harian["sales"].copy()
baris = []
for tgl in pd.date_range(harian.index.max() + pd.Timedelta(days=1), periods=HORIZON):
    out = {"date": tgl}
    for nama, riwayat, model, pakai_skala in [("LR", riwayat_lr, lr, False),
                                              ("NN", riwayat_nn, nn, True)]:
        x = {f"lag_{k}": riwayat.loc[tgl - pd.Timedelta(days=k)] for k in LAGS}
        x["promo"] = promo_test.loc[tgl]
        for d in range(7):
            x[f"dow_{d}"] = 1.0 if tgl.dayofweek == d else 0.0
        xv = np.array([[x[c] for c in X_cols]])
        if pakai_skala:
            yhat = sy.inverse_transform(model.predict(sx.transform(xv)).reshape(-1, 1))[0, 0]
        else:
            yhat = model.predict(xv)[0]
        riwayat.loc[tgl] = yhat
        out[nama] = yhat
    baris.append(out)
ramalan = pd.DataFrame(baris).set_index("date").round(2)
ramalan.columns = ["Ramalan_LR", "Ramalan_NN"]
print(ramalan.to_string())
ramalan.to_csv(os.path.join(OUT_DIR, "ramalan_16_hari.csv"))

# ----------------------------------------------------------------------
# 7. VISUALISASI
# ----------------------------------------------------------------------
# (a) Penjualan harian keseluruhan
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(harian.index, harian["sales"] / 1e3, lw=0.8, color="#1f4e79")
ax.set_title("Total Penjualan Harian Seluruh Toko (2013-2017)")
ax.set_xlabel("Tanggal"); ax.set_ylabel("Penjualan (ribu unit)")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "gambar_1_data_harian.png"), dpi=200); plt.close(fig)

# (b) Aktual vs prediksi pada data uji
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(test_df.index, y_te / 1e3, "o-", color="black", label="Aktual", ms=3)
ax.plot(test_df.index, pred_lr / 1e3, "s--", color="#c0392b", label="Linear Regression", ms=3)
ax.plot(test_df.index, pred_nn / 1e3, "^--", color="#27ae60", label="Neural Network", ms=3)
ax.set_title("Perbandingan Aktual dan Prediksi pada Data Uji (28 Hari Terakhir)")
ax.set_xlabel("Tanggal"); ax.set_ylabel("Penjualan (ribu unit)")
ax.legend(); ax.grid(alpha=0.3); fig.autofmt_xdate()
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "gambar_2_uji.png"), dpi=200); plt.close(fig)

# (c) Peramalan 16 hari ke depan
fig, ax = plt.subplots(figsize=(10, 4))
hist = harian["sales"].iloc[-60:]
ax.plot(hist.index, hist / 1e3, color="black", label="Data historis (60 hari)")
ax.plot(ramalan.index, ramalan["Ramalan_LR"] / 1e3, "s--", color="#c0392b", label="Ramalan Linear Regression", ms=3)
ax.plot(ramalan.index, ramalan["Ramalan_NN"] / 1e3, "^--", color="#27ae60", label="Ramalan Neural Network", ms=3)
ax.set_title("Peramalan Penjualan 16-31 Agustus 2017")
ax.set_xlabel("Tanggal"); ax.set_ylabel("Penjualan (ribu unit)")
ax.legend(); ax.grid(alpha=0.3); fig.autofmt_xdate()
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "gambar_3_ramalan.png"), dpi=200); plt.close(fig)

# (d) Perbandingan metrik
fig, ax = plt.subplots(figsize=(6, 3.8))
idx = np.arange(2); w = 0.35
ax.bar(idx - w / 2, hasil_uji["MAE"] / 1e3, w, label="MAE", color="#2980b9")
ax.bar(idx + w / 2, hasil_uji["RMSE"] / 1e3, w, label="RMSE", color="#e67e22")
ax.set_xticks(idx); ax.set_xticklabels(hasil_uji.index)
ax.set_ylabel("Error (ribu unit)"); ax.set_title("Perbandingan Error pada Data Uji")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "gambar_4_metrik.png"), dpi=200); plt.close(fig)

print("\nSelesai. Seluruh keluaran tersimpan di folder:", OUT_DIR)
