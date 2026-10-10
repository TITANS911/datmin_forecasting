# -*- coding: utf-8 -*-
"""
Kode tambahan: melatih MLP 2-2-1 pada sampel kecil (20 hari) untuk simulasi Excel.
Bobot yang dihasilkan dimasukkan sebagai parameter pada sheet "Neural Network".
"""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

# 1. Ambil 27 hari terakhir total penjualan (ribu unit) lalu bentuk X1 = Y(t-1), X2 = Y(t-7)
df = pd.read_csv("data/train.csv", usecols=["date", "sales"], parse_dates=["date"])
seri = (df.groupby("date")["sales"].sum() / 1000).round(1).iloc[-27:]
baris = [(seri.iloc[i - 1], seri.iloc[i - 7], seri.iloc[i]) for i in range(7, 27)]
d = pd.DataFrame(baris, columns=["x1", "x2", "y"])
latih, uji = d.iloc[:14], d.iloc[14:]          # 14 latih, 6 uji

# 2. Normalisasi min-max (parameter dari data latih)
mn, mx = latih.min(), latih.max()
norm = lambda t: (t - mn[t.columns]) / (mx[t.columns] - mn[t.columns])
X = norm(latih[["x1", "x2"]]).values
y = ((latih["y"] - mn["y"]) / (mx["y"] - mn["y"])).values

# 3. Latih beberapa inisialisasi acak, pilih yang loss data latihnya terkecil
terbaik = None
for seed in range(60):
    m = MLPRegressor(hidden_layer_sizes=(2,), activation="logistic", solver="lbfgs",
                     alpha=0.01, max_iter=5000, tol=1e-10, random_state=seed).fit(X, y)
    loss = np.mean((m.predict(X) - y) ** 2)
    if terbaik is None or loss < terbaik[0]:
        terbaik = (loss, seed, m)
loss, seed, m = terbaik

# 4. Tampilkan bobot (sesuai penamaan pada Excel)
W, B = m.coefs_[0], m.intercepts_[0]
V, by = m.coefs_[1].ravel(), m.intercepts_[1][0]
print("seed terpilih :", seed)
print("w11, w12, b1  :", round(W[0, 0], 4), round(W[1, 0], 4), round(B[0], 4))
print("w21, w22, b2  :", round(W[0, 1], 4), round(W[1, 1], 4), round(B[1], 4))
print("v1, v2, by    :", round(V[0], 4), round(V[1], 4), round(by, 4))
