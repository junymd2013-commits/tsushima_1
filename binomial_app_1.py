import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

from matplotlib import font_manager
font_manager.fontManager.addfont("ipaexg.ttf")
plt.rcParams["font.family"] = "IPAexGothic"


st.title("二項分布と正規近似（数学B）")

# --- UI ---
n = st.slider("試行回数 n", 1, 200, 30)
p = st.slider("成功確率 p（例：サイコロ1の目＝1/6）", 0.0, 1.0, 1/6, step=0.001)
show_norm = st.checkbox("正規近似を表示する", True)

# --- 計算 ---
x = np.arange(0, n+1)
binom_pmf = binom.pmf(x, n, p)

mu = n * p
sigma = np.sqrt(n * p * (1 - p))

# 正規近似
xs = np.linspace(0, n, 400)
norm_pdf = norm.pdf(xs, mu, sigma)

# スケール調整（棒グラフに合わせる）
scale_factor = binom_pmf.max() / norm_pdf.max()
norm_pdf_scaled = norm_pdf * scale_factor

# --- 誤差（L1ノルム）---
norm_at_x = norm.pdf(x, mu, sigma) * scale_factor
error = np.sum(np.abs(binom_pmf - norm_at_x))

# --- グラフ描画 ---
fig, ax = plt.subplots(figsize=(8, 4))

# 二項分布（棒グラフ）
ax.bar(x, binom_pmf, color="skyblue", label="二項分布", alpha=0.7)

# 正規近似
if show_norm:
    ax.plot(xs, norm_pdf_scaled, color="red", linewidth=3, label="正規近似")

# 期待値・分散の縦線
ax.axvline(mu, color="green", linestyle="--", linewidth=2, label=f"期待値 E[X]={mu:.2f}")
ax.axvline(mu + sigma, color="gray", linestyle=":", linewidth=1)
ax.axvline(mu - sigma, color="gray", linestyle=":", linewidth=1)

ax.set_xlabel("成功回数 k")
ax.set_ylabel("確率")
ax.set_title(f"二項分布 B(n={n}, p={p:.4f})")
ax.legend()

st.pyplot(fig)

# --- 数値情報 ---
st.subheader("数値情報")
st.write(f"期待値：E[X] = {mu:.4f}")
st.write(f"分散：Var[X] = {sigma**2:.4f}")
st.write(f"二項分布と正規近似の誤差（L1ノルム）：{error:.6f}")

# --- PNG 保存 ---
st.subheader("グラフ保存")
if st.button("PNGとして保存"):
    fig.savefig("binomial_plot.png")
    st.success("binomial_plot.png として保存しました。")
