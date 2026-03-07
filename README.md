---
title: Riksa-BiLSTM
emoji: 🚀
colorFrom: red
colorTo: red
sdk: streamlit
app_file: app.py
pinned: false
python_version: '3.10'
sdk_version: 1.55.0
---

# 📈 Riksa-BiLSTM: Stock Price Forecaster

[![Live Demo on Hugging Face](https://img.shields.io/badge/Live%20Demo-%F0%9F%A4%97%20Hugging%20Face-blue?style=for-the-badge)](https://huggingface.co/spaces/Ripanrz/Riksa-BiLSTM)

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Deep Learning](https://img.shields.io/badge/Model-BiLSTM_(TensorFlow)-orange)
![Data](https://img.shields.io/badge/API-yfinance-green)
![UI](https://img.shields.io/badge/UI-Streamlit-lightgrey)
![Data Viz](https://img.shields.io/badge/Viz-Plotly-purple)

**Menganalisis dan memproyeksikan pergerakan harga saham di tengah volatilitas pasar yang tinggi membutuhkan alat bantu komputasi yang tajam.** **Riksa-BiLSTM** dibangun untuk memecahkan tantangan tersebut. 

Mengambil filosofi kata **"Riksa"** (Sunda) yang berarti *mengamati dengan saksama dan teliti*, proyek ini mengimplementasikan algoritma *Deep Learning* tingkat lanjut untuk meneliti pola data masa lalu. Menggunakan pendekatan **Bidirectional Long Short-Term Memory (BiLSTM)**, sistem ini mampu membaca konteks data *time-series* finansial dari dua arah (masa lalu ke masa kini, dan sebaliknya) untuk memproyeksikan pergerakan harga aset di masa depan dengan akurasi yang lebih terukur.

---

## 📸 Tampilan Dashboard

> *Aplikasi web interaktif dengan pipeline komputasi end-to-end, di-deploy secara live di ekosistem Hugging Face Spaces.*

![Tampilan Dashboard](src/Cuplikan%20layar%202026-03-07%20170006.png)

---

## 🚀 Fitur Utama

* **Dynamic Data Ingestion**: Tidak perlu mengunduh dataset CSV secara manual. Sistem terintegrasi langsung dengan API `yfinance` untuk menarik data historis saham, indeks, atau *cryptocurrency* secara *real-time* berdasarkan input kode *ticker*.
* **Arsitektur Model BiLSTM**: Memanfaatkan *layer* jaringan saraf tiruan dua arah dari TensorFlow/Keras untuk menangkap dependensi jangka panjang pada pola data *time-series*, lengkap dengan mekanisme *Early Stopping* untuk mencegah *overfitting*.
* **Comprehensive Evaluation Metrics**: Dilengkapi dengan penghitungan metrik standar industri Data Science (MAE, RMSE, MAPE, dan R² Score) untuk mengukur performa prediksi model secara transparan pada data uji.
* **Interactive Visualizations**: Menggunakan `plotly.graph_objects` untuk menghasilkan grafik *Exploratory Data Analysis* (EDA), Evaluasi, dan *Forecasting* yang dinamis (dapat di-*zoom*, digeser, dan di-*hover*).
* **Optimized State Management**: Menggunakan `st.session_state` dan fitur *caching* dari Streamlit untuk memastikan antarmuka web tetap responsif, ringan, dan tidak mengalami *reload* komputasi berulang saat pengguna berinteraksi dengan grafik.

---

## 🔁 Arsitektur Sistem (Data Pipeline)

```mermaid
graph LR
    A[Input UI] -->|Ticker, Window, Forecast| B(Data Collection)
    B -->|yfinance API| C{EDA & Preprocessing}
    C -->|MinMaxScaler & Windowing| D[Model Building]
    D -->|BiLSTM Training| E(Model Evaluation)
    E -->|MAE, RMSE, MAPE, R2| F[Future Forecasting]
    F -->|Iterative Prediction| G[Render UI Dashboard]
    G -->|Session State| H[Tampilkan Visualisasi Plotly]
