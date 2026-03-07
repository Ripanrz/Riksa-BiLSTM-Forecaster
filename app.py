import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Konfigurasi Halaman Web
st.set_page_config(page_title="BiLSTM Stock Forecaster", page_icon="📈", layout="wide")

st.title("📈 Stock Price Forecaster (BiLSTM)")
st.markdown("Aplikasi prediksi harga saham dinamis. Masukkan parameter di sebelah kiri untuk melatih model secara *real-time*.")

# --- 1. INPUT (SIDEBAR) ---
st.sidebar.header("⚙️ Parameter Input")
ticker = st.sidebar.text_input("Kode Saham (Contoh: BRMS.JK, BBCA.JK, AAPL)", value="BRMS.JK")
window_size = st.sidebar.number_input("Windowing (Hari ke belakang)", min_value=30, max_value=120, value=90, step=10)
forecast_days = st.sidebar.number_input("Forecasting (Hari ke depan)", min_value=7, max_value=90, value=30, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Klik tombol di bawah untuk mulai mengunduh data dan melatih model.")

# --- 2. EKSEKUSI ---
if st.sidebar.button("Mulai Analisis & Forecasting"):
    
    status_text = st.empty()
    progress_bar = st.progress(0)

    try:
        # TAHAP 1: Download Data
        status_text.info(f"Mengunduh data {ticker} dari Yahoo Finance...")
        df = yf.download(ticker, start='2020-01-01')
        
        if df.empty:
            st.error("Data tidak ditemukan! Pastikan kode saham benar.")
            st.stop()
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.dropna(inplace=True)
        progress_bar.progress(15)

        # TAHAP 2: Preprocessing & Windowing
        status_text.info("Memproses data dan membaginya menjadi Train & Test...")
        data_close = df.filter(['Close']).values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_close)

        X, y = [], []
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i-window_size:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Train-Test Split (85% Train, 15% Test)
        train_size = int(len(X) * 0.85)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        progress_bar.progress(30)

        # TAHAP 3: Build & Train Model
        status_text.info("Sedang melatih model BiLSTM... (Tunggu sekitar 1-2 menit)")
        model = Sequential()
        model.add(Bidirectional(LSTM(units=64, return_sequences=True), input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(units=32, return_sequences=False)))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Epochs diset 30 agar tidak terlalu lama di server gratis HF, ditambah Early Stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=30, callbacks=[early_stop], verbose=0)
        progress_bar.progress(70)

        # TAHAP 4: Evaluasi pada Data Test
        status_text.info("Mengevaluasi akurasi model pada data historis...")
        test_predict = model.predict(X_test, verbose=0)
        test_predict_inv = scaler.inverse_transform(test_predict)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        test_dates = df.index[-len(y_test):]
        progress_bar.progress(85)

        # TAHAP 5: Forecasting Iteratif
        status_text.info(f"Memprediksi harga untuk {forecast_days} hari ke depan...")
        last_window = data_close[-window_size:]
        last_window_scaled = scaler.transform(last_window)

        X_forecast = []
        X_forecast.append(last_window_scaled)
        X_forecast = np.array(X_forecast)

        forecast_results = []
        for _ in range(forecast_days):
            pred_scaled = model.predict(X_forecast, verbose=0)
            forecast_results.append(pred_scaled[0, 0])
            pred_reshaped = np.reshape(pred_scaled, (1, 1, 1))
            X_forecast = np.append(X_forecast[:, 1:, :], pred_reshaped, axis=1)

        forecast_results = scaler.inverse_transform(np.array(forecast_results).reshape(-1, 1))
        
        last_date = df.index[-1]
        future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=len(forecast_results))
        progress_bar.progress(100)

        # --- 3. OUTPUT (TAMPILAN HASIL) ---
        status_text.success("Selesai! Berikut adalah hasil analisisnya:")
        
        # Output 1: Grafik Evaluasi (Aktual vs Prediktif)
        st.subheader("📊 1. Grafik Evaluasi (Aktual vs Prediksi pada Data Uji)")
        fig_eval, ax_eval = plt.subplots(figsize=(12, 5))
        ax_eval.plot(test_dates, y_test_inv, label='Harga Aktual', color='blue')
        ax_eval.plot(test_dates, test_predict_inv, label='Harga Prediksi Model', color='red', linestyle='dashed')
        ax_eval.set_title(f'Evaluasi Model BiLSTM ({ticker})', fontweight='bold')
        ax_eval.set_ylabel('Harga (IDR/USD)')
        ax_eval.legend()
        ax_eval.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig_eval)

        # Output 2: Grafik Forecasting
        st.subheader(f"🔮 2. Grafik Forecasting ({forecast_days} Hari Kedepan)")
        fig_fore, ax_fore = plt.subplots(figsize=(12, 5))
        ax_fore.plot(df.index[-150:], df['Close'][-150:], label='Data Historis (150 Hari Terakhir)', color='blue')
        ax_fore.plot(future_dates, forecast_results, label='Forecasting Masa Depan', color='orange', linewidth=2, marker='o', markersize=4)
        ax_fore.set_title(f'Proyeksi Harga {ticker}', fontweight='bold')
        ax_fore.set_ylabel('Harga (IDR/USD)')
        ax_fore.legend()
        ax_fore.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig_fore)

        # Output 3: Tabel Harga Forecasting
        st.subheader(f"📋 3. Tabel Harga Forecasting")
        df_forecast = pd.DataFrame({
            'Tanggal': future_dates.strftime('%Y-%m-%d'),
            'Harga Prediksi': np.round(forecast_results.flatten(), 2)
        })
        st.dataframe(df_forecast, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan teknis: {e}")