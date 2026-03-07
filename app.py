# ==============================================================================
# 1. PERSIAPAN DAN IMPORT LIBRARY
# ==============================================================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Konfigurasi Halaman Utama
st.set_page_config(page_title="Riksa-BiLSTM Stock Forecaster", page_icon="📈", layout="wide")
st.title("📈 Riksa-BiLSTM: Stock Price Forecaster")
st.markdown("Aplikasi prediksi harga saham interaktif menggunakan pipeline Deep Learning (BiLSTM).")

# ==============================================================================
# 2. DATA COLLECTION (AKUISISI DATA)
# ==============================================================================
@st.cache_data(ttl=3600)
def load_stock_data(ticker):
    df = yf.download(ticker, start='2020-01-01')
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        df.index = df.index.tz_localize(None) 
    return df

# Inisialisasi Session State
if 'run_success' not in st.session_state:
    st.session_state['run_success'] = False

# Parameter Input (Sidebar)
st.sidebar.header("⚙️ Parameter Input")
ticker = st.sidebar.text_input("Kode Saham (Contoh: BRMS.JK, AAPL)", value="BRMS.JK").upper()
window_size = st.sidebar.number_input("Windowing (Hari ke belakang)", min_value=30, max_value=360, value=90, step=30)
forecast_days = st.sidebar.number_input("Forecasting (Hari ke depan)", min_value=7, max_value=120, value=30, step=7)

st.sidebar.markdown("---")
if forecast_days > 60:
    st.sidebar.warning("⚠️ Forecasting >60 hari rentan compounding error.")
if window_size > 180:
    st.sidebar.info("💡 Windowing besar akan memperlama waktu komputasi.")

# ==============================================================================
# PIPELINE EKSEKUSI UTAMA
# ==============================================================================
if st.sidebar.button("🚀 Mulai Analisis & Forecasting"):
    st.session_state['run_success'] = False
    status_text = st.empty()
    progress_bar = st.progress(0)

    try:
        # --- Data Collection (Eksekusi) ---
        status_text.info(f"Mengunduh data historis {ticker}...")
        df = load_stock_data(ticker)
        if df.empty:
            st.error("Data tidak ditemukan! Pastikan kode saham benar.")
            st.stop()
        progress_bar.progress(10)

        # --- 3. EXPLORATORY DATA ANALYSIS (EDA) PREP ---
        # Kita simpan data mentah ke state untuk divisualisasikan nanti
        st.session_state['raw_df'] = df.copy()

        # --- 4. DATA PREPROCESSING ---
        status_text.info("Melakukan Data Preprocessing (Scaling)...")
        data_close = df.filter(['Close']).values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_close)
        progress_bar.progress(20)

        # --- 5. WINDOWING TIME SERIES ---
        status_text.info("Melakukan Windowing Time Series...")
        X, y = [], []
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i-window_size:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        progress_bar.progress(30)

        # --- 6. TRAIN-TEST SPLIT (TIME-SERIES AWARE) ---
        # Pembagian berurutan (tidak diacak) agar tidak terjadi data leakage
        status_text.info("Membagi data Train & Test (85:15)...")
        train_size = int(len(X) * 0.85)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        progress_bar.progress(40)

        # --- 7. MODEL BUILDING (BiLSTM) ---
        status_text.info("Membangun arsitektur BiLSTM...")
        model = Sequential([
            Bidirectional(LSTM(units=64, return_sequences=True), input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            Bidirectional(LSTM(units=32, return_sequences=False)),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # --- 8. MODEL TRAINING (100 Epochs + Early Stopping) ---
        status_text.info("Melatih model (Max 100 Epochs)... Proses ini memakan waktu.")
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=100, callbacks=[early_stop], verbose=0)
        progress_bar.progress(70)

        # --- 9. MODEL EVALUATION (PREDIKSI PADA DATA TEST) ---
        status_text.info("Menghitung metrik evaluasi...")
        test_predict = model.predict(X_test, verbose=0)
        
        test_predict_inv = scaler.inverse_transform(test_predict).flatten()
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        test_dates = df.index[-len(y_test):]

        mae = mean_absolute_error(y_test_inv, test_predict_inv)
        rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict_inv))
        mape = np.mean(np.abs((y_test_inv - test_predict_inv) / y_test_inv)) * 100
        r2 = r2_score(y_test_inv, test_predict_inv)
        progress_bar.progress(85)

        # --- 10. FORECASTING ---
        status_text.info(f"Memprediksi harga {forecast_days} hari ke depan...")
        last_window = data_close[-window_size:]
        last_window_scaled = scaler.transform(last_window)

        X_forecast = np.array([last_window_scaled])
        forecast_results = []
        
        for _ in range(forecast_days):
            pred_scaled = model.predict(X_forecast, verbose=0)
            forecast_results.append(pred_scaled[0, 0])
            pred_reshaped = np.reshape(pred_scaled, (1, 1, 1))
            X_forecast = np.append(X_forecast[:, 1:, :], pred_reshaped, axis=1)

        forecast_results = scaler.inverse_transform(np.array(forecast_results).reshape(-1, 1)).flatten()
        last_date = df.index[-1]
        future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=len(forecast_results))
        progress_bar.progress(100)

        # Simpan semua hasil ke Session State
        st.session_state['run_success'] = True
        st.session_state['ticker'] = ticker
        st.session_state['metrics'] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}
        st.session_state['test_dates'] = test_dates
        st.session_state['y_test_inv'] = y_test_inv
        st.session_state['test_predict_inv'] = test_predict_inv
        st.session_state['hist_dates'] = df.index[-150:]
        st.session_state['hist_close'] = df['Close'][-150:].values
        st.session_state['future_dates'] = future_dates
        st.session_state['forecast_results'] = forecast_results
        
        status_text.empty()
        progress_bar.empty()

    except Exception as e:
        status_text.empty()
        progress_bar.empty()
        st.error(f"Terjadi kesalahan teknis dalam pipeline: {e}")

# ==============================================================================
# 11. VISUALISASI HASIL (EDA, EVALUASI, FORECASTING)
# ==============================================================================
if st.session_state.get('run_success', False):
    st.success(f"Pipeline Selesai! Menampilkan hasil untuk {st.session_state['ticker']}")

    # Membuat 3 Tab UI untuk memisahkan hasil agar rapi
    tab1, tab2, tab3 = st.tabs(["📊 EDA", "🎯 Evaluasi Model", "🔮 Hasil Forecasting"])

    # --- TAB 1: EDA ---
    with tab1:
        st.subheader(f"Exploratory Data Analysis (EDA) - {st.session_state['ticker']}")
        raw_df = st.session_state['raw_df']
        
        col_eda1, col_eda2, col_eda3 = st.columns(3)
        col_eda1.metric("Harga Tertinggi (All Time)", f"{raw_df['High'].max():,.2f}")
        col_eda2.metric("Harga Terendah (All Time)", f"{raw_df['Low'].min():,.2f}")
        col_eda3.metric("Rata-rata Volume", f"{raw_df['Volume'].mean():,.0f}")

        fig_eda = go.Figure()
        fig_eda.add_trace(go.Scatter(x=raw_df.index, y=raw_df['Close'], mode='lines', line=dict(color='#1f77b4')))
        fig_eda.update_layout(title="Pergerakan Harga Penutupan Historis", xaxis_title="Tanggal", yaxis_title="Harga")
        st.plotly_chart(fig_eda, use_container_width=True)

    # --- TAB 2: EVALUASI MODEL ---
    with tab2:
        st.subheader("Metrik Akurasi pada Data Test")
        m = st.session_state['metrics']
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("MAE", f"{m['MAE']:,.2f}")
        col_m2.metric("RMSE", f"{m['RMSE']:,.2f}")
        col_m3.metric("MAPE", f"{m['MAPE']:.2f}%")
        col_m4.metric("R² Score", f"{m['R2']:.4f}")

        st.markdown("---")
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(x=st.session_state['test_dates'], y=st.session_state['y_test_inv'], mode='lines', name='Harga Aktual', line=dict(color='blue')))
        fig_eval.add_trace(go.Scatter(x=st.session_state['test_dates'], y=st.session_state['test_predict_inv'], mode='lines', name='Prediksi BiLSTM', line=dict(color='red', dash='dash')))
        fig_eval.update_layout(title="Grafik Evaluasi: Aktual vs Prediksi (Data Uji)", xaxis_title="Tanggal", yaxis_title="Harga", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_eval, use_container_width=True)

    # --- TAB 3: FORECASTING & TABEL ---
    with tab3:
        st.subheader("Proyeksi Harga ke Depan")
        fig_fore = go.Figure()
        fig_fore.add_trace(go.Scatter(x=st.session_state['hist_dates'], y=st.session_state['hist_close'], mode='lines', name='Data Historis (150 Hari)', line=dict(color='blue')))
        fig_fore.add_trace(go.Scatter(x=st.session_state['future_dates'], y=st.session_state['forecast_results'], mode='lines+markers', name='Forecasting', line=dict(color='orange'), marker=dict(size=4)))
        fig_fore.update_layout(title="Grafik Forecasting Masa Depan", xaxis_title="Tanggal", yaxis_title="Harga", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_fore, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Tabel Prediksi Harian")
        df_forecast = pd.DataFrame({
            'Tanggal': st.session_state['future_dates'].strftime('%Y-%m-%d'),
            'Harga Prediksi': np.round(st.session_state['forecast_results'], 2)
        })
        st.dataframe(df_forecast, use_container_width=True, hide_index=True)