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

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Riksa-BiLSTM Stock Forecaster", page_icon="📈", layout="wide")

st.title("📈 Riksa-BiLSTM: Stock Price Forecaster")
st.markdown("Aplikasi prediksi harga saham interaktif. Dilengkapi dengan evaluasi metrik dan visualisasi dinamis.")

# --- CACHING FUNGSI DOWNLOAD ---
# Menyimpan data di cache memori agar tidak perlu download berulang kali jika ticker sama
@st.cache_data(ttl=3600)
def load_stock_data(ticker):
    df = yf.download(ticker, start='2020-01-01')
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        # Hilangkan timezone agar aman saat divisualisasikan
        df.index = df.index.tz_localize(None) 
    return df

# --- 1. INPUT (SIDEBAR) ---
st.sidebar.header("⚙️ Parameter Input")
ticker = st.sidebar.text_input("Kode Saham (Contoh: BRMS.JK, AAPL)", value="BRMS.JK")
window_size = st.sidebar.number_input("Windowing (Hari ke belakang)", min_value=30, max_value=720, value=90, step=10)
forecast_days = st.sidebar.number_input("Forecasting (Hari ke depan)", min_value=7, max_value=360, value=30, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Sistem menggunakan session_state untuk mencegah reload otomatis.")

# --- 2. LOGIKA EKSEKUSI ---
if st.sidebar.button("🚀 Mulai Analisis & Forecasting"):
    # Hapus hasil sebelumnya jika pengguna memulai analisis baru
    st.session_state.clear()
    
    status_text = st.empty()
    progress_bar = st.progress(0)

    try:
        # TAHAP 1: Download Data
        status_text.info(f"Mengunduh data {ticker}...")
        df = load_stock_data(ticker)
        
        if df.empty:
            st.error("Data tidak ditemukan! Pastikan kode saham benar.")
            st.stop()
        progress_bar.progress(15)

        # TAHAP 2: Preprocessing
        status_text.info("Memproses data (Scaling & Windowing)...")
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
        status_text.info("Melatih model BiLSTM... (Mohon tunggu)")
        model = Sequential([
            Bidirectional(LSTM(units=64, return_sequences=True), input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            Bidirectional(LSTM(units=32, return_sequences=False)),
            Dropout(0.2),
            Dense(units=1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=30, callbacks=[early_stop], verbose=0)
        progress_bar.progress(70)

        # TAHAP 4: Evaluasi Test Data
        status_text.info("Menghitung metrik evaluasi...")
        test_predict = model.predict(X_test, verbose=0)
        
        test_predict_inv = scaler.inverse_transform(test_predict).flatten()
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        test_dates = df.index[-len(y_test):]

        # Menghitung Metrik
        mae = mean_absolute_error(y_test_inv, test_predict_inv)
        rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict_inv))
        mape = np.mean(np.abs((y_test_inv - test_predict_inv) / y_test_inv)) * 100
        r2 = r2_score(y_test_inv, test_predict_inv)
        progress_bar.progress(85)

        # TAHAP 5: Forecasting Iteratif
        status_text.info("Memprediksi masa depan...")
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

        # --- MENYIMPAN HASIL KE SESSION STATE ---
        # Ini adalah kunci agar web tidak lag dan data tidak hilang saat layar digeser
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
        st.error(f"Terjadi kesalahan teknis: {e}")

# --- 3. MENAMPILKAN OUTPUT DARI SESSION STATE ---
if st.session_state.get('run_success', False):
    st.success(f"Berhasil memproses {st.session_state['ticker']}!")

    # --- METRIK EVALUASI ---
    st.subheader("🎯 Metrik Akurasi Model")
    m = st.session_state['metrics']
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE (Mean Absolute Error)", f"{m['MAE']:,.2f}")
    col2.metric("RMSE (Root Mean Squared Error)", f"{m['RMSE']:,.2f}")
    col3.metric("MAPE (Mean Absolute % Error)", f"{m['MAPE']:.2f}%")
    col4.metric("R² Score", f"{m['R2']:.4f}")

    st.markdown("---")
    
    # --- GRAFIK INTERAKTIF (PLOTLY) ---
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("📊 Aktual vs Prediksi (Data Uji)")
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(x=st.session_state['test_dates'], y=st.session_state['y_test_inv'], mode='lines', name='Harga Aktual', line=dict(color='blue')))
        fig_eval.add_trace(go.Scatter(x=st.session_state['test_dates'], y=st.session_state['test_predict_inv'], mode='lines', name='Prediksi BiLSTM', line=dict(color='red', dash='dash')))
        fig_eval.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_eval, use_container_width=True)

    with col_chart2:
        st.subheader("🔮 Forecasting Masa Depan")
        fig_fore = go.Figure()
        fig_fore.add_trace(go.Scatter(x=st.session_state['hist_dates'], y=st.session_state['hist_close'], mode='lines', name='Data Historis', line=dict(color='blue')))
        fig_fore.add_trace(go.Scatter(x=st.session_state['future_dates'], y=st.session_state['forecast_results'], mode='lines+markers', name='Forecasting', line=dict(color='orange'), marker=dict(size=4)))
        fig_fore.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_fore, use_container_width=True)

    # --- TABEL FORECASTING ---
    st.markdown("---")
    st.subheader("📋 Tabel Harga Forecasting")
    df_forecast = pd.DataFrame({
        'Tanggal': st.session_state['future_dates'].strftime('%Y-%m-%d'),
        'Harga Prediksi': np.round(st.session_state['forecast_results'], 2)
    })
    # Tampilkan tabel yang bisa di-scroll dan disortir
    st.dataframe(df_forecast, use_container_width=True, hide_index=True)