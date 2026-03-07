import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Stock Forecasting BiLSTM",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Forecasting (BiLSTM)")
st.markdown("Forecasting harga saham menggunakan **Bidirectional LSTM**")


# --------------------------------------------------
# SIDEBAR INPUT
# --------------------------------------------------

st.sidebar.header("⚙️ Forecast Settings")

ticker = st.sidebar.text_input(
    "Stock Ticker",
    value="BRMS.JK"
)

window_size = st.sidebar.slider(
    "Window Size (Days)",
    30, 120, 90
)

forecast_days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    7, 90, 30
)

run_button = st.sidebar.button("Run Forecast")


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_stock_data(ticker):
    df = yf.download(ticker, start="2020-01-01")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Close']]
    df.dropna(inplace=True)

    return df


# --------------------------------------------------
# WINDOWING FUNCTION
# --------------------------------------------------

def create_dataset(data, window_size):

    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)


# --------------------------------------------------
# MODEL BUILDER
# --------------------------------------------------

def build_model(window_size):

    model = Sequential()

    model.add(
        Bidirectional(
            LSTM(64, return_sequences=True),
            input_shape=(window_size, 1)
        )
    )

    model.add(Dropout(0.2))

    model.add(
        Bidirectional(
            LSTM(32)
        )
    )

    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    r2 = r2_score(y_true, y_pred)

    return mae, rmse, mape, r2


# --------------------------------------------------
# FORECASTING FUNCTION
# --------------------------------------------------

def forecast_future(model, last_window, scaler, days):

    forecast = []

    current_window = last_window.copy()

    for _ in range(days):

        pred = model.predict(current_window, verbose=0)

        forecast.append(pred[0, 0])

        pred = pred.reshape(1,1,1)

        current_window = np.append(
            current_window[:,1:,:],
            pred,
            axis=1
        )

    forecast = scaler.inverse_transform(
        np.array(forecast).reshape(-1,1)
    )

    return forecast


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

if run_button:

    with st.spinner("Downloading stock data..."):

        df = load_stock_data(ticker)

        if df.empty:
            st.error("Stock ticker not found.")
            st.stop()


    data = df.values

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data)


    # windowing
    X, y = create_dataset(scaled_data, window_size)

    X = X.reshape(X.shape[0], X.shape[1], 1)


    # train test split
    train_size = int(len(X) * 0.85)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]


    with st.spinner("Training BiLSTM model..."):

        model = build_model(window_size)

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=30,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )


    # prediction
    pred = model.predict(X_test)

    pred_inv = scaler.inverse_transform(pred)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))


    # metrics
    mae, rmse, mape, r2 = calculate_metrics(
        y_test_inv,
        pred_inv
    )


    # --------------------------------------------------
    # METRICS DISPLAY
    # --------------------------------------------------

    st.subheader("📊 Model Accuracy")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("MAPE", f"{mape:.2f}%")
    col4.metric("R²", f"{r2:.3f}")


    # --------------------------------------------------
    # ACTUAL VS PREDICTED CHART
    # --------------------------------------------------

    st.subheader("Actual vs Predicted")

    test_dates = df.index[-len(y_test_inv):]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=y_test_inv.flatten(),
            mode='lines',
            name='Actual'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=pred_inv.flatten(),
            mode='lines',
            name='Predicted'
        )
    )

    fig.update_layout(
        height=450,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)


    # --------------------------------------------------
    # FORECAST FUTURE
    # --------------------------------------------------

    last_window = scaled_data[-window_size:]
    last_window = last_window.reshape(1, window_size, 1)

    forecast = forecast_future(
        model,
        last_window,
        scaler,
        forecast_days
    )

    last_date = df.index[-1]

    future_dates = pd.bdate_range(
        last_date + timedelta(days=1),
        periods=forecast_days
    )


    # --------------------------------------------------
    # FORECAST CHART
    # --------------------------------------------------

    st.subheader("Future Forecast")

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=df.index[-150:],
            y=df['Close'][-150:],
            mode='lines',
            name='Historical'
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=future_dates,
            y=forecast.flatten(),
            mode='lines+markers',
            name='Forecast'
        )
    )

    fig2.update_layout(
        height=450,
        template="plotly_white"
    )

    st.plotly_chart(fig2, use_container_width=True)


    # --------------------------------------------------
    # FORECAST TABLE
    # --------------------------------------------------

    st.subheader("Forecast Table")

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": np.round(
            forecast.flatten(),
            2
        )
    })

    st.dataframe(
        forecast_df,
        use_container_width=True
    )