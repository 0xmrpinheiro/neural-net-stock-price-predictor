import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tf_keras
from tf_keras.models import load_model
from pathlib import Path
import plotly.graph_objs as go

# --- Caching helpers ---
@st.cache_resource(show_spinner=False)
def load_trained_model(model_path: str):
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def _flatten_yf_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten MultiIndex columns returned by yfinance.
    - If only one ticker exists in the columns' last level, drop that level.
    - Else, select the block for the requested ticker (xs).
    Finally, ensure standard OHLCV columns in float dtype.
    """
    if isinstance(df.columns, pd.MultiIndex):
        last_level = df.columns.get_level_values(-1)
        unique_last = pd.unique(last_level)
        if len(unique_last) == 1:
            df.columns = df.columns.get_level_values(0)
        else:
            # pick the requested ticker if present
            if ticker in unique_last:
                df = df.xs(ticker, axis=1, level=-1)
            else:
                # take the first block as a fallback
                df = df.xs(unique_last[0], axis=1, level=-1)

    # Standardize column case
    rename_map = {c: c.title() for c in df.columns}
    df = df.rename(columns=rename_map)

    # Keep only expected columns if present
    expected = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[expected]

    # Coerce numerics
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clean index
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]  # drop NaT rows
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Drop rows that are completely NaN
    df = df.dropna(how="all")

    return df

@st.cache_data(show_spinner=False, ttl=3600)
def yf_download_robust(ticker: str, start, end):
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return pd.DataFrame()

    tries = [
        {"interval": "1d", "group_by": "column", "auto_adjust": True},
        {"interval": "1wk", "group_by": "column", "auto_adjust": True},
    ]
    for params in tries:
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                threads=False,
                **params,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = _flatten_yf_columns(df, ticker)
                if not df.empty and "Close" in df.columns:
                    return df
        except Exception as e:
            st.warning(f"Download failed with {params}: {e}")
    return pd.DataFrame()

def calculate_moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

def create_dataset(data, look_back=100):
    X = []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
    return np.array(X)

def display_charts(stock_data):
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
    if 'MA100' in stock_data.columns:
        fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA100'], mode='lines', name='MA100'))
    st.plotly_chart(fig1, width='stretch')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
    if 'MA100' in stock_data.columns:
        fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA100'], mode='lines', name='MA100'))
    if 'MA200' in stock_data.columns:
        fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA200'], mode='lines', name='MA200'))
    st.plotly_chart(fig2, width='stretch')

    # Candlestick only if all OHLC present
    ohlc_cols = {'Open', 'High', 'Low', 'Close'}
    if ohlc_cols.issubset(stock_data.columns):
        candlestick_fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                                         open=stock_data['Open'], high=stock_data['High'],
                                                         low=stock_data['Low'], close=stock_data['Close'])])
        candlestick_fig.update_layout(title='Candlestick Chart')
        st.plotly_chart(candlestick_fig, width='stretch')

    if 'Volume' in stock_data.columns:
        volume_fig = go.Figure(data=[go.Bar(x=stock_data.index, y=stock_data['Volume'])])
        volume_fig.update_layout(title='Volume Plot')
        st.plotly_chart(volume_fig, width='stretch')

def prepare_and_predict(stock_data, model):
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_values = np.array(stock_data['Close']).reshape(-1, 1)
    scaled_data = scaler.fit_transform(close_values)
    x_pred = create_dataset(scaled_data)
    if x_pred.size == 0:
        return scaler, np.array([])
    x_pred = x_pred.reshape((x_pred.shape[0], x_pred.shape[1], 1))
    y_pred = model.predict(x_pred, verbose=0)
    y_pred = scaler.inverse_transform(y_pred)
    return scaler, y_pred

def display_prediction_chart(stock_data, y_pred):
    if y_pred.size == 0:
        st.info("Not enough data to generate predictions (need at least 100 closing prices).")
        return
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=stock_data.index[100:], y=stock_data['Close'][100:], mode='lines', name='Actual Price'))
    fig3.add_trace(go.Scatter(x=stock_data.index[100:], y=y_pred.flatten(), mode='lines', name='Predicted Price'))
    fig3.update_layout(title='Original vs Predicted Prices')
    st.plotly_chart(fig3, width='stretch')

def display_evaluation_metrics(stock_data, y_pred):
    if y_pred.size == 0:
        return
    y_true = stock_data['Close'].values[100:]
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    st.subheader('Model Evaluation')
    st.write(f'Mean Absolute Error (MAE): {mae:.2f}')
    st.write(f'Mean Squared Error (MSE): {mse:.2f}')

def perform_and_display_forecasting(stock_data, model, scaler):
    forecast_period_days = 30
    last_100_days = stock_data['Close'].tail(100).values.reshape(-1, 1)
    if last_100_days.shape[0] < 100:
        st.info("Not enough history (need last 100 days) to produce a 30-day forecast.")
        return
    last_100_days_scaled = scaler.transform(last_100_days)

    forecasted_prices_scaled = []
    last_100_days_scaled_list = last_100_days_scaled.tolist()

    for _ in range(forecast_period_days):
        x_forecast = np.array(last_100_days_scaled_list[-100:]).reshape(1, 100, 1)
        y_forecast_scaled = model.predict(x_forecast, verbose=0)
        forecasted_prices_scaled.append(y_forecast_scaled[0, 0])
        last_100_days_scaled_list.append([y_forecast_scaled[0, 0]])

    forecasted_prices_scaled = np.array(forecasted_prices_scaled).reshape(-1, 1)
    forecasted_prices = scaler.inverse_transform(forecasted_prices_scaled)

    forecast_dates = pd.date_range(start=stock_data.index[-1] + timedelta(days=1), periods=forecast_period_days, freq='D')
    forecast_df = pd.DataFrame(data=forecasted_prices, index=forecast_dates, columns=['Forecast'])

    st.subheader('30-Day Forecast')
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecasted Price'))
    fig4.update_layout(title='Stock Price Forecast for the Next 30 Days')
    st.plotly_chart(fig4, width='stretch')

def main():
    st.sidebar.title('Aeon Stock Price Predict')
    stock_symbol = st.sidebar.text_input('Enter Stock Ticker Symbol (e.g., MSFT):').strip().upper()
    start_date = st.sidebar.date_input('Select Start Date:', datetime(2000, 1, 1))
    end_date = st.sidebar.date_input('Select End Date:', datetime.now())
    selected_model = st.sidebar.radio("Select Model", ("Neural Network",))

    if stock_symbol:
        with st.spinner('Fetching stock data...'):
            stock_data = yf_download_robust(stock_symbol, start=start_date, end=end_date)

        if not stock_data.empty:
            if len(stock_data) < 120:
                st.error("Not enough rows returned for analysis. Try an earlier start date or a broader interval.")
                return

            st.subheader(f'Stock Data for {stock_symbol}')
            st.write(stock_data.tail())

            stock_data['MA100'] = calculate_moving_average(stock_data['Close'], 100)
            stock_data['MA200'] = calculate_moving_average(stock_data['Close'], 200)

            display_charts(stock_data)

            if selected_model == "Neural Network":
                with st.spinner('Loading the prediction model...'):
                    model_path = Path('Models') / 'neural_forecaster.keras'
                    if not model_path.exists():
                        st.error(f"Model file not found at {model_path}. Please add it to the repo.")
                        return
                    model = load_trained_model(str(model_path))

                if model is not None:
                    scaler, y_pred = prepare_and_predict(stock_data, model)
                    display_prediction_chart(stock_data, y_pred)
                    display_evaluation_metrics(stock_data, y_pred)
                    perform_and_display_forecasting(stock_data, model, scaler)
                else:
                    st.error("Model could not be loaded. Please check the model path.")
        else:
            st.error("Failed to fetch stock data. Please check the ticker symbol and date range.")
    else:
        st.info("Enter a stock ticker symbol to begin.")

if __name__ == '__main__':
    main()
