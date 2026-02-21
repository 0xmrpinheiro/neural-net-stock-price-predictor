# Aeon Quant — Stock Price Prediction App

> **Disclaimer**: Stock price prediction is inherently uncertain. This app is for educational purposes only and **not** investment advice.

A Streamlit application that predicts stock prices using an LSTM model and visualizes stock data with interactive Plotly charts.

![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)

## Quick Links

- **Live Demo**: <https://neuralstocks.streamlit.app/>
- **Last Model Training**: October 23, 2024

## Overview

Aeon Quant provides a simple UI for technical analysis and predictive modeling of stock prices. The app uses a TensorFlow LSTM to forecast future prices from historical data.

## Features

- **Historical data via Yahoo Finance** with robust download (handles MultiIndex columns and timezone quirks)
- **Charts**: Close price with MA100 & MA200, candlesticks (when OHLC available), and volume
- **Modeling**: LSTM-based predictions, actual vs. predicted comparison, MAE/MSE metrics, and a 30?day forecast

## Architecture

- **Frontend**: Streamlit
- **Data**: Pandas, NumPy, scikit-learn
- **Model**: TensorFlow (LSTM)
- **Viz**: Plotly

## Installation

### Prerequisites

- Python 3.9+
- `pip`

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-org>/aeon-stock-price-predict.git
   cd aeon-stock-price-predict
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Option A) **Use an existing model file**  
   Place the trained model at:

   ```
   Models/neural_forecaster.keras
   ```

   (Option B) **Train a new model locally**  

   ```bash
   python neural_network_forecaster.py
   ```

   This will save `Models/neural_forecaster.keras`.

4. Run the app:

   ```bash
   streamlit run streamlit_app.py
   ```

5. Open `http://localhost:8501` in your browser.

## Configuration

No manual environment variables required. The app suppresses TF logs internally (`TF_CPP_MIN_LOG_LEVEL=3`).

## Troubleshooting

- **Empty/odd charts or “MSFT” headers under columns**: This happens when Yahoo returns MultiIndex columns. The app now flattens/cleans columns automatically.
- **“No timezone found, symbol may be delisted”**: Caused by Yahoo metadata. Using a recent `yfinance` version and our robust downloader resolves it.

## Dependencies

> Production pins used by the app

```
streamlit==1.25.0
scikit-learn==1.3.0
tensorflow==2.13.0
plotly==5.15.0
numpy==1.23.5
pandas==1.5.3
yfinance>=0.2.40
```

## License

GPL-3.0 — see [LICENSE](LICENSE).

## Contributing

PRs welcome!
