import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import plotly.graph_objs as go
from pathlib import Path

# Set a seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define stock data retrieval parameters
start_date = '2000-01-01'
end_date = '2024-10-23'  # fixed invalid date from original
stock_symbol = 'MSFT'

# Download stock data
data = yf.download(stock_symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)

# Reset index and drop missing values
data.reset_index(inplace=True)
data.dropna(inplace=True)

# Calculate moving averages
data['MA100'] = data['Close'].rolling(100).mean()
data['MA200'] = data['Close'].rolling(200).mean()

# Split data into train and test sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Apply MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data[['Close']])
test_scaled = scaler.transform(test_data[['Close']])

# Prepare training and test sets
def create_dataset(data, look_back=100):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

x_train, y_train = create_dataset(train_scaled)
x_test, y_test = create_dataset(test_scaled)

# Reshape input to be [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# EarlyStopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model (use a validation split to monitor validation loss)
history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Ensure output directory exists and save the trained model
out_dir = Path('Models')
out_dir.mkdir(parents=True, exist_ok=True)
model_path = out_dir / 'neural_forecaster.keras'
model.save(model_path)

# Predictions on test set
y_pred = model.predict(x_test, verbose=0)

# Inverse transform the predictions and actual values
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("Neural Network Mean Absolute Error (MAE):", mae)
print("Neural Network Mean Squared Error (MSE):", mse)

# Plot original vs predicted prices using Plotly
test_dates = test_data['Date'][100:]  # Adjusting date alignment for plotting
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=test_dates, y=y_test.flatten(), mode='lines', name='Original Price'))
fig3.add_trace(go.Scatter(x=test_dates, y=y_pred.flatten(), mode='lines', name='Predicted Price'))
fig3.update_layout(
    title='Neural Network - Original vs Predicted Prices',
    xaxis_title='Date',
    yaxis_title='Price'
)
# Note: In headless environments this won't render, but keeping for local use
# fig3.show()
