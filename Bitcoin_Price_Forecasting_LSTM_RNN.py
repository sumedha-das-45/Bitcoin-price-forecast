
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN

data = yf.download('BTC-USD', start='2017-01-01', end='2024-12-31')
data = data[['Close']]  
data.dropna(inplace=True)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

X, y = [], []
sequence_len = 60

for i in range(sequence_len, len(scaled_data)):
    X.append(scaled_data[i-sequence_len:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))  

model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X, y, epochs=10, batch_size=32)

model_rnn = Sequential([
    SimpleRNN(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    SimpleRNN(50),
    Dense(1)
])
model_rnn.compile(optimizer='adam', loss='mean_squared_error')
model_rnn.fit(X, y, epochs=10, batch_size=32)

last_sequence = scaled_data[-sequence_len:]
forecast = []

input_seq = last_sequence.reshape(1, sequence_len, 1)

for _ in range(30):  
    pred = model_lstm.predict(input_seq)[0][0]
    forecast.append(pred)
    input_seq = np.append(input_seq[:,1:,:], [[[pred]]], axis=1)

forecast_prices = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

plt.figure(figsize=(10,6))
plt.plot(data.index[-200:], scaler.inverse_transform(scaled_data[-200:]), label='Actual Price')
plt.plot(pd.date_range(data.index[-1], periods=31, freq='D')[1:], forecast_prices, label='Forecast (LSTM)', linestyle='--')
plt.title("Bitcoin Price Forecast using LSTM")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
