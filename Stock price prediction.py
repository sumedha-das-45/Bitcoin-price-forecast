import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

ticker = 'AAPL'  
df = yf.download(ticker, start='2015-01-01', end='2024-01-01')

data = df['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

sequence_length = 60
x_train = []
y_train = []

for i in range(sequence_length, len(scaled_data)):
    x_train.append(scaled_data[i-sequence_length:i])
    y_train.append(scaled_data[i])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=32, epochs=5)

test_data = scaled_data[-(sequence_length + 30):]
x_test = []
for i in range(sequence_length, len(test_data)):
    x_test.append(test_data[i-sequence_length:i])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

plt.figure(figsize=(10, 6))
plt.title(f"{ticker} Stock Price Prediction")
plt.plot(df.index[-len(predictions):], data[-len(predictions):], label="Actual")
plt.plot(df.index[-len(predictions):], predictions, label="Predicted")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()


