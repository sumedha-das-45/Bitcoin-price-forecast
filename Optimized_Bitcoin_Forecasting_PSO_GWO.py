
import numpy as np
import pandas as pd
import yfinance as yf
import optuna
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from keras.optimizers import Adam

data = yf.download('BTC-USD', start='2017-01-01', end='2024-12-31', auto_adjust=True)[['Close']].dropna()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)
X = X.reshape((X.shape[0], X.shape[1], 1))

def objective(trial):
    units = trial.suggest_int('units', 32, 128)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 5, 20)

    model = Sequential([
        Input(shape=(X.shape[1], 1)),
        LSTM(units),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    loss = model.evaluate(X, y, verbose=0)
    return loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("Best trial:")
print(study.best_trial)
