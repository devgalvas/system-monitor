from models.base_model import BaseModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler

class SimpleNNModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.freq = "5min"
        self.input_width = 12
        self.forecast_horizon = 6
        self.history = None

    def preprocess(self, df, target_column):
        df['ocnr_dt_date'] = pd.to_datetime(df['ocnr_dt_date'])  
        df = df.set_index('ocnr_dt_date')  
        df = df.resample(self.freq).mean().interpolate()

        df[f"lag"] = df[target_column].shift(1)
        df = df.dropna().reset_index()

        time = df['ocnr_dt_date']
        X = df.drop(columns=[target_column, 'ocnr_dt_date'])
        y = df[target_column]
        return X, y, time

    def split(self):
        n = self.X.shape[0]
        train_end = int(n * 0.7)
        val_end   = int(n * 0.75)

        X_train_raw, y_train_raw = self.X[:train_end], self.y[:train_end]
        X_val_raw,   y_val_raw   = self.X[train_end:val_end], self.y[train_end:val_end]
        X_test_raw,  y_test_raw  = self.X[val_end:], self.y[val_end:]

        scaler_X = StandardScaler().fit(X_train_raw)
        scaler_y = StandardScaler().fit(y_train_raw.values.reshape(-1, 1))

        self.X_train = scaler_X.transform(X_train_raw)
        self.X_val   = scaler_X.transform(X_val_raw)
        self.X_test  = scaler_X.transform(X_test_raw)

        self.y_train = scaler_y.transform(y_train_raw.values.reshape(-1, 1)).flatten()
        self.y_val   = scaler_y.transform(y_val_raw.values.reshape(-1, 1)).flatten()
        self.y_test  = scaler_y.transform(y_test_raw.values.reshape(-1, 1)).flatten()

        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def create_windows(self, X, y):
        df_length = X.shape[0]
        X_windows, y_windows = [], []
        for start in range(0, df_length - self.input_width - self.forecast_horizon + 1):
            end = start + self.input_width
            Xw = X[start:end]
            yw = y[end:end + self.forecast_horizon]
            X_windows.append(Xw)
            y_windows.append(yw)
        return np.array(X_windows), np.array(y_windows)

    def build_all_windows(self):
        self.X, self.y = self.create_windows(self.X, self.y)
        self.X_train, self.y_train = self.create_windows(self.X_train, self.y_train)
        self.X_val, self.y_val = self.create_windows(self.X_val, self.y_val)
        self.X_test, self.y_test = self.create_windows(self.X_test, self.y_test)

    def train(self, df, target_column):
        self.X, self.y, self.time = self.preprocess(df, target_column)
        self.split()
        self.build_all_windows()

        n_features = self.X.shape[2]

        self.model = keras.Sequential([
            layers.Input(shape=(self.input_width, n_features)),
            layers.LSTM(128, return_sequences=True),
            layers.LSTM(64),
            layers.Dropout(0.2),
            layers.Dense(self.forecast_horizon * n_features),  # todas as features x horizontes
            layers.Reshape((self.forecast_horizon, n_features))
        ])

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metrics.MeanAbsoluteError()])

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=5,
            # validation_data=(self.X_val, self.y_val),
        )

        self.y_pred = self.model.predict(self.X_test)
        
        pred_first_step = self.y_pred[:, 0] 
        y_test_first_step = self.y_test[:, 0]

        self.metrics = {
            "MAPE": mean_absolute_percentage_error(y_test_first_step, pred_first_step),
            "R2": r2_score(y_test_first_step, pred_first_step)
        }

        # self.model.save("outputs/neural_network/nn.keras")

    def predict(self, df, target_column):
        X, y, time = self.preprocess(df, target_column)
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.scaler_y.transform(y.values.reshape(-1, 1))
        X, _ = self.create_windows(X_scaled, y_scaled)
        y_pred_scaled = self.model.predict(X)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, self.forecast_horizon))
        return y_pred, time

    def load(self):
        pass

    def grid_search(self, df, target_column):
        pass