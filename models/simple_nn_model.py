from models.nn_model import NNModel

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

import pandas as pd
import numpy as np

class SimpleNNModel(NNModel):
    def __init__(self, **kwargs):
        super().__init__('simple_nn.keras', **kwargs)

    def preprocess(self, df, target_column):
        df['ocnr_dt_date'] = pd.to_datetime(df['ocnr_dt_date'])  
        df = df.set_index('ocnr_dt_date')  
        df = df.resample(self.freq).mean(numeric_only=True).interpolate()

        df[f"lag"] = df[target_column].shift(1)
        df = df.dropna().reset_index()

        time = df['ocnr_dt_date']
        X = df.drop(columns=[target_column, 'ocnr_dt_date'])
        y = df[target_column].values.reshape(-1, 1)
        return X, y, time

    def train(self, df, target_column):
        self.X, self.y, self.time = self.preprocess(df, target_column)
        self.split()
        self.scale()
        self.build_all_windows()

        n_features = self.X.shape[2]

        self.model = keras.Sequential([
            layers.Input(shape=(self.input_width, n_features)),
            layers.LSTM(128, return_sequences=True),
            layers.LSTM(64),
            layers.Dropout(0.2),
            layers.Dense(self.forecast_horizon), 
        ])

        self.model.compile(optimizer='adam', 
                           loss='mean_squared_error', 
                           metrics=[keras.metrics.MAE])

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=50,
            validation_data=(self.X_val, self.y_val),
            callbacks=self.callbacks
        )

        self.y_pred = self.model.predict(self.X_test)

        y_true_scaled = self.y_test.reshape(-1, 6)
        y_pred_scaled = self.y_pred.reshape(-1, 6)

        y_pred_unscaled = self.scaler_y.inverse_transform(y_pred_scaled)
        y_true_unscaled = self.scaler_y.inverse_transform(y_true_scaled)

        self.metrics = {
            "MAPE": keras.metrics.MeanAbsolutePercentageError()(y_true_unscaled, y_pred_unscaled).numpy(),
            "R2": keras.metrics.R2Score()(y_true_unscaled, y_pred_unscaled).numpy(),
            "MAE": keras.metrics.MeanAbsoluteError()(y_true_scaled, y_pred_scaled).numpy()
        }

    def predict(self, df, target_column):
        X, y, time = self.preprocess(df, target_column)
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.scaler_y.transform(y)
        X, y = self.create_windows(X_scaled, y_scaled)
        y_pred_scaled = self.model.predict(X)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_true = self.scaler_y.inverse_transform(np.squeeze(y, axis=-1))
        return y_pred, y_true, time

    def load(self):
        self.model = load_model("outputs/neural_network/simple_nn.keras")