from models.nn_model import NNModel
from tensorflow import keras
from tensorflow.keras import layers, metrics

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import pandas as pd
import numpy as np

class MultiNNModel(NNModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_columns = None
        self.metrics_per_col = []

    def preprocess(self, df):
        df['ocnr_dt_date'] = pd.to_datetime(df['ocnr_dt_date'])  
        df = df.set_index('ocnr_dt_date')  
        df = df.resample(self.freq).mean().interpolate()

        self.target_columns = [col for col in df.columns if col != "ocnr_dt_date"]

        for col in self.target_columns:
            df[f"{col}_lag1"] = df[col].shift(1)
        df = df.dropna().reset_index()

        time = df['ocnr_dt_date']
        X = df[[c for c in df.columns if c.endswith("_lag1")]]
        y = df[self.target_columns]
        return X, y, time

    def train(self, df):
        self.X, self.y, self.time = self.preprocess(df)
        self.split()
        self.scale()
        self.build_all_windows()

        n_features = self.y.shape[2]

        self.model = keras.Sequential([
            layers.Input(shape=(self.input_width, n_features)),
            layers.LSTM(128, return_sequences=True),
            layers.LSTM(64),
            layers.Dropout(0.2),
            layers.Dense(self.forecast_horizon * n_features),  # todas as features x horizontes
            layers.Reshape((self.forecast_horizon, n_features))
        ])

        early_stop = EarlyStopping(
            monitor='val_loss',   # pode ser 'val_loss' ou outra métrica de validação
            patience=10,          # nº de épocas sem melhora antes de parar
            restore_best_weights=True # restaura os melhores pesos ao final
        )

        checkpoint = ModelCheckpoint(
            filepath='outputs/neural_network/multi_nn.keras',
            monitor='val_loss',
            save_best_only=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,   # reduz a learning rate pela metade
            patience=5,   # espera 5 épocas sem melhora
            min_lr=1e-6
        )

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=[keras.losses.MAPE])

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=100,
            validation_data=(self.X_val, self.y_val),
            callbacks=[early_stop, checkpoint, reduce_lr]
        )

        self.y_pred = self.model.predict(self.X_test)
        
        self.metrics_per_col = []   
        y_true_unscaled = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 3))
        y_pred_unscaled = self.scaler_y.inverse_transform(self.y_pred.reshape(-1, 3))
        for i, v in enumerate(self.target_columns):
            y_true_feat = y_true_unscaled[:, i:i+1]
            y_pred_feat = y_pred_unscaled[:, i:i+1]
        
            self.metrics_per_col.append({
                "column": v,
                "MAPE": keras.metrics.MeanAbsolutePercentageError()(y_true_feat, y_pred_feat).numpy(),
                "R2": keras.metrics.R2Score()(y_true_feat, y_pred_feat).numpy()
            })

        self.metrics = {
            "MAPE": np.mean([m["MAPE"] for m in self.metrics_per_col]),
            "R2": np.mean([m["R2"] for m in self.metrics_per_col])
        }

        # self.model.save("outputs/neural_network/multi_nn.keras")

    def predict(self, df):
        X, y, time = self.preprocess(df)
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.scaler_y.transform(y)
        X, _ = self.create_windows(X_scaled, y_scaled)
        y_pred_scaled = self.model.predict(X)
        y_pred_scaled = y_pred_scaled[:, 0, :]
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred, time

    def load(self):
        self.model.load_model("outputs/neural_network/multi_nn.keras")