from models.nn_model import NNModel
from tensorflow import keras
from tensorflow.keras import layers, metrics

import pandas as pd

from sklearn.metrics import mean_absolute_percentage_error, r2_score

class MultiNNModel(NNModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_columns = None

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

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metrics.MeanAbsoluteError()])

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=5,
            # validation_data=(self.X_val, self.y_val),
        )

        self.y_pred = self.model.predict(self.X_test)
        
        self.metrics = []   
        for i, v in enumerate(self.target_columns):
            y_true_feat = self.y_test[:, :, i]  # shape: (batch_size, horizon)
            y_pred_feat = self.y_pred[:, :, i]  # shape: (batch_size, horizon)
        
            self.metrics.append({
                "column": v,
                "MAPE": mean_absolute_percentage_error(y_true_feat, y_pred_feat),
                "R2": r2_score(y_true_feat, y_pred_feat)
            })

        self.model.save("outputs/neural_network/multi_nn.keras")

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