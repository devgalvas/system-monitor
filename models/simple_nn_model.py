from models.nn_model import NNModel

from tensorflow import keras
from tensorflow.keras import layers, metrics

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

class SimpleNNModel(NNModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(self, df, target_column):
        df['ocnr_dt_date'] = pd.to_datetime(df['ocnr_dt_date'])  
        df = df.set_index('ocnr_dt_date')  
        df = df.resample(self.freq).mean().interpolate()

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

        self.model.compile(optimizer='adam', loss='mean_squared_error', 
                           metrics=[metrics.MeanAbsoluteError(), metrics.R2Score()])

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=5,
            validation_data=(self.X_val, self.y_val),
        )

        self.y_pred = self.model.predict(self.X_test)
        
        pred_first_step = self.y_pred[:, 0] 
        y_test_first_step = self.y_test[:, 0]

        self.metrics = {
            "MAPE": mean_absolute_error(y_test_first_step, pred_first_step),
            "R2": r2_score(y_test_first_step, pred_first_step)
        }

        self.model.save("outputs/neural_network/simple_nn.keras")

    def predict(self, df, target_column):
        X, y, time = self.preprocess(df, target_column)
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.scaler_y.transform(y)
        X, _ = self.create_windows(X_scaled, y_scaled)
        y_pred_scaled = self.model.predict(X)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred, time

    def load(self):
        self.model.load_model("outputs/neural_network/simple_nn.keras")