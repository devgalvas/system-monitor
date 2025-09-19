from models.nn_model import NNModel

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

class SuperNNModel(NNModel):
    def __init__(self, **kwargs):
        super().__init__('super_nn.keras', **kwargs)
        self.X = np.empty((0, self.input_width, 1))
        self.y = np.empty((0, self.forecast_horizon, 1))
        self.namespaces_ids = np.empty((0,), dtype=int)
        self.time = np.empty((0,), dtype='datetime64[ns]')

    def preprocess(self, df, target_column, training=True):
        df['ocnr_dt_date'] = pd.to_datetime(df['ocnr_dt_date'])  

        df[f"lag"] = df[target_column].shift(1)
        df = df.dropna()

        namespace_id = []
        if training:
            namespace_id = self.namespace_encoder.transform(df['ocnr_tx_namespace'])

        time = df['ocnr_dt_date']
        X = df.drop(columns=[target_column, 'ocnr_dt_date', 'ocnr_tx_namespace'])
        y = df[target_column].values.reshape(-1, 1)
        return X, y, time, namespace_id

    def split(self):
        n = self.X.shape[0]
        train_end = int(n * 0.7)
        val_end   = int(n * 0.9)

        self.X_train_raw, self.y_train_raw, self.ns_train = (
            self.X[:train_end], self.y[:train_end], self.namespaces_ids[:train_end]
        )
        self.X_val_raw, self.y_val_raw, self.ns_val = (
            self.X[train_end:val_end], self.y[train_end:val_end], self.namespaces_ids[train_end:val_end]
        )
        self.X_test_raw, self.y_test_raw, self.ns_test = (
            self.X[val_end:], self.y[val_end:], self.namespaces_ids[val_end:]
        )

    def create_windows(self, X, y, ns_ids):
        df_length = X.shape[0]
        X_windows, y_windows, ns_windows = [], [], []
        for start in range(0, df_length - self.input_width - self.forecast_horizon + 1):
            end = start + self.input_width
            Xw = X[start:end]
            yw = y[end:end + self.forecast_horizon]
            nsw = ns_ids[end-1]  # namespace do último timestamp da janela
            X_windows.append(Xw)
            y_windows.append(yw)
            ns_windows.append(nsw)
        return np.array(X_windows), np.array(y_windows), np.array(ns_windows)

    def mix_namespaces(self):
        unique_ns = np.unique(self.namespaces_ids)
        groups = [np.where(self.namespaces_ids == ns)[0] for ns in unique_ns]
        max_len = max(len(g) for g in groups)

        interleaved_idx = []
        for i in range(max_len):
            for g in groups:
                if i < len(g):  
                    interleaved_idx.append(g[i])

        interleaved_idx = np.array(interleaved_idx)

        self.X = self.X[interleaved_idx]
        self.y = self.y[interleaved_idx]
        self.namespaces_ids = self.namespaces_ids[interleaved_idx]

    def split_by_namespace(self, X, y, nms_ids):
        """Separa X, y e ids em dicionários por namespace_id"""
        unique_ns = np.unique(nms_ids)
        groups = {}

        for ns in unique_ns:
            idx = np.where(nms_ids == ns)[0]
            groups[ns] = {
                "X": X[idx],
                "y": y[idx],
                "ids": nms_ids[idx]
            }

        return groups

    def train(self, all_dfs, target_column):
        self.namespace_encoder = LabelEncoder()
        self.namespace_encoder.fit(pd.concat([df['ocnr_tx_namespace'] for df in all_dfs]))
        for df in all_dfs:
            X, y, time, namespaces_ids = self.preprocess(df, target_column)
            X, y, namespaces_ids = self.create_windows(X, y, namespaces_ids)
            self.X = np.concatenate([self.X, X], axis=0)
            self.y = np.concatenate([self.y, y], axis=0)
            self.namespaces_ids = np.concatenate([self.namespaces_ids, namespaces_ids], axis=0)
            self.time = np.concatenate([self.time, time], axis=0)
        self.mix_namespaces()
        self.X = np.squeeze(self.X, axis=-1)
        self.y = np.squeeze(self.y, axis=-1)
        self.split()
        self.scale()

        vocab_size = len(self.namespace_encoder.classes_)
        embed_dim = 8

        input_series = layers.Input(shape=(self.input_width, 1), name="series_input")
        self.lstm_model = layers.LSTM(128, return_sequences=True)(input_series)
        self.lstm_model = layers.LSTM(128)(self.lstm_model)
        self.lstm_model = layers.Dropout(0.2)(self.lstm_model)

        namespace_input = layers.Input(shape=(1,), name="namespace_input")
        self.embedding_model = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(namespace_input)
        self.embedding_model = layers.Flatten()(self.embedding_model)

        concat = layers.concatenate([self.lstm_model, self.embedding_model])
        out = layers.Dense(self.forecast_horizon)(concat)
        self.model = keras.Model(inputs=[input_series, namespace_input], outputs=out)

        self.model.compile(optimizer='adam', 
                           loss='mean_squared_error', 
                           metrics=[keras.metrics.MAE])

        self.history = self.model.fit(
            [self.X_train, self.ns_train],   # duas entradas!
            self.y_train,
            epochs=1,
            validation_data=([self.X_val, self.ns_val], self.y_val),
            callbacks=self.callbacks
        )

        groups = self.split_by_namespace(self.X_test, self.y_test, self.ns_test)

        self.metrics = []
        for g in groups.values():
            print(g['X'])
            self.y_pred = self.model.predict([g['X'], g['ids']])

            y_true_scaled = g['y'].reshape(-1, 6)
            y_pred_scaled = self.y_pred.reshape(-1, 6)

            y_pred_unscaled = self.scaler_y.inverse_transform(y_pred_scaled)
            y_true_unscaled = self.scaler_y.inverse_transform(y_true_scaled)

            ns = self.namespace_encoder.inverse_transform(g['ids'])[0]

            self.metrics.append({
                "namespace": ns,
                "MAPE": keras.metrics.MeanAbsolutePercentageError()(y_true_unscaled, y_pred_unscaled).numpy(),
                "R2": keras.metrics.R2Score()(y_true_unscaled, y_pred_unscaled).numpy(),
                "MAE": keras.metrics.MeanAbsoluteError()(y_true_scaled, y_pred_scaled).numpy()
            })

    def predict(self, df, target_column):
        X, y, time = self.preprocess(df, target_column, training=False)
        X, y = self.create_windows(X, y)
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred, time

    def load(self):
        self.model.load_model(f"outputs/neural_network/{self.saving_file}")