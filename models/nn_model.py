from models.base_model import BaseModel

from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import numpy as np

class NNModel(BaseModel):
    def __init__(self, saving_file, **kwargs):
        super().__init__(**kwargs)
        self.freq = "5min"
        self.input_width = 12
        self.forecast_horizon = 6
        self.history = None

        self.X_train_raw = None
        self.y_train_raw = None
        self.X_val_raw = None
        self.y_val_raw = None 
        self.X_test_raw = None
        self.y_test_raw = None

        self.saving_file = saving_file

        self.callbacks = [EarlyStopping(
            monitor='val_loss',   # pode ser 'val_loss' ou outra métrica de validação
            patience=10,          # nº de épocas sem melhora antes de parar
            restore_best_weights=True # restaura os melhores pesos ao final
        ), ModelCheckpoint(
            filepath=f'outputs/neural_network/{self.saving_file}',
            monitor='val_loss',
            save_best_only=True
        ), ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,   # reduz a learning rate pela metade
            patience=5,   # espera 5 épocas sem melhora
            min_lr=1e-6
        )]

    def split(self):
        n = self.X.shape[0]
        train_end = int(n * 0.7)
        val_end   = int(n * 0.9)

        self.X_train_raw, self.y_train_raw = self.X[:train_end], self.y[:train_end]
        self.X_val_raw,   self.y_val_raw   = self.X[train_end:val_end], self.y[train_end:val_end]
        self.X_test_raw,  self.y_test_raw  = self.X[val_end:], self.y[val_end:]

    def scale(self):
        scaler_X = StandardScaler().fit(self.X_train_raw)
        scaler_y = StandardScaler().fit(self.y_train_raw)

        self.X_train = scaler_X.transform(self.X_train_raw)
        self.X_val   = scaler_X.transform(self.X_val_raw)
        self.X_test  = scaler_X.transform(self.X_test_raw)

        self.y_train = scaler_y.transform(self.y_train_raw)
        self.y_val   = scaler_y.transform(self.y_val_raw)
        self.y_test  = scaler_y.transform(self.y_test_raw)

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
