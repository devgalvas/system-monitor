import tensorflow as tf
from tensorflow import keras

import numpy as np

class DWA_Callback(keras.callbacks.Callback):
    def __init__(self, X_val, y_val, T=2.0, n_features=3):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.T = T
        self.n_features = n_features
        self.prev_losses = [[] for _ in range(n_features)]

    def on_epoch_end(self, epoch, logs=None):
        # predição sobre o conjunto de validação
        y_pred = self.model.predict(self.X_val, verbose=0)
        
        # calcular MSE por feature
        losses = tf.reduce_mean(tf.square(self.y_val - y_pred), axis=[0,1]).numpy()
        
        # atualizar histórico
        for i in range(self.n_features):
            self.prev_losses[i].append(losses[i])
        
        # calcular pesos DWA se houver histórico suficiente
        if len(self.prev_losses[0]) >= 3:
            w = []
            for i in range(self.n_features):
                l_prev = self.prev_losses[i][-2]
                l_prevprev = self.prev_losses[i][-3]
                w.append(l_prev / (l_prevprev + 1e-8))
            w = np.array(w)
            exp_w = np.exp(w / self.T)
            weights = exp_w / np.sum(exp_w)
            print(f"Epoch {epoch+1} - DWA weights: {weights}")


