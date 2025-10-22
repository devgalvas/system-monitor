from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

class SafeLabelEncoder(LabelEncoder):
    def fit(self, y):
        super().fit(np.append(y, "<UNK>"))
        return self

    def transform_safe(self, y):
        classes = set(self.classes_)
        return np.array([c if c in classes else "<UNK>" for c in y])

    def transform(self, y):
        return super().transform(self.transform_safe(y))