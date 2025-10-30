from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        self.metrics = {}
        self.X = []
        self.y = []
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        self.y_pred = None

        self.best_params = None
        self.best_score = None


    @abstractmethod
    def preprocess(self, df: pd.DataFrame, target_column: str = None):
        pass

    @abstractmethod
    def train(self, df: pd.DataFrame, target_column: str = None):
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame, target_column: str = None):
        pass

    @abstractmethod
    def load(self):
        pass 

    @abstractmethod
    def save(self):
        pass
