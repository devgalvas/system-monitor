from models.base_model import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
import xgboost as xgb

class XGBoostModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.time = None
        self.test_time = None

    def preprocess(self, df, target_column, lags=[1]):
        df['ocnr_dt_date'] = pd.to_datetime(df['ocnr_dt_date'])
        df["month"] = df["ocnr_dt_date"].dt.month
        df["day"] = df["ocnr_dt_date"].dt.day
        df["weekday"] = df["ocnr_dt_date"].dt.weekday
        df["hour"] = df["ocnr_dt_date"].dt.hour
        df["minute"] = df["ocnr_dt_date"].dt.minute
        df["second"] = df["ocnr_dt_date"].dt.second

        for lag in lags:
            df[f'lag_{lag}'] = df[target_column].shift(lag)
        df = df.dropna()

        time = df['ocnr_dt_date'].loc[df.index]
        X = df.drop(columns=[target_column, 'ocnr_dt_date'])
        y = df[target_column]
        return X, y, time
    
    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.20, shuffle=False)
        self.test_time = self.time.loc[self.X_test.index]

    def train(self, df, target_column, lags=[1]):
        self.X, self.y, self.time = self.preprocess(df, target_column, lags)
        self.split()

        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=50,
            max_depth=3,
            random_state=42,
        )
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

        self.metrics = {
            "MAPE": mean_absolute_percentage_error(self.y_test, self.y_pred),
            "R2": r2_score(self.y_test, self.y_pred)
        }

    def predict(self, df, target_column):
        X, _, time = self.preprocess(df, target_column)
        return self.model.predict(X), time

    def grid_search(self, df, target_column, scoring='r2'):
        self.X, self.y, self.time = self.preprocess(df, target_column)
        self.split()
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 6],
        }

        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='r2',
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(self.X_train, self.y_train)

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_

    def search_lag(self, df, target_column):
        r2 = []
        mape = []
        lags = range(1,50)
        for i in lags:
            self.train(df, target_column, lags=range(1,i))
            r2.append(self.metrics['R2'])
            mape.append(self.metrics['MAPE'])

        return r2, mape, lags


    def load(self):
        pass

    def save(self):
        pass