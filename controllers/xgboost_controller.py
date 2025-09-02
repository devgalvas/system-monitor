from views.time_series_view import TimeSeriesView 
from controllers.base_controllers import BaseController
from models.xgboost_model import XGBoostModel

import pandas as pd

class XGBoostController(BaseController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def test_model(self, namespace, view_name):
        self.dl.createNamespaceView(namespace, view_name)
        df = self.dl.query_to_db(f"""
            SELECT ocnr_dt_date, ocnr_nm_result FROM {view_name} 
            WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE'
            AND EXTRACT(MONTH FROM ocnr_dt_date) = 11
            ORDER BY ocnr_dt_date
        """)

        model = XGBoostModel()
        model.train(df, 'ocnr_nm_result')
        print(model.metrics)

        pred, pred_time = model.predict(df, 'ocnr_nm_result')
        ts_view = TimeSeriesView(dir_path = 'outputs/xgboost/')
        ts_view.plot_model_evolution(pred_time, pred,
                                    model.time, model.y, filename='nov_xgboost_forecast_memory',
                                    title=f"Uso de Memória (GB)", namespace=namespace,
                                    mape=model.metrics['MAPE'],
                                    r2=model.metrics['R2'],
                                    split_idx=model.X_train.index[-1])

        r2, mape, lags = model.search_lag(df, 'ocnr_nm_result')
        ts_view.plot_lag_search(r2, mape, lags, title="Performance de cada lag", namespace=namespace, filename='nov_xgboost_forecast_memory')

    def run(self):
        for n, v in zip(self.namespace, self.view_name):
            self.test_model(n, v)

        