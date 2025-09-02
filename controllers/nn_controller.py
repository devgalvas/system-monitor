from views.time_series_view import TimeSeriesView 
from controllers.base_controllers import BaseController
from models.simple_nn_model import SimpleNNModel
from models.multi_nn_model import MultiNNModel

import pandas as pd

class NeuralNetworkController(BaseController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def test_simple_nn(self, namespace, view_name):
        self.dl.createNamespaceView(namespace, view_name)
        df = self.dl.query_to_db(f"""
            SELECT ocnr_dt_date, ocnr_nm_result FROM {view_name} 
            WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE'
            AND EXTRACT(MONTH FROM ocnr_dt_date) = 11
            ORDER BY ocnr_dt_date
        """)

        model = SimpleNNModel()
        model.train(df, "ocnr_nm_result")
        print(model.metrics)

        pred, pred_time = model.predict(df, 'ocnr_nm_result')

        pred_first_step = pred[:, 0] 
        y_true_first_step = model.y[:, 0]

        ts_view = TimeSeriesView(dir_path = 'outputs/neural_network/')
        ts_view.plot_model_evolution(pred_time[:len(pred_first_step)],
                                    pred_first_step,
                                    model.time[-len(y_true_first_step):],
                                    y_true_first_step,
                                    filename='nov_nn_forecast_memory',
                                    title=f"Uso de Memória (GB)", namespace=namespace,
                                    mape=model.metrics['MAPE'],
                                    r2=model.metrics['R2'],
                                    split_idx=len(model.X_train))
        
        df_15 = self.dl.query_to_db(f"""
            SELECT ocnr_dt_date, ocnr_nm_result FROM {view_name} 
            WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE'
            AND EXTRACT(MONTH FROM ocnr_dt_date) = 11
            AND EXTRACT(DAY FROM ocnr_dt_date) = 20
            ORDER BY ocnr_dt_date
        """)

        pred, pred_time = model.predict(df_15, 'ocnr_nm_result')
        pred_first_step = pred[:, 0] 
        pred_time = pred_time[:len(pred_first_step)]

        ts_view.plot_model_evolution(pred_time,
                                    pred_first_step,
                                    df_15['ocnr_dt_date'],
                                    df_15['ocnr_nm_result'],
                                    filename='day15_nn_forecast_memory',
                                    title=f"Uso de Memória (GB)", namespace=namespace,
                                    mape=model.metrics['MAPE'],
                                    r2=model.metrics['R2'])
        
    def test_multi_nn(self, namespace, view_name):
        self.dl.createNamespaceView(namespace, view_name)
        df = self.dl.query_to_db(f"""
            SELECT
                ocnr_dt_date,
                MAX(ocnr_nm_result) FILTER (WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE') AS memory_usage,
                MAX(ocnr_nm_result) FILTER (WHERE ocnr_tx_query = 'NAMESPACE_CPU_USAGE') AS cpu_usage,
                MAX(ocnr_nm_result) FILTER (WHERE ocnr_tx_query = 'NAMESPACE_DISK_PERCENT') AS disk_percent
            FROM {view_name}
            WHERE ocnr_tx_query IN ('NAMESPACE_MEMORY_USAGE', 'NAMESPACE_CPU_USAGE', 'NAMESPACE_DISK_PERCENT')
            AND EXTRACT(MONTH FROM ocnr_dt_date) = 11
            GROUP BY ocnr_dt_date
            ORDER BY ocnr_dt_date;
        """)
        model = MultiNNModel()
        model.train(df)
        print(model.metrics)

        pred, pred_time = model.predict(df)

        ts_view = TimeSeriesView(dir_path = 'outputs/neural_network/')
        for i, c in enumerate(model.target_columns):
            pred_first_step = pred[:, i] 
            y_true_first_step = model.y[:, 0, i]
            metrics = model.metrics[i]

            ts_view.plot_model_evolution(pred_time[:len(pred_first_step)],
                                pred_first_step,
                                model.time[-len(y_true_first_step):],
                                y_true_first_step,
                                filename=f'nov_multi_nn_forecast_{c}',
                                title=f"{c}", namespace=namespace,
                                mape=metrics['MAPE'],
                                r2=metrics['R2'],
                                split_idx=len(model.X_train))

        df_26 = self.dl.query_to_db(f"""
            SELECT
                ocnr_dt_date,
                MAX(ocnr_nm_result) FILTER (WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE') AS memory_usage,
                MAX(ocnr_nm_result) FILTER (WHERE ocnr_tx_query = 'NAMESPACE_CPU_USAGE') AS cpu_usage,
                MAX(ocnr_nm_result) FILTER (WHERE ocnr_tx_query = 'NAMESPACE_DISK_PERCENT') AS disk_percent
            FROM {view_name}
            WHERE ocnr_tx_query IN ('NAMESPACE_MEMORY_USAGE', 'NAMESPACE_CPU_USAGE', 'NAMESPACE_DISK_PERCENT')
            AND EXTRACT(MONTH FROM ocnr_dt_date) = 11
            AND EXTRACT(DAY FROM ocnr_dt_date) = 26
            GROUP BY ocnr_dt_date
            ORDER BY ocnr_dt_date;
        """)

        pred, pred_time = model.predict(df_26)

        ts_view = TimeSeriesView(dir_path = 'outputs/neural_network/')
        for i, c in enumerate(model.target_columns):
            pred_first_step = pred[:, i] 
            metrics = model.metrics[i]

            ts_view.plot_model_evolution(pred_time[:len(pred_first_step)],
                                pred_first_step,
                                df_26['ocnr_dt_date'],
                                df_26[c],
                                filename=f'day26_multi_nn_forecast_{c}',
                                title=f"{c}", namespace=namespace,
                                mape=metrics['MAPE'],
                                r2=metrics['R2'])

    def run(self):
        for n, v in zip(self.namespace, self.view_name):
            # self.test_multi_nn(n, v)
            self.test_simple_nn(n, v)
