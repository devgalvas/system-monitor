from utils.controller_utils import get_view_name 
from utils.smape import smape

from models.dataloader import DataLoader

from models.super_nn_model import SuperNNModel

from views.results_view import atualizar_medias_csv
import views.time_series_view as ts_view

from tensorflow import keras
import pandas as pd
import numpy as np

dl = DataLoader()
if __name__ == "__main__":
    dl.connect_to_db()

    queries = ['NAMESPACE_MEMORY_USAGE', 'NAMESPACE_DISK_PERCENT', 'NAMESPACE_CPU_USAGE']

    namespaces = dl.query_to_db("""
        SELECT ocnr_tx_namespace, SUM(count_result) AS total_count
        FROM ocnr_overview
        GROUP BY ocnr_tx_namespace
        ORDER BY total_count DESC
        LIMIT 25;                         
    """)['ocnr_tx_namespace'][-5:]

    rows = []

    for q in queries:
        super_nn = SuperNNModel()
        super_nn.load(q)

        for n in namespaces:
            v = get_view_name(n)
            dl.createNamespaceView(n, v)

            df = dl.query_to_db(f"""
                    SELECT ocnr_dt_date, ocnr_nm_result, ocnr_tx_namespace FROM {v} 
                    WHERE ocnr_tx_query = '{q}'
                    ORDER BY ocnr_dt_date
            """)

            super_y, super_true, super_time = super_nn.predict(df, 'ocnr_nm_result')
            super_true = np.squeeze(super_true, axis=-1)

            row = {
                "super_MAPE": smape(super_true, super_y),
                "super_R2": keras.metrics.R2Score()(super_true, super_y).numpy(),
                "namespace": n,
                "query": q
            }
            rows.append(row)

            ts_view.plot_model_performance(super_time[:len(super_y)],
                                super_y[:, 0],
                                super_time[-len(super_true):],
                                super_true[:, 0],
                                filename=f'no_training_{q}',
                                title=f"{q}", namespace=n,
                                mape=row['super_MAPE'],
                                r2=row['super_R2'],
                                dir_path='outputs/neural_network/')

    df_results = pd.DataFrame(rows)
    df_results = atualizar_medias_csv(df_results)

    results_path = 'outputs/neural_network/no_training_results.csv'
    print(f"Salvando em {results_path}")
    df_results.to_csv(results_path, index=False)



