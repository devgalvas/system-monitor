from utils.controller_utils import get_view_name 

from models.multi_nn_model import MultiNNModel
import views.time_series_view as ts_view
import views.hyperparameters_view as hp_view
from models.dataloader import DataLoader

import pandas as pd

def train_multi_nn(view_name):
    df = dl.query_to_db(f"""
        SELECT
            ocnr_dt_date,
            MAX(ocnr_nm_result) FILTER (WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE') AS memory_usage,
            MAX(ocnr_nm_result) FILTER (WHERE ocnr_tx_query = 'NAMESPACE_CPU_USAGE') AS cpu_usage,
            MAX(ocnr_nm_result) FILTER (WHERE ocnr_tx_query = 'NAMESPACE_DISK_PERCENT') AS disk_percent
        FROM {view_name}
        WHERE ocnr_tx_query IN ('NAMESPACE_MEMORY_USAGE', 'NAMESPACE_CPU_USAGE', 'NAMESPACE_DISK_PERCENT')
        GROUP BY ocnr_dt_date
        ORDER BY ocnr_dt_date;
    """)
    # AND EXTRACT(MONTH FROM ocnr_dt_date) = 11
    model = MultiNNModel()
    model.train(df)

    return model, df
    
def test_multi_nn_entire(namespace, model, df):
    pred, pred_time = model.predict(df)
    
    for i, c in enumerate(model.target_columns):
        pred_first_step = pred[:, i] 
        y_true_first_step = model.y[:, 0, i]
        metrics_per_col = model.metrics_per_col[i]

        ts_view.plot_model_evolution(pred_time[:len(pred_first_step)],
                            pred_first_step,
                            model.time[-len(y_true_first_step):],
                            y_true_first_step,
                            filename=f'entire_multi_nn_forecast_{c}',
                            title=f"{c}", namespace=namespace,
                            mape=metrics_per_col['MAPE'],
                            r2=metrics_per_col['R2'],
                            split_idx=len(model.X_train),
                            dir_path='outputs/neural_network/')
        

    hp_view.plot_history(model.history, model.metrics, filename='multi_nn', dir_path='outputs/neural_network/',
                             namespace=namespace, title='Treinamento do MultiNN')

def test_multi_nn_day(namespace, view_name, model):
    df_26 = dl.query_to_db(f"""
        SELECT
            ocnr_dt_date,
            MAX(ocnr_nm_result) FILTER (WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE') AS memory_usage,
            MAX(ocnr_nm_result) FILTER (WHERE ocnr_tx_query = 'NAMESPACE_CPU_USAGE') AS cpu_usage,
            MAX(ocnr_nm_result) FILTER (WHERE ocnr_tx_query = 'NAMESPACE_DISK_PERCENT') AS disk_percent
        FROM {view_name}
        WHERE ocnr_tx_query IN ('NAMESPACE_MEMORY_USAGE', 'NAMESPACE_CPU_USAGE', 'NAMESPACE_DISK_PERCENT')
        AND EXTRACT(MONTH FROM ocnr_dt_date) = 4
        AND EXTRACT(DAY FROM ocnr_dt_date) = 15
        GROUP BY ocnr_dt_date
        ORDER BY ocnr_dt_date;
    """)

    pred, pred_time = model.predict(df_26)

    for i, c in enumerate(model.target_columns):
        pred_first_step = pred[:, i] 
        metrics_per_col = model.metrics_per_col[i]

        ts_view.plot_model_evolution(pred_time[:len(pred_first_step)],
                            pred_first_step,
                            df_26['ocnr_dt_date'],
                            df_26[c],
                            filename=f'day15april_multi_nn_forecast_{c}',
                            title=f"{c}", namespace=namespace,
                            mape=metrics_per_col['MAPE'],
                            r2=metrics_per_col['R2'],
                            dir_path='outputs/neural_network/')

dl = DataLoader()
if __name__ == "__main__":
    dl.connect_to_db()

    # namespaces = ['panda-nifi', 'panda-druid', 'telefonica-cem-mbb-prod']
    namespaces = dl.query_to_db("""
        SELECT ocnr_tx_namespace, SUM(count_result) AS total_count
        FROM ocnr_overview
        GROUP BY ocnr_tx_namespace
        ORDER BY total_count DESC
        LIMIT 20;                         
    """)['ocnr_tx_namespace']

    rows = []

    for n in namespaces:
        v = get_view_name(n)
        dl.createNamespaceView(n, v)
        model, df = train_multi_nn(v)
        for m in model.metrics_per_col:
            row = {"namespace": n, "query": m['column']}
            row.update({k: v for k, v in m.items() if k != "column"})
            rows.append(row)

    df_results = pd.DataFrame(rows)

    results_path = 'outputs/neural_network/multi_nn_results.csv'
    print(f"Salvando em {results_path}")
    df_results.to_csv(results_path, index=False)

    dl.close()