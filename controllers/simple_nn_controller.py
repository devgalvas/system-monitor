from utils.controller_utils import get_view_name 

from models.simple_nn_model import SimpleNNModel
import views.time_series_view as ts_view
import views.hyperparameters_view as hp_view
from models.dataloader import DataLoader

import pandas as pd

def train_simple_nn(view_name, query):
        df = dl.query_to_db(f"""
            SELECT ocnr_dt_date, ocnr_nm_result FROM {view_name} 
            WHERE ocnr_tx_query = '{query}'
            ORDER BY ocnr_dt_date
        """)

        model = SimpleNNModel()
        model.train(df, "ocnr_nm_result")
        print(model.metrics)
        
        return model, df
        
def test_simple_nn_entire(namespace, model, df):
    pred, pred_time = model.predict(df, 'ocnr_nm_result')

    pred_first_step = pred[:, 0] 
    y_true_first_step = model.y[:, 0]

    ts_view.plot_model_evolution(pred_time[:len(pred_first_step)],
                                pred_first_step,
                                model.time[-len(y_true_first_step):],
                                y_true_first_step,
                                filename='entire_nn_forecast_memory',
                                title=f"Uso de Memória (GB)", namespace=namespace,
                                mape=model.metrics['MAPE'],
                                r2=model.metrics['R2'],
                                split_idx=len(model.X_train),
                                dir_path='outputs/neural_network/')
    
    hp_view.plot_history(model.history, model.metrics, filename='simple_nn', dir_path='outputs/neural_network/',
                            namespace=namespace, title='Treinamento do SimpleNN')

def test_simple_nn_day(namespace, view_name, model):
        df_15 = dl.query_to_db(f"""
            SELECT ocnr_dt_date, ocnr_nm_result FROM {view_name} 
            WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE'
            AND EXTRACT(MONTH FROM ocnr_dt_date) = 4
            AND EXTRACT(DAY FROM ocnr_dt_date) = 15
            ORDER BY ocnr_dt_date
        """)

        pred, pred_time = model.predict(df_15, 'ocnr_nm_result')
        pred_first_step = pred[:, 0] 
        pred_time = pred_time[:len(pred_first_step)]

        ts_view.plot_model_evolution(pred_time,
                                    pred_first_step,
                                    df_15['ocnr_dt_date'],
                                    df_15['ocnr_nm_result'],
                                    filename='day15april_nn_forecast_memory',
                                    title=f"Uso de Memória (GB)", namespace=namespace,
                                    mape=model.metrics['MAPE'],
                                    r2=model.metrics['R2'],
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

    queries = ['NAMESPACE_MEMORY_USAGE', 'NAMESPACE_DISK_PERCENT', 'NAMESPACE_CPU_USAGE']
    rows = []

    for n in namespaces:
        v = get_view_name(n)
        dl.createNamespaceView(n, v)
        for q in queries:
            model, df = train_simple_nn(v, q)
            row = {"namespace": n, "query": q}
            row.update(model.metrics)  # adiciona métricas como colunas
            rows.append(row)

    df_results = pd.DataFrame(rows)

    results_path = 'outputs/neural_network/simple_nn_results.csv'
    print(f"Salvando em {results_path}")
    df_results.to_csv(results_path, index=False)

    dl.close()