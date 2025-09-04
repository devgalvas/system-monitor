from controllers.controller_utils import get_view_name 

from models.multi_nn_model import MultiNNModel
import views.time_series_view as ts_view
from models.dataloader import DataLoader

def train_multi_nn(namespace, view_name):
    dl.createNamespaceView(namespace, view_name)
    df = dl.query_to_db(f"""
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
                            split_idx=len(model.X_train),
                            dir_path='outputs/neural_network/')
        return model
    
def test_multi_nn_day(namespace, view_name, model):
    df_26 = dl.query_to_db(f"""
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
                            r2=metrics['R2'],
                            dir_path='outputs/neural_network/')

dl = DataLoader()
if __name__ == "__main__":
    dl.connect_to_db()

    namespaces = ['panda-nifi', 'panda-druid', 'telefonica-cem-mbb-prod']
    for n in namespaces:
        v = get_view_name(n)
        model = train_multi_nn(n, v)
        test_multi_nn_day(n, v, model)

    dl.close()