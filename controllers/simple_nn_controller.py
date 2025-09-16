from controllers.controller_utils import get_view_name 

from models.simple_nn_model import SimpleNNModel
import views.time_series_view as ts_view
import views.hyperparameters_view as hp_view
from models.dataloader import DataLoader

def train_simple_nn(namespace, view_name):
        dl.createNamespaceView(namespace, view_name)
        df = dl.query_to_db(f"""
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

        ts_view.plot_model_evolution(pred_time[:len(pred_first_step)],
                                    pred_first_step,
                                    model.time[-len(y_true_first_step):],
                                    y_true_first_step,
                                    filename='nov_nn_forecast_memory',
                                    title=f"Uso de Memória (GB)", namespace=namespace,
                                    mape=model.metrics['MAPE'],
                                    r2=model.metrics['R2'],
                                    split_idx=len(model.X_train),
                                    dir_path='outputs/neural_network/')
        
        hp_view.plot_history(model.history, model.metrics, filename='simple_nn', dir_path='outputs/neural_network/',
                             namespace=namespace, title='Treinamento do SimpleNN')
        
        return model
        
def test_simple_nn_day(namespace, view_name, model):
        df_15 = dl.query_to_db(f"""
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
                                    r2=model.metrics['R2'],
                                    dir_path='outputs/neural_network/')

dl = DataLoader()
if __name__ == "__main__":
    dl.connect_to_db()

    namespaces = ['panda-nifi', 'panda-druid', 'telefonica-cem-mbb-prod']
    for n in namespaces:
        v = get_view_name(n)
        dl.createNamespaceView(n, v)
        df = dl.query_to_db(f"""
            SELECT ocnr_dt_date, ocnr_nm_result FROM {v} 
            WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE'
            ORDER BY ocnr_dt_date          
        """)

        ts_view.plot_time_series(df['ocnr_dt_date'], df['ocnr_nm_result'],
                                filename=f'whole_{n}', title='{n}')
        # model = train_simple_nn(n, v)
        # test_simple_nn_day(n, v, model)

    dl.close()