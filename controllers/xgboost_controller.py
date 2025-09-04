from controllers.controller_utils import get_view_name 

import views.time_series_view as ts_view
from models.xgboost_model import XGBoostModel
from models.dataloader import DataLoader

def test_model(namespace, view_name):
    dl.createNamespaceView(namespace, view_name)
    df = dl.query_to_db(f"""
        SELECT ocnr_dt_date, ocnr_nm_result FROM {view_name} 
        WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE'
        AND EXTRACT(MONTH FROM ocnr_dt_date) = 11
        ORDER BY ocnr_dt_date
    """)

    model = XGBoostModel()
    model.train(df, 'ocnr_nm_result')
    print(model.metrics)

    pred, pred_time = model.predict(df, 'ocnr_nm_result')
    ts_view.plot_model_evolution(pred_time, pred,
                                model.time, model.y, filename='nov_xgboost_forecast_memory',
                                title=f"Uso de Memória (GB)", namespace=namespace,
                                mape=model.metrics['MAPE'],
                                r2=model.metrics['R2'],
                                split_idx=model.X_train.index[-1],
                                dir_path='outputs/xgboost/')

    r2, mape, lags = model.search_lag(df, 'ocnr_nm_result')
    ts_view.plot_lag_search(r2, mape, lags, title="Performance de cada lag", 
                            namespace=namespace, 
                            filename='nov_xgboost_forecast_memory',
                            dir_path='outputs/xgboost/')

dl = DataLoader()
if __name__ == "__main__":
    dl.connect_to_db()

    namespaces = ['panda-nifi', 'panda-druid', 'telefonica-cem-mbb-prod']
    for n in namespaces:
        v = get_view_name(n)
        test_model(n, v)

    dl.close()

        