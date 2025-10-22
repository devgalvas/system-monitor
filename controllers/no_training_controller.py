from utils.controller_utils import get_view_name 
from models.dataloader import DataLoader

from models.simple_nn_model import SimpleNNModel
from models.super_nn_model import SuperNNModel

from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder
from utils.safe_encoder import SafeLabelEncoder

from tensorflow import keras

import numpy as np

dl = DataLoader()
if __name__ == "__main__":
    dl.connect_to_db()

    # namespaces = ['panda-nifi', 'panda-druid', 'telefonica-cem-mbb-prod']
    namespaces = dl.query_to_db("""
        SELECT ocnr_tx_namespace, SUM(count_result) AS total_count
        FROM ocnr_overview
        GROUP BY ocnr_tx_namespace
        ORDER BY total_count DESC
        LIMIT 25;                         
    """)['ocnr_tx_namespace']

    target_namespaces = namespaces[-5:]
    label_namespaces = namespaces[:21]

    simple_nn = SimpleNNModel()
    simple_nn.load()

    super_nn = SuperNNModel()
    super_nn.load()

    queries = ['NAMESPACE_MEMORY_USAGE', 'NAMESPACE_DISK_PERCENT', 'NAMESPACE_CPU_USAGE']

    for n in target_namespaces: 
        v = get_view_name(n)
        dl.createNamespaceView(n, v)
        for q in queries: 
            df = dl.query_to_db(f"""
                SELECT ocnr_dt_date, ocnr_nm_result, ocnr_tx_namespace FROM {v} 
                WHERE ocnr_tx_query = '{q}'
                ORDER BY ocnr_dt_date
            """)

            X, y, time = simple_nn.preprocess(df, 'ocnr_nm_result')
            simple_nn.scaler_X = StandardScaler().fit(X)
            simple_nn.scaler_y = StandardScaler().fit(y)

            super_nn.namespace_encoder = SafeLabelEncoder()
            super_nn.namespace_encoder.fit(label_namespaces)
            X, y, _, ns = super_nn.preprocess(df, 'ocnr_nm_result')
            X, y, ns = super_nn.create_windows(X, y, ns)
            super_nn.scaler_X = StandardScaler().fit(np.squeeze(X, axis=-1))
            super_nn.scaler_y = StandardScaler().fit(np.squeeze(y, axis=-1))

            simple_y, simple_true, _ = simple_nn.predict(df, 'ocnr_nm_result')
            super_y, super_true, _ = super_nn.predict(df, 'ocnr_nm_result')

            # simple_y = simple_y[:, 0]
            # super_y = super_y[:, 0]

            metrics = {
                "simple_MAPE": keras.metrics.MeanAbsolutePercentageError()(simple_true, simple_y).numpy(),
                "simple_R2": keras.metrics.R2Score()(simple_true, simple_y).numpy(),
                "super_MAPE": keras.metrics.MeanAbsolutePercentageError()(super_true, super_y).numpy(),
                "super_R2": keras.metrics.R2Score()(super_true, super_y).numpy(),
                "namespace": n,
                "query": q
            }

            print(metrics)
