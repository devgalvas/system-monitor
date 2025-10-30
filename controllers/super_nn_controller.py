from utils.controller_utils import get_view_name 

from models.dataloader import DataLoader

from models.super_nn_model import SuperNNModel

import pandas as pd

dl = DataLoader()
if __name__ == "__main__":
    dl.connect_to_db()

    namespaces = dl.query_to_db("""
        SELECT ocnr_tx_namespace, SUM(count_result) AS total_count
        FROM ocnr_overview
        GROUP BY ocnr_tx_namespace
        ORDER BY total_count DESC
        LIMIT 20;                         
    """)['ocnr_tx_namespace']

    namespaces = namespaces[namespaces != "smartoss-application-prod"]

    queries = ['NAMESPACE_MEMORY_USAGE', 'NAMESPACE_DISK_PERCENT', 'NAMESPACE_CPU_USAGE']

    all_dfs = []
    for n in namespaces:
        v = get_view_name(n)
        dl.createNamespaceView(n, v)
        df = dl.query_to_db(f"""
            SELECT ocnr_dt_date, ocnr_nm_result, ocnr_tx_namespace FROM {v} 
            WHERE ocnr_tx_query = '{queries[0]}'
            ORDER BY ocnr_dt_date
        """)
        all_dfs.append(df)

    rows = []

    for q in queries:
        model = SuperNNModel()
        model.train(all_dfs, 'ocnr_nm_result')
        model.save(q)
        for m in model.metrics:
            row = {"query": q}
            row.update(m)  # adiciona métricas como colunas
            rows.append(row)

    df_results = pd.DataFrame(rows)

    results_path = 'outputs/neural_network/super_nn_results.csv'
    print(f"Salvando em {results_path}")
    df_results.to_csv(results_path, index=False)

    dl.close()