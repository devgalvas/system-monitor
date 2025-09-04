from controllers.controller_utils import get_view_name 

import views.overview as ov
from models.dataloader import DataLoader
from statsmodels.tsa.seasonal import STL

def run_volume_data():
    count_namespace = dl.query_to_db("""
        SELECT ocnr_tx_namespace, SUM(count_result) AS count FROM ocnr_overview 
        GROUP BY ocnr_tx_namespace ORDER BY count DESC
    """)
    ov.plot_volume_data(count_namespace)

def run_means():
    memory_means = dl.query_to_db("""
        SELECT ocnr_tx_namespace, avg_result FROM ocnr_overview 
        WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE' ORDER BY avg_result DESC                                   
    """)
    ov.plot_means(memory_means, title='Uso de Memória')

def run_stdev():
    memory_stdev = dl.query_to_db("""
        SELECT ocnr_tx_namespace, stddev_result FROM ocnr_overview 
        WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE' ORDER BY stddev_result DESC                                   
    """).dropna()
    ov.plot_stdev(memory_stdev, title='Uso de Memória')

def run_decomposition(namespace, view_name):
    dl.createNamespaceView(namespace, view_name)
    df = dl.query_to_db(f"""
        SELECT ocnr_dt_date, ocnr_nm_result FROM {view_name} 
        WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE'
        AND EXTRACT(MONTH FROM ocnr_dt_date) = 11
        ORDER BY ocnr_dt_date                                
    """)

    count = dl.query_to_db(f"""
        SELECT COUNT(*) as count FROM {view_name} 
        WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE'
        AND EXTRACT(MONTH FROM ocnr_dt_date) = 11
        AND EXTRACT(DAY FROM ocnr_dt_date) = 16
    """)
    print(count)

    stl = STL(df['ocnr_nm_result'], period=count.iloc[0, 0])
    result = stl.fit()
    ov.plot_decomposition(result, namespace=namespace, filename='nov_memory')

def run_samples_daily(namespace, view_name):
    dl.createNamespaceView(namespace, view_name)
    df = dl.query_to_db(f"""
        SELECT 
            EXTRACT(DAY FROM ocnr_dt_date) AS dia,
            COUNT(*) AS total
        FROM {view_name}
        WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE'
        AND EXTRACT(MONTH FROM ocnr_dt_date) = 11
        GROUP BY dia
        ORDER BY dia;
    """)

    mean = df['total'].mean()
    ov.plot_samples_daily(df, 'nov', namespace, mean)

dl = DataLoader()
if __name__ == '__main__':
    dl.connect_to_db()
    namespaces = ['panda-nifi', 'panda-druid', 'telefonica-cem-mbb-prod']
    for n in namespaces:
        v = get_view_name(n)
        run_samples_daily(n, v)
    dl.close()
        

