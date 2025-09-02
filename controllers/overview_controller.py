from views.overview import Overview
from controllers.base_controllers import BaseController

from statsmodels.tsa.seasonal import STL

class OverviewController(BaseController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.view = None

    def run_volume_data(self):
        count_namespace = self.dl.query_to_db("""
            SELECT ocnr_tx_namespace, SUM(count_result) AS count FROM ocnr_overview 
            GROUP BY ocnr_tx_namespace ORDER BY count DESC
        """)
        self.view.plot_volume_data(count_namespace)

    def run_means(self):
        memory_means = self.dl.query_to_db("""
            SELECT ocnr_tx_namespace, avg_result FROM ocnr_overview 
            WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE' ORDER BY avg_result DESC                                   
        """)
        self.view.plot_means(memory_means, title='Uso de Memória')

    def run_stdev(self):
        memory_stdev = self.dl.query_to_db("""
            SELECT ocnr_tx_namespace, stddev_result FROM ocnr_overview 
            WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE' ORDER BY stddev_result DESC                                   
        """).dropna()
        self.view.plot_stdev(memory_stdev, title='Uso de Memória')

    def run_decomposition(self, namespace, view_name):
        self.dl.createNamespaceView(namespace, view_name)
        df = self.dl.query_to_db(f"""
            SELECT ocnr_dt_date, ocnr_nm_result FROM {view_name} 
            WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE'
            AND EXTRACT(MONTH FROM ocnr_dt_date) = 11
            ORDER BY ocnr_dt_date                                
        """)

        count = self.dl.query_to_db(f"""
            SELECT COUNT(*) as count FROM {view_name} 
            WHERE ocnr_tx_query = 'NAMESPACE_MEMORY_USAGE'
            AND EXTRACT(MONTH FROM ocnr_dt_date) = 11
            AND EXTRACT(DAY FROM ocnr_dt_date) = 16
        """)
        print(count)

        stl = STL(df['ocnr_nm_result'], period=count.iloc[0, 0])
        result = stl.fit()
        self.view.plot_decomposition(result, namespace=namespace, filename='nov_memory')

    def run_samples_daily(self, namespace, view_name):
        self.dl.createNamespaceView(namespace, view_name)
        df = self.dl.query_to_db(f"""
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
        self.view.plot_samples_daily(df, 'nov', namespace, mean)

    def run(self):
        self.view = Overview()
        for n, v in zip(self.namespace, self.view_name):
            self.run_samples_daily(n, v)

        
        

