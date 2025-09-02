import os
import subprocess
import time

import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv

class DataLoader:
    def __init__(self):
        load_dotenv()

        self.engine = None
        self.ssh_process = None

        self.table_name = os.getenv("DB_TABLE", "ocnr_dados")
        self.host = os.getenv("DB_HOST", "localhost")    
        self.dbname = os.getenv("DB_NAME")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASS")

        self.local_port = os.getenv("LOCAL_PORT")

        self.ssh_host = os.getenv("SSH_HOST")
        self.ssh_port = int(os.getenv("SSH_PORT", 22))
        self.ssh_user = os.getenv("SSH_USER")
        self.ssh_pass = os.getenv("SSH_PASS")

    def start_ssh_tunnel(self):
        cmd = [
            "sshpass", "-p", self.ssh_pass,
            "ssh", "-N", "-L", f"{self.local_port}:localhost:5432",
            f"{self.ssh_user}@{self.ssh_host}",
            "-p", str(self.ssh_port)
        ]
        self.ssh_process = subprocess.Popen(cmd)
        time.sleep(2)  # espera o túnel iniciar
        print(f"Túnel SSH aberto na porta {self.local_port}")

    def connect_to_db(self):
        db_url = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.local_port}/{self.dbname}"
        self.engine = create_engine(db_url)
        print("Connected")

    def close(self):
        if self.engine:
            self.engine.dispose()
            print("Connection closed.")
        if self.ssh_process:
            self.ssh_process.terminate()
            print("Túnel SSH encerrado.")

    def query_to_db(self, query):
        return pd.read_sql(f"{query}", self.engine)
    
    def createOverviewView(self):
        sql = """
        CREATE MATERIALIZED VIEW ocnr_overview AS
        SELECT
            ocnr_tx_query,
            ocnr_tx_namespace,
            MAX(ocnr_nm_result) AS max_result,
            MIN(ocnr_nm_result) AS min_result,
            AVG(ocnr_nm_result) AS avg_result,
            STDDEV(ocnr_nm_result) AS stddev_result,
            COUNT(*) AS count_result
        FROM
            ocnr_dados
        GROUP BY
            ocnr_tx_namespace,
            ocnr_tx_query;
        """
        try:
            with self.engine.begin() as conn:
                conn.execute(sqlalchemy.text(sql))
            print("View criada com sucesso.")
        except Exception as e:
            print(f"Erro ao criar a view: {e}")

    def fullDataOverview(self): 
        sql = "SELECT * FROM ocnr_overview ORDER BY ocnr_tx_namespace, ocnr_tx_query;"
        return pd.read_sql(sql, self.engine)
    
    def createNamespaceView(self, namespace, view_name):
        check_sql = """
            SELECT EXISTS (
                SELECT 1
                FROM pg_matviews
                WHERE matviewname = :view_name
            )
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(check_sql), {"view_name": view_name})
                exists = result.scalar()
                if exists:
                    print(f"A view '{view_name}' já existe. Nenhuma ação tomada.")
                    return
            sql = f"""
                CREATE MATERIALIZED VIEW {view_name} AS
                SELECT *
                FROM ocnr_dados
                WHERE ocnr_tx_namespace = :namespace
            """
            with self.engine.begin() as conn:
                conn.execute(sqlalchemy.text(sql), {"namespace": namespace})
            print(f"View '{view_name}' criada com sucesso.")
        except Exception as e:
            print(f"Erro ao criar a view: {e}")