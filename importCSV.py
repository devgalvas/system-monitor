import os
import io
import time
import pandas as pd
import psycopg2
from psycopg2 import OperationalError

from dotenv import load_dotenv

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ocnr_dados (
    ocnr_cd_id uuid,
    ocnr_tx_query VARCHAR(255),
    ocnr_dt_date TIMESTAMP,
    ocnr_nm_severity INT,
    ocnr_tx_namespace VARCHAR(255),
    ocnr_nm_result NUMERIC,
    opco_cd_id uuid,
    opcn_cd_id uuid,
    opce_cd_id uuid,
    ocnr_tx_key VARCHAR(255),
    ocnr_tx_key2 VARCHAR(255)
);
"""

def connect_to_db(conn_str, retries=10, delay=3):
    for i in range(retries):
        try:
            conn = psycopg2.connect(conn_str)
            print("Postgres está pronto!")
            return conn
        except OperationalError as e:
            print(f"Postgres não disponível ({e}), tentando novamente em {delay} segundos...")
            time.sleep(delay)
    raise Exception("Não foi possível conectar ao Postgres após várias tentativas")


def create_table(conn):
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
        conn.commit()
        print("Tabela criada (ou já existia).")

def copy_csv_to_db(conn, csv_path, table_name, chunk_size=100_000):
    total_size = os.path.getsize(csv_path)  # Tamanho total em bytes

    processed_bytes = 0

    with conn.cursor() as cur, open(csv_path, 'r', encoding='utf-16') as f:
        print("Iniciando a importação.")

        reader = pd.read_csv(f, chunksize=chunk_size)

        for chunk in reader:
            buffer = io.StringIO()
            chunk.to_csv(buffer, index=False, header=False)
            buffer.seek(0)
            cur.copy_expert(f"COPY {table_name} FROM STDIN WITH CSV", buffer)
            conn.commit()

            # Atualiza progresso com base no ponteiro atual do arquivo original
            processed_bytes = f.tell()
            progress = (processed_bytes / total_size) * 100
            print(f"Progresso: {progress:.2f}%")

    print("Importação concluída com sucesso.")

def main():
    load_dotenv()
    DB_CONFIG = {
        'host': os.getenv("DB_HOST"), 
        'port': os.getenv("DB_PORT"), 
        'dbname': os.getenv("DB_NAME"), 
        'user': os.getenv("DB_USER"), 
        'password': os.getenv("DB_PASS"), 
    }
    conn_str = f"host={DB_CONFIG['host']} port={DB_CONFIG['port']} dbname={DB_CONFIG['dbname']} user={DB_CONFIG['user']} password={DB_CONFIG['password']}"
    conn = connect_to_db(conn_str)
    create_table(conn)
    for i in range(5):
        copy_csv_to_db(conn, f"/archive/0{i}.csv", 'ocnr_dados')

if __name__ == "__main__":
    main()