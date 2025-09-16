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

def connect_to_db(conn_str):
    conn = psycopg2.connect(conn_str)
    print("Postgres está pronto!")
    return conn


def create_table(conn):
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
        conn.commit()
        print("Tabela criada (ou já existia).")

def copy_csv_to_db(conn, table_name='ocnr_dados', chunk_size=100_000):
    try:
        leftover = ""
        for i in range(0, 6):
            csv_path = rf"D:\Vertis\volumes\archive\0{i}.csv"
            # tamanho_bytes = os.path.getsize(csv_path)
            # print(tamanho_bytes)
            if i % 2 == 0:
                encoding = 'utf-16-le'
            else:
                encoding = 'utf-16-be'
            with open(csv_path, "r", encoding=encoding, errors="ignore") as f:
                batch = []
                with conn.cursor() as cur:
                    for line_number, line in enumerate(f, 1):
                        # Junta sobra do arquivo anterior com a primeira linha
                        if leftover:
                            line = leftover + line
                            leftover = ""

                        # Se a linha terminar sem quebra de linha, guarda em leftover
                        if not line.endswith("\n"):
                            leftover = line
                            continue

                        batch.append(line)

                        # Se chegou no tamanho do chunk → manda pro Postgres
                        if len(batch) >= chunk_size:
                            buffer = io.StringIO("".join(batch))
                            cur.copy_expert(
                                f"COPY {table_name} FROM STDIN WITH (FORMAT CSV, HEADER, DELIMITER ',')",
                                buffer
                            )
                            conn.commit()
                            batch.clear()
                            print(f"{line_number} linhas processadas em {csv_path}...")

                    # Últimas linhas do arquivo atual
                    if batch:
                        buffer = io.StringIO("".join(batch))
                        cur.copy_expert(
                            f"COPY {table_name} FROM STDIN WITH (FORMAT CSV, HEADER, DELIMITER ',')",
                            buffer
                        )
                        conn.commit()
                        print(f"Finalizado {csv_path}.")

        # Se depois do último arquivo ainda sobrou linha → manda também
        if leftover:
            with conn.cursor() as cur:
                buffer = io.StringIO(leftover)
                cur.copy_expert(
                    f"COPY {table_name} FROM STDIN WITH (FORMAT CSV, HEADER FALSE, DELIMITER ',')",
                    buffer
                )
                conn.commit()
            print("Última linha concatenada e inserida.")

        print("✅ Importação concluída com sucesso.")

    except Exception as e:
        print("❌ Erro na importação:", e)

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
    copy_csv_to_db(conn)

if __name__ == "__main__":
    main()