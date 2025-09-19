import pandas as pd

class PostgresDataLoader:
    def __init__(self, engine, query, chunksize=10000):
        self.query = query
        self.chunksize = chunksize
        self.engine = engine

    def __iter__(self):
        for chunk in pd.read_sql_query(self.query, self.engine, chunksize=self.chunksize):
            yield chunk