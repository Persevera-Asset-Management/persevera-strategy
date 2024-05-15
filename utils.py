import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')


def get_data(fields: list):
    df = pd.read_parquet(os.path.join(DATA_PATH, "consolidado-indicators.parquet"),
                         filters=[('code', 'in', fields)])
    df = df.pivot_table(index='date', columns='code', values='value')
    df = df.filter(fields)
    return df
