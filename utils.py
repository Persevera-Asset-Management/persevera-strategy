import pandas as pd
import os
import streamlit as st
from st_files_connection import FilesConnection

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def get_data(fields: list):
    df = pd.read_parquet(os.path.join(DATA_PATH, "consolidado-indicators.parquet"),
                         filters=[('code', 'in', fields)])
    df = df.pivot_table(index='date', columns='code', values='value')
    df = df.filter(fields)
    return df


def get_fs_connection(file_name: str):
    conn = st.connection('s3', type=FilesConnection)
    fs = conn.open(f"s3://persevera/{file_name}", input_format='parquet')
    return fs
