import streamlit as st
from st_files_connection import FilesConnection
import pandas as pd


def show_home():
    st.header("Persevera Asset Management")

    conn = st.connection('s3', type=FilesConnection)
    fs = conn.open("s3://persevera/factor_zoo.parquet", input_format='parquet')
    df = pd.read_parquet(fs, filters=[('code', '==', 'VALE3')], columns=['price_close', 'dividend_per_share'])
    st.dataframe(df)
