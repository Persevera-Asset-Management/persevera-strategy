import streamlit as st
from st_files_connection import FilesConnection


def show_home():
    st.header("Persevera Asset Management")

    conn = st.connection('s3', type=FilesConnection)
    df = conn.read("s3://persevera/factor_zoo.parquet", input_format='parquet')
    df = df.query('code == "VALE3"').filter(['price_close', 'dividend_per_share'])
    st.dataframe(df)
