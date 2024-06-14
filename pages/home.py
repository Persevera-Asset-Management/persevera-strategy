import streamlit as st
from st_files_connection import FilesConnection


def show_home():
    st.header("Persevera Asset Management")

    conn = st.connection('s3', type=FilesConnection)
    df = conn.read("s3://persevera/multi_factor_screening.parquet", input_format='parquet')
    st.dataframe(df)
