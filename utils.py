import streamlit as st
from PIL import Image


def display_sidebar_header() -> None:
    # Logo
    logo = Image.open(r"G:\My Drive\dash_estrategia\assets\Persevera_logo.png")
    with st.sidebar:
        st.image(logo, use_column_width=True)
        col1, col2 = st.columns(2)
        st.header("")  # add space between logo and selectors
