import streamlit as st
from streamlit_option_menu import option_menu


def show_screener():
    st.header("Screener")

    selected_category = option_menu(
        menu_title=None,
        options=["Geral", "Persevera MultiFactor Model (PMM)"],
        orientation="horizontal"
    )