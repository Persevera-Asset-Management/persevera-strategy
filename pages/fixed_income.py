import streamlit as st
from streamlit_option_menu import option_menu


def show_fixed_income():
    st.header("Renda Fixa")

    selected_category = option_menu(
        menu_title=None,
        options=["Crédito Corporativo", "Títulos Públicos"],
        orientation="horizontal"
    )

    if selected_category == "Crédito Corporativo":
        pass

    elif selected_category == "Títulos Públicos":
        pass
