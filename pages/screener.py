import pandas as pd
import os
import streamlit as st
from streamlit_option_menu import option_menu

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')


def get_screen(fields: list):
    zoo = pd.read_parquet(os.path.join(DATA_PATH, "factors-factor_zoo.parquet"), columns=fields)
    zoo = zoo.droplevel(1)
    sectors = pd.read_excel(os.path.join(DATA_PATH, "cadastro-base.xlsx"), sheet_name='equities')
    sectors = sectors[['code', 'name', 'sector_layer_1']].set_index('code')

    df = sectors.merge(zoo, left_index=True, right_index=True, how='right')

    return df


def show_screener():
    st.header("Screener")

    selected_category = option_menu(
        menu_title=None,
        options=["Geral", "Persevera MultiFactor Model (PMM)"],
        orientation="horizontal"
    )

    if selected_category == "Geral":
        variables_available = pd.read_parquet(os.path.join(DATA_PATH, "factors-factor_zoo.parquet")).columns
        sectors_available = sorted(pd.read_excel(os.path.join(DATA_PATH, "cadastro-base.xlsx"), sheet_name='equities')['sector_layer_1'].dropna().unique())
        sectors_available.insert(0, 'Todos')

        cols = st.columns(2, 'large')

        selected_sector = cols[0].multiselect(label='Selecione os setores:',
                                              options=sectors_available,
                                              default=['Todos'])
        if 'Todos' in selected_sector:
            selected_peers = sectors_available

        selected_variables = cols[1].multiselect(label='Selecione as variáveis:',
                                                 options=variables_available,
                                                 default=['price_close'])

        data = get_screen(fields=selected_variables)
        st.dataframe(data)
