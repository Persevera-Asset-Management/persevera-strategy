import pandas as pd
import numpy as np
import logging, os
from datetime import datetime, timedelta
import streamlit as st
from streamlit_option_menu import option_menu

import plotly.graph_objects as go
import plotly.express as px

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))


def format_chart(figure):
    figure.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=False),
            type="date",
        ),
        yaxis_title=None, xaxis_title=None,
        yaxis=dict(autorange=True, fixedrange=False, griddash="dash"),
        legend=dict(title=None, yanchor="top", orientation="v"),
        showlegend=True,
    )
    return figure


def show_fixed_income():
    st.header("Renda Fixa")

    selected_category = option_menu(
        menu_title=None,
        options=["Crédito Corporativo", "Títulos Públicos"],
        orientation="horizontal"
    )

    if selected_category == "Crédito Corporativo":
        st.subheader("Yield Curve")

        sectors = pd.read_excel(os.path.join(DATA_PATH, "cadastro-base.xlsx"), sheet_name="equities")
        sectors['issuer_equity_code'] = sectors['code'] + ' ' + sectors['code_exchange']
        sectors = sectors[['issuer_equity_code', 'sector_layer_0']]

        bonds = pd.read_parquet(os.path.join(DATA_PATH, "fixed_income-data.parquet")).reset_index()
        df = pd.merge(left=bonds, right=sectors, on='issuer_equity_code', how='left')

        cols = st.columns(2, gap='large')
        with cols[0]:
            sector_list = sorted(sectors['sector_layer_0'].dropna().unique())
            sector_list.insert(0, 'Todos')
            selected_sectors = st.multiselect(label='Selecione os setores:',
                                              options=sector_list,
                                              default=['Todos'])
            if 'Todos' in selected_sectors:
                selected_sectors = sector_list

        with cols[1]:
            index_type = st.radio(label='Selecione o indexador:',
                                  options=['BRAZIL DI+SPREAD', 'BRAZIL IPCA+SPREAD'],
                                  horizontal=True)

        latest_date = df['date'].max() - timedelta(days=1)
        df = df[df['date'] == latest_date]
        df = df[df['currency'] == "BRL"]
        df = df[df['index_type'] == index_type]
        df = df.sort_values("maturity")
        df = df.query('sector_layer_0 == @selected_sectors')

        fig = px.scatter(df, x="maturity", y="yield_to_maturity", size='amount_issued',
                         hover_data=['security_name', 'issuer'], color="sector_layer_0", trendline="lowess")
        fig.update_yaxes(autorangeoptions_clipmin=0)
        st.plotly_chart(format_chart(figure=fig), use_container_width=True)

        st.write(latest_date)
        st.dataframe(df.set_index('code').filter(['issuer', 'date', 'issue_date', 'maturity', 'yield_to_maturity', 'price_close', 'amount_issued', 'index_type', 'sector_layer_0']))

    elif selected_category == "Títulos Públicos":
        pass
