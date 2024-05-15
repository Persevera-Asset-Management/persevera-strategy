import pandas as pd
import numpy as np
from datetime import datetime
import logging, os
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import utils

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))


def get_copom_data(meeting):
    df = pd.read_parquet(os.path.join(DATA_PATH, "b3-copom_options.parquet"),
                         columns=['date', 'date_expiration', 'price', 'decision'],
                         filters=[('date_expiration', '==', meeting)])

    df = df.pivot_table(index='date', columns='decision', values='price')
    df = df.replace(0.0, np.nan)
    df = df.dropna(how='all', axis='columns')
    return df


def get_copom_meeting_dates():
    df = pd.read_parquet(os.path.join(DATA_PATH, "b3-copom_options.parquet"), columns=['date_expiration'])
    unique_dates = df['date_expiration'].unique()
    dates = np.flip(unique_dates)
    return dates


def format_chart(figure, yaxis_range=None, showlegend=True, connect_gaps=False):
    figure.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=False),
            type="date",
        ),
        yaxis_title=None, xaxis_title=None,
        yaxis=dict(range=yaxis_range, fixedrange=False, griddash="dash"),
        legend=dict(title=None, yanchor="top", orientation="h"),
        showlegend=showlegend,
        hovermode="x unified",
    )
    figure.update_traces(connectgaps=connect_gaps, hovertemplate="%{y}")
    return figure


def format_bar_chart(figure):
    figure.update_layout(
        yaxis_title=None,
        xaxis_title=None,
        yaxis=dict(range=[0, 100], fixedrange=False, griddash="dash"),
        legend=dict(title=None, yanchor="top", orientation="h"),
        showlegend=False,
    )
    return figure


def show_tools():
    st.header("Tools")

    selected_category = option_menu(
        menu_title=None,
        options=["Opções de Copom", "Drivers", "Selic Implícita", "Valuation"],
        orientation="horizontal"
    )

    if selected_category == "Opções de Copom":
        cols = st.columns(2, gap='large')
        with cols[0]:
            meeting_date = st.selectbox(label="Selecione a data da reunião do Copom:",
                                        options=get_copom_meeting_dates(),
                                        index=int((np.abs(get_copom_meeting_dates() - datetime.today())).argmin()),
                                        format_func=lambda x: format(x, "%Y-%m-%d")
                                        )

        cols = st.columns(2, gap='large')
        with cols[0]:
            cols[0].markdown("**Histórico**")
            df_copom_history = get_copom_data(meeting_date)
            fig = px.line(df_copom_history, markers=True)
            st.plotly_chart(format_chart(figure=fig, yaxis_range=[0, 100], connect_gaps=True), use_container_width=True)

        with cols[1]:
            cols[1].markdown("**Distribuição**")
            decisions = [
                'Queda de 2%',
                'Queda de 1.75%',
                'Queda de 1.5%',
                'Queda de 1.25%',
                'Queda de 1%',
                'Queda de 0.75%',
                'Queda de 0.5%',
                'Queda de 0.25%',
                'Manutenção',
                'Aumento de 0.25%',
                'Aumento de 0.5%',
                'Aumento de 0.75%',
                'Aumento de 1%',
                'Aumento de 1.25%',
                'Aumento de 1.5%',
                'Aumento de 1.75%',
                'Aumento de 2%',
            ]
            df_copom_dist = df_copom_history.filter(decisions).iloc[-1].fillna(0)
            fig = px.bar(df_copom_dist)
            st.plotly_chart(format_bar_chart(figure=fig), use_container_width=True)

    elif selected_category == "Drivers":
        cols = st.columns(2, gap='large')
        with cols[0]:
            df_drivers = pd.read_parquet(os.path.join(DATA_PATH, "macro_drivers.parquet"))
            df_benchmark = utils.get_data(fields=['br_ibovespa'])
            fig = px.area(df_drivers)
            st.plotly_chart(format_chart(figure=fig), use_container_width=True)

            # fig = make_subplots(specs=[[{"secondary_y": True}]])
            # fig.add_trace(go.Scatter(x=df_drivers.index, y=df_drivers, stackgroup='one'), secondary_y=False)
            # fig.add_trace(go.Scatter(x=df_benchmark.index, y=df_benchmark), secondary_y=True)

        with cols[1]:
            cols[1].markdown("**Distribuição**")
            selected_benchmark = st.selectbox("Selecione o ativo de comparação",
                                              options=['br_pre_1y', 'br_pre_10y', 'br_cds_5y', 'crb_index', 'crude_oil_brent',
                                                       'crb_metals_index', 'brl_usd', 'us_sp500'])
            df = utils.get_data(fields=['br_ibovespa', selected_benchmark])
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, 0], name=df.iloc[:, 0].name), secondary_y=False)
            fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, 1], name=df.iloc[:, 1].name), secondary_y=True)
            st.plotly_chart(format_chart(figure=fig), use_container_width=True)
