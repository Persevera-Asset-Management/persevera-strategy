import pandas as pd
import numpy as np
from datetime import datetime
import logging, os
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from factor_strategies import factor_screening
import utils

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))


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


def show_factor_playground():
    st.header("Factor Playground")

    selected_category = option_menu(
        menu_title=None,
        options=["Factor Radar", "Performance", "Backtester"],
        orientation="horizontal"
    )

    if selected_category == "Factor Radar":
        cols = st.columns(2, gap='large')
        with cols[0]:
            stocks_info = pd.read_excel(os.path.join(DATA_PATH, "cadastro-base.xlsx"), sheet_name="equities").query('code_exchange == "BZ"')
            selected_stocks = st.multiselect(label='Selecione as ações:',
                                             options=sorted(stocks_info['code']),
                                             default=["VALE3"],
                                             max_selections=2)

    elif selected_category == "Backtester":
        with st.form("factor_definition"):
            st.markdown("**Definição dos fatores**")

            cols = st.columns(2, gap='large')
            with cols[0]:
                freq = st.selectbox("Frequência de rebalanceamento", options=["D", "M"], index=1)
                holding_period = st.number_input("Holding period", value=1, min_value=1, max_value=30)
                start = st.date_input("Data de início", value=datetime(2008, 1, 1), format="YYYY-MM-DD")
                quantile = st.number_input("Quantis", value=5, min_value=1, max_value=5)
                sector = st.selectbox("Agrupamento setorial", index=1,
                                      options=["sector_layer_0", "sector_layer_1", "sector_layer_2", "sector_layer_3"])

            with cols[1]:
                container = st.container(border=True)
                with container:
                    st.markdown("**Definição do universo**")
                    liquidity_thresh = st.number_input("Percentil de liquidez", value=0.4, min_value=0., max_value=1.,
                                                       step=0.1)
                    liquidity_lookback = st.selectbox("Janela de liquidez (em dias úteis)", options=["21", "63", "252"],
                                                      index=0)
                    size_segment = st.selectbox("Tamanho", options=["ALL", "Large", "Mid", "Small"], index=0)

                container = st.container(border=True)
                with container:
                    use_buckets = st.checkbox("Buckets", value=True)
                    use_factor_relevance = st.checkbox("Relevância dos fatores", value=True)
                    use_sector_score = st.checkbox("Score storial", value=True)

            submitted = st.form_submit_button("Submit")
            if submitted:
                f = factor_screening.MultiFactorStrategy(
                    factor_list={
                        'momentum': {
                            'price_momentum': [
                                '12m_momentum',
                                '6m_momentum'
                            ],
                        },
                    },
                    custom_name=None,
                    sector_filter=None,
                    sector_comparison=sector,
                    sector_specifics=sector,
                    holding_period=holding_period,
                    freq=freq,
                    start=format(start, '%Y-%m-%d'),
                    business_day=-2,
                    use_buckets=use_buckets,
                    use_factor_relevance=use_factor_relevance,
                    use_sector_score=use_sector_score,
                    market_cap_neutralization=False,
                    investment_universe={'liquidity_thresh': liquidity_thresh,
                                         'liquidity_lookback': int(liquidity_lookback), 'size_segment': size_segment},
                    quantile=quantile,
                    memory=False,
                    outlier_percentile=[0.02, 0.98],
                )
                f.initialize()
                f.historical_members(save=False, how='overwrite')
                screening = f.raw_data
                screening_last = screening[screening['date'] == screening['date'].max()].set_index('code')
                st.dataframe(screening_last)