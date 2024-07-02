import pandas as pd
import numpy as np
import logging, os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import streamlit as st
from streamlit_option_menu import option_menu
from st_files_connection import FilesConnection

import plotly.graph_objects as go
import plotly.express as px

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

de_para = {"Trinity": {"initial_date": datetime(2022, 11, 10),
                       "fund_name": "Persevera Trinity FI RF Ref DI",
                       "benchmark": ["br_cdi_index"]},
           "Yield": {"initial_date": datetime(2023, 9, 29),
                     "fund_name": "Persevera Yield FI RF Ref DI",
                     "benchmark": ["br_cdi_index"]},
           "Phoenix": {"initial_date": datetime(2023, 9, 29),
                       "fund_name": "Persevera Phoenix RF Ativo LP FI",
                       "benchmark": ["br_cdi_index"]},
           "Compass": {"initial_date": datetime(2019, 2, 11),
                       "fund_name": "Persevera Compass Advisory FIC FIM",
                       "benchmark": ["br_cdi_index"]},
           "Nemesis": {"initial_date": datetime(2022, 2, 25),
                       "fund_name": "Persevera Nemesis Total Return FIM",
                       "benchmark": ["br_ibovespa", "br_cdi_index"]},
           "Proteus": {"initial_date": datetime(2023, 9, 29),
                       "fund_name": "Persevera Proteus Ações FIA",
                       "benchmark": ["br_ibovespa", "br_smll", "br_cdi_index"]},}


def get_fund_peers(fund_name):
    peers = pd.read_excel(os.path.join(DATA_PATH, "persevera_peers.xlsx"), sheet_name=fund_name, index_col=0)
    peers = peers["short_name"].to_dict()
    return peers


def get_fund_data(fund_name, selected_peers, benchmark, start_date, end_date=datetime.today(), relative=False):
    logging.info(f'Loading data for {fund_name} and its peers since {start_date}...')
    listed_peers = get_fund_peers(fund_name)
    filtered_peers = {k: v for k, v in listed_peers.items() if v in selected_peers}

    df = pd.read_parquet(
        os.path.join(DATA_PATH, f"cvm-cotas_fundos-{fund_name.lower()}.parquet"),
        filters=[
            ('fund_cnpj', 'in', filtered_peers.keys()),
            ('date', '>=', start_date),
            ('date', '<=', end_date)
        ]
    )
    df = df.pivot(index='date', columns='fund_cnpj', values='fund_nav')
    df = df.rename(columns=filtered_peers)
    df = df.replace(0, np.nan).dropna(how='all')

    logging.info("Importing benchmark...")
    df_benchmark = pd.read_parquet(
        path=os.path.join(DATA_PATH, "consolidado-indicators.parquet"),
        filters=[('code', 'in', benchmark)]
    )
    df_benchmark = df_benchmark.pivot_table(index='date', columns='code', values='value')
    df_benchmark = df_benchmark.ffill()

    logging.info("Including benchmark to DataFrame...")
    df = pd.merge(left=df, right=df_benchmark, left_index=True, right_index=True, how='left')

    if relative:
        df = df.pct_change()
        df = df.apply(lambda x: x - df[benchmark[0]])
        df = (1 + df.drop(columns=benchmark)).cumprod()
        df.iloc[0] = 1
        df = df.ffill()

    df = df.rename(columns={"br_cdi_index": "CDI", "br_ibovespa": "Ibovespa", "br_smll": "SMLL"})
    return df


def get_performance_table(df, start_date, end_date, relative=False):
    time_frames = {
        'day': df.groupby(pd.Grouper(level='date', freq="1D")).last().pct_change().iloc[-1],
        'mtd': df.groupby(pd.Grouper(level='date', freq="1M")).last().pct_change().iloc[-1],
        'ytd': df.groupby(pd.Grouper(level='date', freq="Y")).last().pct_change().iloc[-1],
        '3m': (df[start_date:end_date].iloc[-1] / df[df[start_date:end_date].iloc[-1].name - relativedelta(months=3):end_date].iloc[0] - 1),
        '6m': (df[start_date:end_date].iloc[-1] / df[df[start_date:end_date].iloc[-1].name - relativedelta(months=6):end_date].iloc[0] - 1),
        '12m': (df[start_date:end_date].iloc[-1] / df[df[start_date:end_date].iloc[-1].name - relativedelta(months=12):end_date].iloc[0] - 1),
        'custom': df[start_date:end_date].iloc[-1] / df[start_date:end_date].iloc[0] - 1,

        'day_rank': df.groupby(pd.Grouper(level='date', freq="1D")).last().pct_change().iloc[-1].rank(ascending=False),
        'mtd_rank': df.groupby(pd.Grouper(level='date', freq="1M")).last().pct_change().iloc[-1].rank(ascending=False),
        'ytd_rank': df.groupby(pd.Grouper(level='date', freq="Y")).last().pct_change().iloc[-1].rank(ascending=False),
        '3m_rank': (df[start_date:end_date].iloc[-1] / df[df[start_date:end_date].iloc[-1].name - relativedelta(months=3):end_date].iloc[0] - 1).rank(ascending=False),
        '6m_rank': (df[start_date:end_date].iloc[-1] / df[df[start_date:end_date].iloc[-1].name - relativedelta(months=6):end_date].iloc[0] - 1).rank(ascending=False),
        '12m_rank': (df[start_date:end_date].iloc[-1] / df[df[start_date:end_date].iloc[-1].name - relativedelta(months=12):end_date].iloc[0] - 1).rank(ascending=False),
        'custom_rank': (df[start_date:end_date].iloc[-1] / df[start_date:end_date].iloc[0] - 1).rank(ascending=False),
    }
    df = pd.DataFrame(time_frames)
    if relative:
        if 'Ibovespa' in df.index:
            df = df.sub(df.loc['Ibovespa'])
            df = df.drop(index=['CDI', 'Ibovespa'])
        else:
            df = df.div(df.loc['CDI'])
            df = df.drop(index='CDI')
    return df


def format_table(df):
    return df.style.format({'day': '{:,.2%}'.format,
                            'mtd': '{:,.2%}'.format,
                            'ytd': '{:,.2%}'.format,
                            '3m': '{:,.2%}'.format,
                            '6m': '{:,.2%}'.format,
                            '12m': '{:,.2%}'.format,
                            'custom': '{:,.2%}'.format,
                            'day_rank': '{:.0f}'.format,
                            'mtd_rank': '{:.0f}'.format,
                            'ytd_rank': '{:.0f}'.format,
                            '3m_rank': '{:.0f}'.format,
                            '6m_rank': '{:.0f}'.format,
                            '12m_rank': '{:.0f}'.format,
                            'custom_rank': '{:.0f}'.format}
                           )


def format_chart(figure, connect_gaps=False):
    figure.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=False),
            type="date",
        ),
        yaxis_title=None, xaxis_title=None,
        yaxis=dict(autorange=True, fixedrange=False, griddash="dash", tickformat=".1%"),
        legend=dict(title=None, yanchor="top", orientation="v"),
        showlegend=True,
        hovermode="x unified",
    )
    figure.update_traces(connectgaps=connect_gaps, hovertemplate="%{y}")
    return figure


def calculate_drawdown(cum_returns: pd.Series) -> pd.Series:
    """Calculate drawdown from cumulative returns."""
    drawdown = (cum_returns / cum_returns.cummax() - 1)
    return drawdown


def show_fund_analysis():
    st.header("Fund Analysis")

    selected_fund = option_menu(
        menu_title=None,
        options=["Trinity", "Yield", "Phoenix", "Compass", "Nemesis", "Proteus"],
        orientation="horizontal"
    )

    st.header(selected_fund)

    cols = st.columns(2, gap='large')
    with cols[0]:
        nested_cols = st.columns(2)
        with nested_cols[0]:
            start_date = st.date_input(label="Selecione a data inicial:",
                                       value=de_para[selected_fund]["initial_date"],
                                       min_value=de_para[selected_fund]["initial_date"],
                                       max_value=datetime.today(),
                                       format="YYYY-MM-DD")
        with nested_cols[1]:
            end_date = st.date_input(label="Selecione a data final:",
                                     value=datetime.today(),
                                     min_value=start_date,
                                     max_value=datetime.today(),
                                     format="YYYY-MM-DD")

    with cols[1]:
        peers_list = list(get_fund_peers(selected_fund).values())
        peers_list.insert(0, 'Todos')
        selected_peers = st.multiselect(label='Selecione os peers:',
                                        options=peers_list,
                                        default=[de_para[selected_fund]["fund_name"]])
        if 'Todos' in selected_peers:
            selected_peers = peers_list

    st.subheader("Rentabilidade Acumulada")
    tab1, tab2 = st.tabs(["Absoluto", "Relativo"])

    # Tab1: Retorno absoluto
    with tab1:
        col1, col2 = st.columns(2, gap='large')

        with col1:
            data = get_fund_data(
                fund_name=selected_fund,
                start_date=start_date,
                end_date=end_date,
                selected_peers=selected_peers,
                benchmark=de_para[selected_fund]["benchmark"]
            )
            data = (1 + data.pct_change()).cumprod().sub(1)
            data.iloc[0] = 0
            data = data.ffill()
            fig = px.line(data)
            st.plotly_chart(format_chart(figure=fig, connect_gaps=True), use_container_width=True)

        with col2:
            st.write("Data mais recente:", data.index.max())
            table_data = get_fund_data(
                fund_name=selected_fund,
                start_date=de_para[selected_fund]["initial_date"],
                selected_peers=selected_peers,
                benchmark=de_para[selected_fund]["benchmark"]
            )
            table_data = table_data.ffill(limit=3)
            df = get_performance_table(table_data, start_date=start_date, end_date=end_date)
            st.dataframe(format_table(df), use_container_width=True)

    # Retorno relativo
    with tab2:
        col1, col2 = st.columns(2, gap='large')

        with col1:
            data = get_fund_data(
                fund_name=selected_fund,
                start_date=start_date,
                end_date=end_date,
                selected_peers=selected_peers,
                benchmark=de_para[selected_fund]["benchmark"],
                relative=True
            )
            data = (1 + data.pct_change()).cumprod().sub(1)
            data.iloc[0] = 0
            data = data.ffill()
            fig = px.line(data)
            st.plotly_chart(format_chart(figure=fig, connect_gaps=True), use_container_width=True)

        with col2:
            table_data = get_fund_data(
                fund_name=selected_fund,
                start_date=de_para[selected_fund]["initial_date"],
                selected_peers=selected_peers,
                benchmark=de_para[selected_fund]["benchmark"]
            )
            table_data = table_data.ffill(limit=3)
            df = get_performance_table(table_data, start_date=start_date, end_date=end_date, relative=True)
            st.dataframe(format_table(df), use_container_width=True)

    # Outras estatísticas
    data = get_fund_data(
        fund_name=selected_fund,
        start_date=de_para[selected_fund]["initial_date"],
        selected_peers=selected_peers,
        benchmark=de_para[selected_fund]["benchmark"]
    )
    cols_stats = st.columns(2, gap='large')
    with cols_stats[0]:
        st.subheader("Drawdown")
        df_drawdown = data.apply(calculate_drawdown)
        fig = px.line(df_drawdown)
        st.plotly_chart(format_chart(figure=fig, connect_gaps=True), use_container_width=True)

    with cols_stats[1]:
        st.subheader("Volatilidade Realizada")
        window = st.radio("Janela (dias):", options=[21, 42, 63], horizontal=True)
        df_volatility = data.pct_change().rolling(window).std() * np.sqrt(252)
        fig = px.line(df_volatility)
        st.plotly_chart(format_chart(figure=fig, connect_gaps=True), use_container_width=True)

    if selected_fund == "Trinity":
        pass
    elif selected_fund == "Nemesis":
        pass
