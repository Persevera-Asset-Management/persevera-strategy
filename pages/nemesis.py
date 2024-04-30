import pandas as pd
import logging, os
import streamlit as st
from datetime import datetime, timedelta
import polars as pl

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))


def get_fund_peers(fund_name):
    peers = pd.read_excel(PROJECT_PATH + "/peers.xlsx", sheet_name=fund_name, index_col=0)
    peers = peers["short_name"].to_dict()
    return peers


def get_fund_data(fund_name, start_date, selected_peers, relative=False):
    logging.info(f'Loading data for {fund_name} and its peers since {start_date}...')
    listed_peers = get_fund_peers(fund_name)
    filtered_peers = {k: v for k, v in listed_peers.items() if v in selected_peers}

    df = (
        pl.scan_parquet(source=PROJECT_PATH + f"/cvm-cotas_fundos-{fund_name.lower()}.parquet")
        .drop("fund_value")
        .filter(pl.col("fund_cnpj").is_in(filtered_peers.keys()))
        .filter(pl.col("date") >= start_date)
        .collect()
        .pivot(index='date', columns='fund_cnpj', values='fund_nav')
        .to_pandas()
    )
    df = df.rename(columns=filtered_peers)
    df = df.set_index('date')

    logging.info("Importing IBOV...")
    cdi = pd.read_parquet(
        path=PROJECT_PATH + "/indicators-equity.parquet",
        filters=[('code', '==', 'br_ibovespa')]
    ).pivot_table(index='date', columns='code', values='value')

    logging.info("Including IBOV to DataFrame...")
    df = pd.merge(left=df, right=cdi, left_index=True, right_index=True, how='left')
    df = df.rename(columns={"br_ibovespa": "IBOV"})

    if relative:
        df = df.pct_change()
        df = df.apply(lambda x: x - df['IBOV'])
        df = (1 + df.drop(columns='IBOV')).cumprod()
        df.iloc[0] = 1
        df = df.ffill()
    return df


def get_performance_table(df, relative=False):
    time_frames = {
        'day': df.groupby(pd.Grouper(level='date', freq="1D")).last().pct_change().iloc[-1],
        'mtd': df.groupby(pd.Grouper(level='date', freq="1M")).last().pct_change().iloc[-1],
        'ytd': df.groupby(pd.Grouper(level='date', freq="Y")).last().pct_change().iloc[-1],
        '3m': df.pct_change(3 * 21).iloc[-1],
        '6m': df.pct_change(6 * 21).iloc[-1],
        '12m': df.pct_change(12 * 21).iloc[-1],
    }
    df = pd.DataFrame(time_frames)
    if relative:
        df = df.sub(df.loc['IBOV'])
        df = df.drop(index='IBOV')
    return df


def format_chart(figure, connectgaps=False):
    figure.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=False),
            type="date",
        ),
        yaxis_title=None, xaxis_title=None,
        yaxis=dict(autorange=True, fixedrange=False, griddash="dash", tickformat=".1%"),
        legend=dict(title=None, yanchor="top", orientation="v"),
        showlegend=True,
    )
    return figure


def show_nemesis():
    st.header("Nêmesis")
    fund_name = "Nemesis"

    st.write(
        """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Mauris id diam 
        pharetra, dapibus est fermentum, laoreet diam. Integer vitae consequat augue:
        """
    )

    col1, col2 = st.columns(2, gap='large')
    with col1:
        start_date = st.date_input(label="Selecione a data inicial:",
                                   value=datetime(2022, 2, 25),
                                   min_value=datetime(2022, 2, 25),
                                   format="YYYY-MM-DD")

    with col2:
        peers_list = get_fund_peers(fund_name).values()
        selected_peers = st.multiselect(label='Selecione os peers:',
                                        options=peers_list,
                                        default=['Persevera Nemesis Total Return FIM'])

    st.subheader("Rentabilidade Acumulada")
    tab1, tab2 = st.tabs(["Absoluto", "Relativo"])

    # Tab1: Retorno absoluto
    with tab1:
        col1, col2 = st.columns(2, gap='large')

        with col1:
            data = get_fund_data(fund_name=fund_name, start_date=start_date, selected_peers=selected_peers)
            data = (1 + data.pct_change()).cumprod()
            data = data.sub(1)
            data.iloc[0] = 0
            data = data.ffill()
            fig = px.line(data)
            st.plotly_chart(format_chart(figure=fig, connectgaps=True), use_container_width=True)

        with col2:
            st.write("Data mais recente:", data.index.max())
            table_data = get_fund_data(fund_name=fund_name, start_date=datetime(2022, 2, 25), selected_peers=selected_peers)
            df = get_performance_table(table_data)

            st.dataframe(df
                         .style
                         .format({'day': '{:,.2%}'.format,
                                  'mtd': '{:,.2%}'.format,
                                  'ytd': '{:,.2%}'.format,
                                  '3m': '{:,.2%}'.format,
                                  '6m': '{:,.2%}'.format,
                                  '12m': '{:,.2%}'.format}),
                         use_container_width=True)

    # Retorno em excesso (CDI)
    with tab2:
        col1, col2 = st.columns(2, gap='large')

        with col1:
            data = get_fund_data(fund_name=fund_name, start_date=start_date, selected_peers=selected_peers, relative=True)
            data = (1 + data.pct_change()).cumprod()
            data = data.sub(1)
            data.iloc[0] = 0
            data = data.ffill()
            fig = px.line(data)
            st.plotly_chart(format_chart(figure=fig, connectgaps=True), use_container_width=True)

        with col2:
            table_data = get_fund_data(fund_name=fund_name, start_date=datetime(2022, 2, 25), selected_peers=selected_peers)
            df = get_performance_table(table_data, relative=True)

            st.dataframe(df
                         .style
                         .format({'day': '{:,.2%}'.format,
                                  'mtd': '{:,.2%}'.format,
                                  'ytd': '{:,.2%}'.format,
                                  '3m': '{:,.2%}'.format,
                                  '6m': '{:,.2%}'.format,
                                  '12m': '{:,.2%}'.format}),
                         use_container_width=True)
