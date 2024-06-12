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


def get_stock_data(code: str, fields: list):
    df = pd.read_parquet("https://persevera.s3.sa-east-1.amazonaws.com/factor_zoo.parquet",
                         filters=[("code", "==", code)], columns=fields)
    return df


def get_performance_table(df, start_date, end_date):
    time_frames = {
        'day': df.groupby(pd.Grouper(level='date', freq="1D")).last().pct_change().iloc[-1],
        'mtd': df.groupby(pd.Grouper(level='date', freq="1M")).last().pct_change().iloc[-1],
        'ytd': df.groupby(pd.Grouper(level='date', freq="Y")).last().pct_change().iloc[-1],
        '3m': df.groupby(pd.Grouper(level='date', freq="1D")).last().pct_change(3 * 21).iloc[-1],
        '6m': df.groupby(pd.Grouper(level='date', freq="1D")).last().pct_change(6 * 21).iloc[-1],
        '12m': df.groupby(pd.Grouper(level='date', freq="1D")).last().pct_change(12 * 21).iloc[-1],
        'custom': df[start_date:end_date].iloc[-1] / df[start_date:end_date].iloc[0] - 1,
    }
    df = pd.DataFrame(time_frames)
    return df


def get_factor_performance(selected_factors, quantile, excess, start_date, end_date=datetime.today()):
    try:
        df = pd.read_parquet(os.path.join(DATA_PATH, "factors-quantile_performance.parquet"),
                             filters=[('date', '>=', start_date), ('date', '<=', end_date)])
        cols = [col for col in df.columns for factor in selected_factors if factor in col]
        cols = [col for col in cols if quantile in col]
        cols = [col for col in cols if 'excess' in col] if excess else [col for col in cols if 'excess' not in col]

        df = df.filter(cols)
        df = np.cumprod(1 + df.pct_change()).fillna(1)
        df.columns = df.columns.str.replace(f'_{quantile}', '')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    return df


def format_table(df):
    return df.style.format({'day': '{:,.2%}'.format,
                            'mtd': '{:,.2%}'.format,
                            'ytd': '{:,.2%}'.format,
                            '3m': '{:,.2%}'.format,
                            '6m': '{:,.2%}'.format,
                            '12m': '{:,.2%}'.format,
                            'custom': '{:,.2%}'.format}
                           )


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


def load_factor_data(selected_stocks):
    try:
        df = pd.read_parquet(os.path.join(DATA_PATH, "factors-signal_ranks.parquet"),
                             filters=[('code', 'in', selected_stocks)])
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

    df = df.set_index('code').iloc[:, 3:].T.reset_index()
    df = df.rename(columns={'index': 'signal'})
    df['factor_group'] = df['signal'].str.split('_').str[0]
    df['factor'] = df['signal'].str.split('_').str[2:].str.join('_')
    df = df[df['factor_group'] != "size"]
    return df


def create_polar_chart(df, selected_stocks, background_color, title="Factor Zoo"):
    fig = go.Figure()

    for stock, color in zip(selected_stocks, ['#16537e', '#722f37']):
        fig.add_trace(
            go.Scatterpolar(
                r=df[stock],
                theta=df['factor'],
                mode="lines+markers",
                name=stock,
                fill='toself',
                line_color=color,
            )
        )

    for factor_group, color in background_color.items():
        temp = df[df['factor_group'] == factor_group]
        fig.add_trace(
            go.Barpolar(
                r=[100] * len(temp),
                theta=temp['factor'],
                width=[1] * len(temp),
                marker_color=[color] * len(temp),
                marker_line_color="white",
                opacity=0.5,
                name=factor_group
            )
        )

    fig.update_polars(angularaxis_showgrid=True, radialaxis_showgrid=True)

    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True
    )

    return fig


def show_factor_playground():
    st.header("Factor Playground")

    selected_category = option_menu(
        menu_title=None,
        options=["Factor Radar", "Performance", "Backtester", "Universo"],
        orientation="horizontal"
    )

    if selected_category == "Factor Radar":
        cols_header = st.columns(2, gap='large')
        with cols_header[0]:
            stocks_info = pd.read_excel(os.path.join(DATA_PATH, "cadastro-base.xlsx"), sheet_name="equities").query(
                'code_exchange == "BZ"')
            selected_stocks = st.multiselect(label='Selecione as ações:',
                                             options=sorted(stocks_info['code']),
                                             default=["VALE3"],
                                             max_selections=2)

        # Define background colors for factor groups
        background_color = {
            'value': '#d4e8c6',
            'quality': '#7fd7f7',
            'momentum': '#fbe5d6',
            'risk': '#ffe080',
            'liquidity': '#ffff7f',
            'short': '#b11226',
        }

        df = load_factor_data(selected_stocks)
        if df is None:
            return

        fig = create_polar_chart(df, selected_stocks, background_color)
        st.plotly_chart(fig, use_container_width=True)

        cols = st.columns(2, gap='large')
        with cols[0]:
            fig_value = create_polar_chart(df[df['factor_group'] == "value"], selected_stocks, background_color,
                                           title="Value Factors")
            st.plotly_chart(fig_value, use_container_width=True)

            fig_value = create_polar_chart(df[df['factor_group'] == "momentum"], selected_stocks, background_color,
                                           title="Momentum Factors")
            st.plotly_chart(fig_value, use_container_width=True)

        with cols[1]:
            fig_value = create_polar_chart(df[df['factor_group'] == "quality"], selected_stocks, background_color,
                                           title="Quality Factors")
            st.plotly_chart(fig_value, use_container_width=True)

            fig_value = create_polar_chart(df[df['factor_group'] == "risk"], selected_stocks, background_color,
                                           title="Risk Factors")
            st.plotly_chart(fig_value, use_container_width=True)

    elif selected_category == "Performance":
        cols = st.columns(2, gap='large')
        with cols[0]:
            nested_cols = st.columns(2)
            with nested_cols[0]:
                start_date = st.date_input(label="Selecione a data inicial:",
                                           value=datetime(2008, 1, 1),
                                           min_value=datetime(2008, 1, 1),
                                           max_value=datetime.today(),
                                           format="YYYY-MM-DD")
            with nested_cols[1]:
                end_date = st.date_input(label="Selecione a data final:",
                                         value=datetime.today(),
                                         min_value=start_date,
                                         max_value=datetime.today(),
                                         format="YYYY-MM-DD")

        with cols[1]:
            factor_list = ['liquidity', 'momentum', 'quality', 'risk', 'short_interest', 'size', 'value',
                           'multi_factor']
            selected_factors = st.multiselect(label='Selecione os fatores:',
                                              options=factor_list,
                                              default=factor_list)

        st.subheader("Rentabilidade Acumulada")
        tabs = st.tabs(["Top", "L/S", "vs. Benchmark"])

        with tabs[0]:
            check_logy = st.checkbox("Log Scale", key=0)
            col1, col2 = st.columns(2, gap='large')

            with col1:
                data = get_factor_performance(selected_factors=selected_factors,
                                              quantile='rank_1',
                                              excess=False,
                                              start_date=start_date,
                                              end_date=end_date)

                data = data.sub(1)
                data = data.ffill()
                fig = px.line(data, log_y=check_logy)
                st.plotly_chart(format_chart(figure=fig, connect_gaps=True), use_container_width=True)

            with col2:
                st.write("Data mais recente:", data.index.max())
                table_data = get_factor_performance(selected_factors=selected_factors,
                                                    quantile='rank_1',
                                                    excess=False,
                                                    start_date=start_date)

                df = get_performance_table(table_data, start_date=start_date, end_date=end_date)
                st.dataframe(format_table(df), use_container_width=True)

        with tabs[1]:
            check_logy = st.checkbox("Log Scale", key=1)
            col1, col2 = st.columns(2, gap='large')

            with col1:
                data = get_factor_performance(selected_factors=selected_factors,
                                              quantile='long_short',
                                              excess=False,
                                              start_date=start_date,
                                              end_date=end_date)

                data = data.sub(1)
                data = data.ffill()
                fig = px.line(data, log_y=check_logy)
                st.plotly_chart(format_chart(figure=fig, connect_gaps=True), use_container_width=True)

            with col2:
                st.write("Data mais recente:", data.index.max())
                table_data = get_factor_performance(selected_factors=selected_factors,
                                                    quantile='long_short',
                                                    excess=False,
                                                    start_date=start_date)

                df = get_performance_table(table_data, start_date=start_date, end_date=end_date)
                st.dataframe(format_table(df), use_container_width=True)

        with tabs[2]:
            check_logy = st.checkbox("Log Scale", key=2)
            col1, col2 = st.columns(2, gap='large')

            with col1:
                data = get_factor_performance(selected_factors=selected_factors,
                                              quantile='rank_1',
                                              excess=True,
                                              start_date=start_date,
                                              end_date=end_date)

                data = data.sub(1)
                data = data.ffill()
                fig = px.line(data, log_y=check_logy)
                st.plotly_chart(format_chart(figure=fig, connect_gaps=True), use_container_width=True)

            with col2:
                st.write("Data mais recente:", data.index.max())
                table_data = get_factor_performance(selected_factors=selected_factors,
                                                    quantile='rank_1',
                                                    excess=True,
                                                    start_date=start_date)

                df = get_performance_table(table_data, start_date=start_date, end_date=end_date)
                st.dataframe(format_table(df), use_container_width=True)

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
