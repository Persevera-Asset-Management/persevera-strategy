import pandas as pd
import numpy as np
import os
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')


def get_data(fields: list):
    df = pd.read_parquet(os.path.join(DATA_PATH, "consolidado-indicators.parquet"),
                         filters=[('code', 'in', fields)])
    df = df.pivot_table(index='date', columns='code', values='value')
    df = df.filter(fields)
    return df


def get_data_old(category: str, fields: list):
    df = pd.read_parquet(
        os.path.join(DATA_PATH, f"indicators-{category}.parquet"),
        filters=[('code', 'in', fields)]
    )
    # df = df.query('code == @fields')
    df = df.pivot_table(index='date', columns='code', values='value')
    df = df.filter(fields)
    return df


def get_cohort(assets: list, benchmark: str):
    individual_assets = get_data(fields=assets)
    individual_assets = individual_assets.filter(assets)
    cohort = individual_assets.iloc[:, 0] / individual_assets.iloc[:, 1]
    cohort.name = ' / '.join(individual_assets.columns)

    df_benchmark = get_data(fields=[benchmark])
    df = cohort.to_frame().merge(df_benchmark, left_index=True, right_index=True, how='left')
    df = df.dropna()
    return df


def get_index_fundamentals(codes: list, field: str):
    df = pd.read_parquet(os.path.join(DATA_PATH, f"index_fundamentals-equity.parquet"))
    df = df.query('code == @codes')
    df = df.pivot_table(index='date', columns='code', values=field)
    df = df.filter(codes)
    return df


def get_yield_curve(contract):
    df = pd.read_parquet(DATA_PATH + "/indicators-futures_curve.parquet",
                         filters=[('contract', '==', contract)])

    df = df.pivot_table(index='date_maturity', columns='date', values='yield')
    df = df.iloc[:, :-1]
    df = df.dropna(subset=[df.columns[-1]])

    df['30d_count'] = df.iloc[:, -31:-1].count(axis=1)
    df.loc[df['30d_count'] < 20] = np.nan
    df = df.drop(columns='30d_count')

    median = df.iloc[:, -31:-1].median(axis=1)
    min = df.iloc[:, -31:-1].min(axis=1)
    max = df.iloc[:, -31:-1].max(axis=1)

    df['30d_median'] = median
    df['30d_min'] = min
    df['30d_max'] = max
    df = df[df.columns[-4:]]
    return df


def create_line_chart(data, title, connect_gaps):
    fig = px.line(data)
    fig.update_layout(
        title=title,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(count=10, label="10y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=False),
            type="date",
        ),
        yaxis_title=None,
        xaxis_title=None,
        yaxis=dict(autorange=True, fixedrange=False, griddash="dash"),
        legend=dict(title=None, yanchor="top", orientation="h"),
        showlegend=True,
        hovermode="x unified",
    )
    fig.update_traces(connectgaps=connect_gaps, hovertemplate="%{y}")
    return fig


def create_bar_chart(data, title):
    fig = px.bar(data)
    fig.update_layout(
        title=title,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(count=10, label="10y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=False),
            type="date",
        ),
        yaxis_title=None,
        xaxis_title=None,
        yaxis=dict(autorange=True, fixedrange=False, griddash="dash"),
        legend=dict(title=None, yanchor="top", orientation="h"),
        showlegend=True,
    )
    return fig


def create_two_yaxis_line_chart(data, title, connect_gaps):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=data.index, y=data.iloc[:, 0], name=data.iloc[:, 0].name), secondary_y=False)
    fig.add_trace(go.Scatter(x=data.index, y=data.iloc[:, 1], name=data.iloc[:, 1].name), secondary_y=True)
    fig.update_layout(
        title=title,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(count=10, label="10y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=False),
            type="date",
        ),
        yaxis_title=None, xaxis_title=None,
        yaxis=dict(autorange=True, fixedrange=False, showgrid=False),
        legend=dict(title=None, yanchor="top", orientation="h"),
        showlegend=True,
        hovermode="x unified",
    )
    fig.update_traces(connectgaps=connect_gaps, hovertemplate="%{y}")
    return fig


def get_performance_table(df):
    time_frames = {
        'last': df.iloc[-1],
        'wtd': df.groupby(pd.Grouper(level='date', freq="1W-FRI")).last().pct_change().iloc[-1],
        'mtd': df.groupby(pd.Grouper(level='date', freq="1M")).last().pct_change().iloc[-1],
        '1m': df.groupby(pd.Grouper(level='date', freq="1D")).last().pct_change(1 * 21).iloc[-1],
        'ytd': df.groupby(pd.Grouper(level='date', freq="Y")).last().pct_change().iloc[-1],
        '1y': df.groupby(pd.Grouper(level='date', freq="1D")).last().pct_change(12 * 21).iloc[-1],
        '2y': df.groupby(pd.Grouper(level='date', freq="1D")).last().pct_change(2 * 12 * 21).iloc[-1],
    }
    df = pd.DataFrame(time_frames)
    return df


def format_table(df):
    return df.style \
        .format({'last': '{:,.2f}'.format,
                 'wtd': '{:,.2%}'.format,
                 'mtd': '{:,.2%}'.format,
                 '1m': '{:,.2%}'.format,
                 'ytd': '{:,.2%}'.format,
                 '1y': '{:,.2%}'.format,
                 '2y': '{:,.2%}'.format}) \
        .bar(align='zero', color=['#FCC0CB', '#90EE90'])


def show_chartbook():
    st.header("Chartbook")

    def display__oldchart_with_expander(expander_title, chart_titles, datasets, connect_gaps=False, two_yaxis=False):
        with st.expander(expander_title, expanded=False):
            num_cols = 2
            num_charts = len(chart_titles)
            num_rows = (num_charts + num_cols - 1) // num_cols

            for row in range(num_rows):
                cols = st.columns(num_cols, gap='large')
                start_index = row * num_cols
                end_index = min((row + 1) * num_cols, num_charts)

                for col, title, dataset in zip(cols, chart_titles[start_index:end_index],
                                               datasets[start_index:end_index]):
                    if two_yaxis:
                        col.plotly_chart(create_two_yaxis_line_chart(dataset, title, connect_gaps), use_container_width=True)
                    else:
                        col.plotly_chart(create_line_chart(dataset, title, connect_gaps), use_container_width=True)

    def display_chart_with_expander(expander_title, chart_titles, chart_types, datasets, connect_gaps=False, two_yaxis=False):
        with st.expander(expander_title, expanded=False):
            num_cols = 2
            num_charts = len(chart_titles)
            num_rows = (num_charts + num_cols - 1) // num_cols

            for row in range(num_rows):
                cols = st.columns(num_cols, gap='large')
                start_index = row * num_cols
                end_index = min((row + 1) * num_cols, num_charts)

                for col, title, chart_type, dataset in zip(cols, chart_titles[start_index:end_index],
                                                           chart_types[start_index:end_index],
                                                           datasets[start_index:end_index]):
                    if chart_type == 'line':
                        if two_yaxis:
                            col.plotly_chart(create_two_yaxis_line_chart(dataset, title, connect_gaps), use_container_width=True)
                        else:
                            col.plotly_chart(create_line_chart(dataset, title, connect_gaps), use_container_width=True)
                    elif chart_type == 'bar':
                        col.plotly_chart(create_bar_chart(dataset, title), use_container_width=True)


    def display_table_with_expander(expander_title, table_titles, datasets):
        with st.expander(expander_title, expanded=False):
            num_cols = 2
            num_charts = len(table_titles)
            num_rows = (num_charts + num_cols - 1) // num_cols

            for row in range(num_rows):
                cols = st.columns(num_cols, gap='large')
                start_index = row * num_cols
                end_index = min((row + 1) * num_cols, num_charts)

                for col, title, dataset in zip(cols, table_titles[start_index:end_index],
                                               datasets[start_index:end_index]):
                    col.markdown(f"**{title}**")
                    table = get_performance_table(dataset)
                    col.dataframe(format_table(table), use_container_width=True)

    selected_category = option_menu(
        menu_title=None,
        options=["Estados Unidos", "Brasil", "Juros", "Commodities", "Moedas", "Mercados", "Posicionamento", "Cohorts"],
        icons=['globe', 'globe-americas', "clipboard2-pulse", 'tree', 'graph-up', 'pie-chart', 'broadcast-pin', 'arrow-left-right'],
        orientation="horizontal"
    )

    if selected_category == "Estados Unidos":
        display_chart_with_expander(
            "PIB",
            ["PIB", "PIB (QoQ)", "PIB (YoY)"],
            ['line', 'bar', 'bar'],
            [
                get_data(fields=['us_gdp_index']),
                get_data(fields=['us_gdp_yoy']),
                get_data(fields=['us_gdp_qoq']),
            ]
        )

        display_chart_with_expander(
            "Taxas Referenciais",
            ["Curva Pré (Treasuries)", "Curva Inflação (TIPS)", "Curva Implícita (Breakeven)", "Curva de Juros", "Inclinações"],
            ['line', 'line', 'line', 'line', 'line'],
            [
                get_data(fields=['us_generic_2y', 'us_generic_5y', 'us_generic_10y', 'us_generic_30y']),
                get_data(fields=['us_generic_inflation_5y', 'us_generic_inflation_10y', 'us_generic_inflation_20y',
                                 'us_generic_inflation_30y']),
                get_data(
                    fields=['us_breakeven_2y', 'us_breakeven_5y', 'us_breakeven_10y', 'usd_inflation_swap_fwd_5y5y']),
                get_yield_curve(contract='us_fed_funds_curve'),
                get_data(fields=['us_2y10y_steepness', 'us_5y10y_steepness', 'us_5y30y_steepness']),
            ],
            connect_gaps=True
        )

        display_chart_with_expander(
            "Taxas Corporativas",
            ["IG Spreads", "IG Taxas", "HY Spreads", "HY Taxas"],
            ['line', 'line', 'line', 'line'],
            [
                get_data(fields=['us_corporate_ig_5y_spread', 'us_corporate_ig_10y_spread']),
                get_data(fields=['us_corporate_ig_5y_yield', 'us_corporate_ig_10y_yield']),
                get_data(fields=['us_corporate_hy_5y_spread', 'us_corporate_hy_10y_spread']),
                get_data(fields=['us_corporate_hy_5y_yield', 'us_corporate_hy_10y_yield'])
            ]
        )

        display_chart_with_expander(
            "Inflação",
            ["Índices de Inflação"],
            ['line'],
            [
                get_data(fields=['us_cpi_yoy', 'us_core_cpi_yoy', 'us_pce_yoy', 'us_supercore_cpi_yoy']),
            ]
        )

    elif selected_category == "Brasil":
        display_chart_with_expander(
            "PIB",
            ["PIB", "PIB (% YoY)", "PIB (% QoQ)"],
            ['line', 'bar', 'bar'],
            [
                get_data(fields=['br_gdp_index']),
                get_data(fields=['br_gdp_yoy']),
                get_data(fields=['br_gdp_qoq']),
            ]
        )

        display_chart_with_expander(
            "Taxas Referenciais",
            ["Curva Pré", "Curva IPCA", "Curva Implícita", "Curva de Juros"],
            ['line', 'line', 'line', 'line'],
            [
                get_data(fields=['br_pre_1y', 'br_pre_2y', 'br_pre_3y', 'br_pre_5y', 'br_pre_10y']),
                get_data(fields=['br_ipca_1y', 'br_ipca_2y', 'br_ipca_3y', 'br_ipca_5y', 'br_ipca_10y', 'br_ipca_35y']),
                get_data(fields=['br_breakeven_1y', 'br_breakeven_2y', 'br_breakeven_3y', 'br_breakeven_5y',
                                 'br_breakeven_10y']),
                get_yield_curve('br_di_curve')
            ],
            connect_gaps=True
        )

        display_chart_with_expander(
            "Inflação",
            ["Índices de Inflação"],
            ['line'],
            [
                get_data(fields=['br_ipca_yoy']),
            ]
        )

        display_chart_with_expander(
            "Termos de Troca",
            ["Citi", "MDIC"],
            ['line', 'line'],
            [
                get_data(fields=['br_citi_terms_of_trade_index', 'br_current_account_to_gdp']),
                get_data(fields=['br_mdic_terms_of_trade_index', 'br_current_account_to_gdp']),
            ],
            two_yaxis=True
        )

    elif selected_category == "Juros":
        display_chart_with_expander(
            "Desenvolvidos",
            ["Taxa de 1 ano", "Taxa de 5 anos", "Taxa de 1 ano", "Taxa de 5 anos"],
            ['line', 'line', 'line', 'line'],
            [
                get_data(fields=['germany_generic_1y', 'spain_generic_1y', 'france_generic_1y', 'italy_generic_1y',
                                 'japan_generic_1y',
                                 'switzerland_generic_1y', 'sweden_generic_1y']),
                get_data(fields=['germany_generic_5y', 'spain_generic_5y', 'france_generic_5y', 'italy_generic_5y',
                                 'japan_generic_5y',
                                 'switzerland_generic_5y', 'sweden_generic_5y']),
                get_data(fields=['new_zealand_generic_1y', 'australia_generic_1y', 'canada_generic_1y',
                                 'norway_generic_1y',
                                 'us_generic_1y', 'uk_generic_1y']),
                get_data(fields=['new_zealand_generic_5y', 'australia_generic_5y', 'canada_generic_5y',
                                 'norway_generic_5y',
                                 'us_generic_5y', 'uk_generic_5y']),
            ]
        )

        display_chart_with_expander(
            "Emergentes",
            ["Taxa de 1 ano", "Taxa de 5 anos", "Taxa de 1 ano", "Taxa de 5 anos"],
            ['line', 'line', 'line', 'line'],
            [
                get_data(fields=['china_generic_1y', 'chile_generic_1y', 'colombia_generic_1y', 'hungary_generic_1y',
                                 'poland_generic_1y', 'peru_generic_1y']),
                get_data(fields=['china_generic_5y', 'chile_generic_5y', 'colombia_generic_5y', 'hungary_generic_5y',
                                 'poland_generic_5y', 'peru_generic_5y']),
                get_data(fields=['south_africa_generic_1y', 'russia_generic_1y', 'br_generic_1y', 'mexico_generic_1y',
                                 'india_generic_1y', 'indonesia_generic_1y', 'turkey_generic_1y']),
                get_data(fields=['south_africa_generic_5y', 'russia_generic_5y', 'br_generic_5y', 'mexico_generic_5y',
                                 'india_generic_5y', 'indonesia_generic_5y', 'turkey_generic_1y']),
            ]
        )

    elif selected_category == "Commodities":
        display_table_with_expander(
            "Performance",
            ["Energia", "Metais"],
            [
                get_data(fields=['crude_oil_wti', 'crude_oil_brent', 'gasoline', 'usda_diesel', 'natural_gas', 'thermal_coal']).fillna(method='ffill', limit=2),
                get_data(fields=['gold', 'silver', 'lme_aluminum', 'lme_copper', 'lme_nickel_cash', 'sgx_iron_ore_62', 'platinum', 'palladium', 'lme_zinc_spot', 'coking_coal']).fillna(method='ffill', limit=2),
            ]
        )

        display_chart_with_expander(
            "Commodity Research Bureau (CRB)",
            ["Índice CRB", "Índice CRB (% 12 meses)"],
            [
                get_data(fields=['crb_index', 'crb_fats_oils_index', 'crb_food_index', 'crb_livestock_index', 'crb_metals_index', 'crb_raw_industrials_index', 'crb_textiles_index']),
                get_data(fields=['crb_index']).pct_change(252).dropna(),
            ],
            connect_gaps=True
        )
        
        display_chart_with_expander(
            "Fretes",
            ["XXX"],
            [
                get_data(fields=['baltic_dry_index', 'shanghai_containerized_freight_index'])
            ],
            connect_gaps=True
        )

        display_chart_with_expander(
            "Combustível",
            ["Atacado", "Varejo"],
            [
                get_data(fields=['crude_oil_brent', 'crude_oil_wti', 'gasoline', 'usda_diesel']),
                get_data(fields=['br_anp_gasoline_retail', 'br_anp_diesel_retail', 'br_anp_hydrated_ethanol_retail',
                                 'br_anp_lpg_retail']),
            ],
            connect_gaps=True
        )

    elif selected_category == "Moedas":
        display_table_with_expander(
            "Performance",
            ["Desenvolvidos", "Emergentes"],
            [
                get_data(
                    fields=['twd_usd', 'bloomberg_dollar_index', 'eur_usd', 'jpy_usd', 'gbp_usd', 'chf_usd', 'cad_usd', 'aud_usd', 'nok_usd', 'sek_usd']).fillna(method='ffill', limit=2),
                get_data(
                    fields=['brl_usd', 'mxn_usd', 'clp_usd', 'zar_usd', 'try_usd', 'cnh_usd']).fillna(method='ffill', limit=2),
            ]
        )

    elif selected_category == "Mercados":
        st.empty()

        display_chart_with_expander(
            "EPS",
            ["S&P 500", "Ibovespa"],
            [
                get_index_fundamentals(codes=['us_sp500'], field='earnings_per_share_fwd'),
                get_index_fundamentals(codes=['br_ibovespa'], field='earnings_per_share_fwd'),
            ]
        )

        display_chart_with_expander(
            "P/E",
            ["Desenvolvidos", "Desenvolvidos (vs. S&P 500)", "Emergentes", "Emergentes (vs. S&P 500)"],
            [
                get_index_fundamentals(codes=['us_sp500', 'us_russell2000', 'us_nasdaq100', 'germany_dax40', 'japan_nikkei225', 'uk_ukx'], field='price_to_earnings_fwd'),
                get_index_fundamentals(
                    codes=['us_sp500', 'us_russell2000', 'us_nasdaq100', 'germany_dax40', 'japan_nikkei225', 'uk_ukx'],
                    field='price_to_earnings_fwd').apply(lambda x: x / x['us_sp500'], axis=1).drop(columns='us_sp500'),
                get_index_fundamentals(codes=['br_ibovespa', 'china_csi300', 'south_africa_top40', 'mexico_bmv', 'chile_ipsa', 'india_nifty50', 'indonesia_jci'], field='price_to_earnings_fwd'),
                get_index_fundamentals(
                    codes=['us_sp500', 'br_ibovespa', 'china_csi300', 'south_africa_top40', 'mexico_bmv', 'chile_ipsa', 'india_nifty50', 'indonesia_jci'],
                    field='price_to_earnings_fwd').apply(lambda x: x / x['us_sp500'], axis=1).drop(columns='us_sp500').dropna(how='all'),
            ],
            connect_gaps=True
        )

    elif selected_category == "Posicionamento":
        display_chart_with_expander(
            "Treasuries",
            ["Treasury 2Y", "Treasury 5Y", "Treasury 10Y", "Treasury Bonds"],
            ['line', 'line', 'line', 'line'],
            [
                get_data(fields=['cftc_cbt_treasury_2y']),
                get_data(fields=['cftc_cbt_treasury_5y']),
                get_data(fields=['cftc_cbt_treasury_10y']),
                get_data(fields=['cftc_cbt_treasury_bonds']),
            ]
        )

        display_chart_with_expander(
            "Commodities",
            ["Copper", "Gold", "Silver", "Crude Oil"],
            ['line', 'line', 'line', 'line'],
            [
                get_data(fields=['cftc_cmx_copper']),
                get_data(fields=['cftc_cmx_gold']),
                get_data(fields=['cftc_cmx_silver']),
                get_data(fields=['cftc_nyme_crude_oil']),
            ]
        )

        display_chart_with_expander(
            "Moedas",
            ["AUD", "BRL", "CAD", "CHF", "EUR", "GBP", "JPY", "MXN", "NZD", "RUB", "ZAR"],
            ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line'],
            [
                get_data(fields=['cftc_cme_aud']),
                get_data(fields=['cftc_cme_brl']),
                get_data(fields=['cftc_cme_cad']),
                get_data(fields=['cftc_cme_chf']),
                get_data(fields=['cftc_cme_eur']),
                get_data(fields=['cftc_cme_gbp']),
                get_data(fields=['cftc_cme_jpy']),
                get_data(fields=['cftc_cme_mxn']),
                get_data(fields=['cftc_cme_nzd']),
                get_data(fields=['cftc_cme_rub']),
                get_data(fields=['cftc_cme_zar']),
            ]
        )

        display_chart_with_expander(
            "Bolsas",
            ["S&P 500", "Nasdaq", "Nikkei", "Russell 2000"],
            ['line', 'line', 'line', 'line'],
            [
                get_data(fields=['cftc_cme_sp500']),
                get_data(fields=['cftc_cme_nasdaq']),
                get_data(fields=['cftc_cme_nikkei']),
                get_data(fields=['cftc_cme_russell2000']),
            ]
        )

    elif selected_category == "Cohorts":
        display_chart_with_expander(
            "Estados Unidos",
            ['SOXX vs SPY', 'Discretionary vs Staples', 'VIX3M vs VIX', 'High Beta vs Low Volatility'],
            ['line', 'line', 'line', 'line', 'line'],
            [
                get_cohort(assets=['us_semiconductor_soxx', 'us_sp500'], benchmark='us_sp500'),
                get_cohort(assets=['us_ew_discretionary_rspd', 'us_ew_staples_rspd'], benchmark='us_sp500'),
                get_cohort(assets=['us_semiconductor_soxx', 'us_sp500'], benchmark='us_sp500'),
                get_cohort(assets=['us_vix3m', 'us_vix'], benchmark='us_sp500'),
                get_cohort(assets=['us_high_beta_sphb', 'us_low_volatility_usmv'], benchmark='us_sp500'),
            ],
            two_yaxis=True
        )
