import pandas as pd
import logging, os
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))


def get_data(category: str, fields: list):
    logging.info(f'Importing data from {fields}')
    df = pd.read_parquet(PROJECT_PATH + f"/indicators-{category}.parquet")
    df = df.query('code == @fields')
    df = df.pivot_table(index='date', columns='code', values='value')
    return df


def get_yield_curve(contract):
    df = pd.read_parquet(PROJECT_PATH + "/indicators-futures_curve.parquet",
                         filters=[('contract', '==', contract)])

    df = df.pivot_table(index='date_maturity', columns='date', values='yield')
    df = df.iloc[:, :-1]
    df = df.dropna(subset=[df.columns[-1]])

    df['30d_median'] = df.iloc[:, -31:-1].median(axis=1)
    df['30d_min'] = df.iloc[:, -31:-1].min(axis=1)
    df['30d_max'] = df.iloc[:, -31:-1].max(axis=1)
    df = df[df.columns[-4:]]
    return df


def format_chart(figure, title, connectgaps=False):
    figure.update_layout(
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
        yaxis=dict(autorange=True, fixedrange=False, griddash="dash"),
        legend=dict(title=None, yanchor="top", orientation="h"),
        showlegend=True,
    )
    figure.update_traces(connectgaps=connectgaps)
    return figure


def show_reuniao_estrategia():
    st.header("Reunião de Estratégia")
    st.write(
        """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Mauris id diam 
        pharetra, dapibus est fermentum, laoreet diam. Integer vitae consequat augue:
        """
    )

    # Taxas Corporativas
    # TODO: Substituir por subplots (mais eficiente)
    with st.expander("Taxas Corporativas (US)"):
        fig_corporate = [
            px.line(get_data(category='macro', fields=['us_corporate_ig_5y_spread', 'us_corporate_ig_10y_spread'])),
            px.line(get_data(category='macro', fields=['us_corporate_ig_5y_yield', 'us_corporate_ig_10y_yield'])),
            px.line(get_data(category='macro', fields=['us_corporate_hy_5y_spread', 'us_corporate_hy_10y_spread'])),
            px.line(get_data(category='macro', fields=['us_corporate_hy_5y_yield', 'us_corporate_hy_10y_yield']))
        ]

        col1, col2 = st.columns(2, gap='large')

        with col1:
            st.plotly_chart(format_chart(figure=fig_corporate[0], title="IG Spreads"), use_container_width=True)
            st.plotly_chart(format_chart(figure=fig_corporate[1], title="IG Taxas"), use_container_width=True)

        with col2:
            st.plotly_chart(format_chart(figure=fig_corporate[2], title="HY Spreads"), use_container_width=True)
            st.plotly_chart(format_chart(figure=fig_corporate[3], title="HY Taxas"), use_container_width=True)

    # Taxas de Juros
    with st.expander("Taxas de Juros (US)"):
        fig_rates_us = [
            px.line(get_data(category='macro', fields=['us_generic_2y', 'us_generic_5y', 'us_generic_10y', 'us_generic_30y'])),
            px.line(get_data(category='macro', fields=['us_2y10y_steepness', 'us_5y10y_steepness', 'us_5y30y_steepness'])),
            px.line(get_yield_curve(contract='us_fed_funds_curve'), line_shape='spline')
        ]

        col1, col2 = st.columns(2, gap='large')

        with col1:
            st.plotly_chart(format_chart(figure=fig_rates_us[0], title="Treasuries"), use_container_width=True)
            st.plotly_chart(format_chart(figure=fig_rates_us[2], title="Fed Funds Futures"), use_container_width=True)

        with col2:
            st.plotly_chart(format_chart(figure=fig_rates_us[1], title="Inclinações"), use_container_width=True)

    # Taxas Reais e Implícitas
    with st.expander("Taxas Reais e Implícitas (US)"):
        fig_rates_implicit = [
            px.line(get_data(category='macro', fields=['us_generic_2y', 'us_generic_5y', 'us_generic_10y', 'us_generic_30y'])),
            px.line(get_data(category='macro', fields=['us_generic_inflation_5y', 'us_generic_inflation_10y', 'us_generic_inflation_20y',
                              'us_generic_inflation_30y'])),
            px.line(get_data(category='macro', fields=['us_breakeven_2y', 'us_breakeven_5y', 'us_breakeven_10y', 'usd_inflation_swap_fwd_5y5y']))
        ]

        col1, col2 = st.columns(2, gap='large')

        with col1:
            st.plotly_chart(format_chart(figure=fig_rates_implicit[0], title="Treasuries", connectgaps=True),
                            use_container_width=True)
            st.plotly_chart(format_chart(figure=fig_rates_implicit[2], title="Breakevens", connectgaps=True),
                            use_container_width=True)

        with col2:
            st.plotly_chart(format_chart(figure=fig_rates_implicit[1], title="TIPS", connectgaps=True),
                            use_container_width=True)

    # Trajetória da Inflação
    with st.expander("Trajetória da Inflação"):
        fig_inflation_path = [
            px.line(get_data(category='macro', fields=['us_cpi_yoy'])).add_hline(y=2, line_width=2, line_dash="dash", line_color="black"),
            px.line(get_data(category='macro', fields=['us_core_cpi_yoy'])).add_hline(y=2, line_width=2, line_dash="dash", line_color="black"),
            px.line(get_data(category='macro', fields=['br_ipca_yoy'])).add_hline(y=3.5, line_width=2, line_dash="dash", line_color="black")
        ]

        col1, col2 = st.columns(2, gap='large')

        with col1:
            st.plotly_chart(format_chart(figure=fig_inflation_path[0], title="CPI", connectgaps=True),
                            use_container_width=True)
            st.plotly_chart(format_chart(figure=fig_inflation_path[2], title="IPCA", connectgaps=True),
                            use_container_width=True)

        with col2:
            st.plotly_chart(format_chart(figure=fig_inflation_path[1], title="Core CPI", connectgaps=True),
                            use_container_width=True)

    # Trajetória do PIB
    with st.expander("Trajetória do PIB"):
        fig_gdp_path = [
            px.line(get_data(category='macro', fields=['us_gdp_yoy'])).add_hline(y=3, line_width=2, line_dash="dash", line_color="black"),
            px.line(get_data(category='macro', fields=['br_gdp_yoy'])).add_hline(y=3, line_width=2, line_dash="dash", line_color="black"),
        ]

        col1, col2 = st.columns(2, gap='large')

        with col1:
            st.plotly_chart(format_chart(figure=fig_gdp_path[0], title="PIB US", connectgaps=True),
                            use_container_width=True)

        with col2:
            st.plotly_chart(format_chart(figure=fig_gdp_path[1], title="PIB Brasil", connectgaps=True),
                            use_container_width=True)

    # DM Rates
    with st.expander("DM Rates"):
        fig_rates_dm = [
            px.line(get_data(category='macro', fields=
                ['germany_generic_1y', 'spain_generic_1y', 'france_generic_1y', 'italy_generic_1y', 'japan_generic_1y',
                 'switzerland_generic_1y', 'sweden_generic_1y'])),
            px.line(get_data(category='macro', fields=
                ['germany_generic_5y', 'spain_generic_5y', 'france_generic_5y', 'italy_generic_5y', 'japan_generic_5y',
                 'switzerland_generic_5y', 'sweden_generic_5y'])),
            px.line(get_data(category='macro', fields=
                ['new_zealand_generic_1y', 'australia_generic_1y', 'canada_generic_1y', 'norway_generic_1y',
                 'us_generic_1y', 'uk_generic_1y'])),
            px.line(get_data(category='macro', fields=
                ['new_zealand_generic_5y', 'australia_generic_5y', 'canada_generic_5y', 'norway_generic_5y',
                 'us_generic_5y', 'uk_generic_5y'])),
        ]

        col1, col2 = st.columns(2, gap='large')

        with col1:
            st.plotly_chart(format_chart(figure=fig_rates_dm[0], title="Taxa de 1 ano"),
                            use_container_width=True)
            st.plotly_chart(format_chart(figure=fig_rates_dm[1], title="Taxa de 5 anos"),
                            use_container_width=True)

        with col2:
            st.plotly_chart(format_chart(figure=fig_rates_dm[2], title="Taxa de 1 ano"),
                            use_container_width=True)
            st.plotly_chart(format_chart(figure=fig_rates_dm[3], title="Taxa de 5 anos"),
                            use_container_width=True)

    # EM Rates
    with st.expander("EM Rates"):
        fig_rates_em = [
            px.line(get_data(category='macro', fields=['china_generic_1y', 'chile_generic_1y', 'colombia_generic_1y', 'hungary_generic_1y',
                              'poland_generic_1y', 'peru_generic_1y'])),
            px.line(get_data(category='macro', fields=['china_generic_5y', 'chile_generic_5y', 'colombia_generic_5y', 'hungary_generic_5y',
                              'poland_generic_5y', 'peru_generic_5y'])),
            px.line(get_data(category='macro', fields=['south_africa_generic_1y', 'russia_generic_1y', 'br_generic_1y', 'mexico_generic_1y',
                              'india_generic_1y', 'indonesia_generic_1y', 'turkey_generic_1y'])),
            px.line(get_data(category='macro', fields=['south_africa_generic_5y', 'russia_generic_5y', 'br_generic_5y', 'mexico_generic_5y',
                              'india_generic_5y', 'indonesia_generic_5y', 'turkey_generic_1y'])),
        ]

        col1, col2 = st.columns(2, gap='large')

        with col1:
            st.plotly_chart(format_chart(figure=fig_rates_em[0], title="Taxa de 1 ano"),
                            use_container_width=True)
            st.plotly_chart(format_chart(figure=fig_rates_em[1], title="Taxa de 5 anos"),
                            use_container_width=True)

        with col2:
            st.plotly_chart(format_chart(figure=fig_rates_em[2], title="Taxa de 1 ano"),
                            use_container_width=True)
            st.plotly_chart(format_chart(figure=fig_rates_em[3], title="Taxa de 5 anos"),
                            use_container_width=True)

    # Juros Históricos
    with st.expander("Juros Históricos"):
        temp = get_data(category='macro', fields=['br_pre_1y', 'br_pre_2y', 'br_pre_3y', 'br_pre_5y', 'br_pre_10y', 'br_ipca_1y',
                                       'br_ipca_2y', 'br_ipca_3y', 'br_ipca_5y', 'br_ipca_10y', 'br_ipca_35y'])
        for vertice in ['_1y', '_2y', '_3y', '_5y', '_10y']:
            cols = temp.filter(like=vertice).columns
            temp[f'br_implicita{vertice}'] = temp.eval(f"((1 + {cols[1]}/100) / (1 + {cols[0]}/100) - 1) * 100")

        fig_rates_br = [
            px.line(temp[['br_pre_1y', 'br_pre_2y', 'br_pre_3y', 'br_pre_5y', 'br_pre_10y']]),
            px.line(temp[['br_ipca_1y', 'br_ipca_2y', 'br_ipca_3y', 'br_ipca_5y', 'br_ipca_10y', 'br_ipca_35y']].dropna(how='all')),
            px.line(temp[['br_implicita_1y', 'br_implicita_2y', 'br_implicita_3y', 'br_implicita_5y', 'br_implicita_10y']].dropna(how='all')),
            px.line(get_yield_curve('br_di_curve'), line_shape='spline')
        ]

        col1, col2 = st.columns(2, gap='large')

        with col1:
            st.plotly_chart(format_chart(figure=fig_rates_br[0], title="Curva Pré", connectgaps=True),
                            use_container_width=True)
            st.plotly_chart(format_chart(figure=fig_rates_br[2], title="Curva Implícita", connectgaps=True),
                            use_container_width=True)

        with col2:
            st.plotly_chart(format_chart(figure=fig_rates_br[1], title="Curva IPCA", connectgaps=True),
                            use_container_width=True)
            st.plotly_chart(format_chart(figure=fig_rates_br[3], title="DI Futures", connectgaps=True),
                            use_container_width=True)

    # CRB e Fretes
    with st.expander("CRB e Fretes"):
        # CRB
        temp = get_data(category='macro', fields=['crb_index', 'brl_usd'])
        temp['crb_index_brl'] = temp['crb_index'] * temp['brl_usd']

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=temp.index, y=temp['crb_index'], name="crb_index"), secondary_y=False)
        fig.add_trace(go.Scatter(x=temp.index, y=temp['crb_index_brl'], name="crb_index_brl"), secondary_y=True)

        fig_crb_freight = [
            fig,
            px.line(temp['crb_index'].pct_change(252)),
            px.line(get_data(category='', fields=['baltic_dry_index', 'shanghai_containerized_freight_index'])),
        ]

        col1, col2 = st.columns(2, gap='large')

        with col1:
            st.plotly_chart(format_chart(figure=fig_crb_freight[0], title="Índice CRB", connectgaps=True),
                            use_container_width=True)
            st.plotly_chart(format_chart(figure=fig_crb_freight[2], title="Fretes", connectgaps=True),
                            use_container_width=True)

        with col2:
            st.plotly_chart(format_chart(figure=fig_crb_freight[1], title="Índice CRB (% 12 meses)", connectgaps=True),
                            use_container_width=True)
