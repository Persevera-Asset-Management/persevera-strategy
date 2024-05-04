import pandas as pd
import os
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')


def get_data(category: str, fields: list):
    df = pd.read_parquet(os.path.join(DATA_PATH, f"indicators-{category}.parquet"))
    df = df.query('code == @fields')
    df = df.pivot_table(index='date', columns='code', values='value')
    return df


def get_yield_curve(contract):
    df = pd.read_parquet(DATA_PATH + "/indicators-futures_curve.parquet",
                         filters=[('contract', '==', contract)])

    df = df.pivot_table(index='date_maturity', columns='date', values='yield')
    df = df.iloc[:, :-1]
    df = df.dropna(subset=[df.columns[-1]])

    df['30d_median'] = df.iloc[:, -31:-1].median(axis=1)
    df['30d_min'] = df.iloc[:, -31:-1].min(axis=1)
    df['30d_max'] = df.iloc[:, -31:-1].max(axis=1)
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
        yaxis_title=None, xaxis_title=None,
        yaxis=dict(autorange=True, fixedrange=False, griddash="dash"),
        legend=dict(title=None, yanchor="top", orientation="h"),
        showlegend=True,
        hovermode="x unified",
        hovertemplate="%{code}%{value}"
    )
    fig.update_traces(connectgaps=connect_gaps)
    return fig


def show_others():
    st.header("Chartbook")

    def display_chart_with_expander(expander_title, chart_titles, datasets, connect_gaps=False):
        with st.expander(expander_title):
            num_cols = 2
            num_charts = len(chart_titles)
            num_rows = (num_charts + num_cols - 1) // num_cols

            for row in range(num_rows):
                cols = st.columns(num_cols, gap='large')
                start_index = row * num_cols
                end_index = min((row + 1) * num_cols, num_charts)

                for col, title, dataset in zip(cols, chart_titles[start_index:end_index],
                                               datasets[start_index:end_index]):
                    col.plotly_chart(create_line_chart(dataset, title, connect_gaps), use_container_width=True)

    selected_category = option_menu(
        menu_title=None,
        options=["United States", "Brazil", "Global", "Commodities", "Markets"],
        icons=['globe', 'table', "list-task", 'graph-up', 'graph-up'],
        orientation="horizontal"
    )

    if selected_category == "United States":
        display_chart_with_expander(
            "Taxas Corporativas (US)",
            ["IG Spreads", "IG Taxas", "HY Spreads", "HY Taxas"],
            [
                get_data(category='macro', fields=['us_corporate_ig_5y_spread', 'us_corporate_ig_10y_spread']),
                get_data(category='macro', fields=['us_corporate_ig_5y_yield', 'us_corporate_ig_10y_yield']),
                get_data(category='macro', fields=['us_corporate_hy_5y_spread', 'us_corporate_hy_10y_spread']),
                get_data(category='macro', fields=['us_corporate_hy_5y_yield', 'us_corporate_hy_10y_yield'])
            ]
        )

        display_chart_with_expander(
            "Taxas de Juros (US)",
            ["Treasuries", "Inclinações", "Fed Funds Futures"],
            [
                get_data(category='macro', fields=['us_generic_2y', 'us_generic_5y', 'us_generic_10y', 'us_generic_30y']),
                get_data(category='macro', fields=['us_2y10y_steepness', 'us_5y10y_steepness', 'us_5y30y_steepness']),
                get_yield_curve(contract='us_fed_funds_curve')
            ]
        )

        display_chart_with_expander(
            "Taxas Reais e Implícitas (US)",
            ["Treasuries", "TIPS", "Breakevens"],
            [
                get_data(category='macro', fields=['us_generic_2y', 'us_generic_5y', 'us_generic_10y', 'us_generic_30y']),
                get_data(category='macro',
                         fields=['us_generic_inflation_5y', 'us_generic_inflation_10y', 'us_generic_inflation_20y',
                                 'us_generic_inflation_30y']),
                get_data(category='macro',
                         fields=['us_breakeven_2y', 'us_breakeven_5y', 'us_breakeven_10y', 'usd_inflation_swap_fwd_5y5y'])
            ]
        )

        display_chart_with_expander(
            "Trajetória da Inflação",
            ["Inflação US", "Inflação Brasil"],
            [
                get_data(category='macro', fields=['us_cpi_yoy', 'us_core_cpi_yoy', 'us_pce_yoy', 'us_supercore_cpi_yoy']),
                get_data(category='macro', fields=['br_ipca_yoy'])
            ]
        )

    elif selected_category == "Brazil":
        display_chart_with_expander(
            "Trajetória do PIB",
            ["PIB US", "PIB Brasil"],
            [
                get_data(category='macro', fields=['us_gdp_yoy']),
                get_data(category='macro', fields=['br_gdp_yoy'])
            ]
        )

        display_chart_with_expander(
            "DM Rates",
            ["Taxa de 1 ano", "Taxa de 1 ano", "Taxa de 5 anos", "Taxa de 5 anos"],
            [
                get_data(category='macro',
                         fields=['germany_generic_1y', 'spain_generic_1y', 'france_generic_1y', 'italy_generic_1y',
                                 'japan_generic_1y',
                                 'switzerland_generic_1y', 'sweden_generic_1y']),
                get_data(category='macro',
                         fields=['germany_generic_5y', 'spain_generic_5y', 'france_generic_5y', 'italy_generic_5y',
                                 'japan_generic_5y',
                                 'switzerland_generic_5y', 'sweden_generic_5y']),
                get_data(category='macro',
                         fields=['new_zealand_generic_1y', 'australia_generic_1y', 'canada_generic_1y', 'norway_generic_1y',
                                 'us_generic_1y', 'uk_generic_1y']),
                get_data(category='macro',
                         fields=['new_zealand_generic_5y', 'australia_generic_5y', 'canada_generic_5y', 'norway_generic_5y',
                                 'us_generic_5y', 'uk_generic_5y']),
            ]
        )

        display_chart_with_expander(
            "EM Rates",
            ["Taxa de 1 ano", "Taxa de 1 ano", "Taxa de 5 anos", "Taxa de 5 anos"],
            [
                get_data(category='macro',
                         fields=['china_generic_1y', 'chile_generic_1y', 'colombia_generic_1y', 'hungary_generic_1y',
                                 'poland_generic_1y', 'peru_generic_1y']),
                get_data(category='macro',
                         fields=['china_generic_5y', 'chile_generic_5y', 'colombia_generic_5y', 'hungary_generic_5y',
                                 'poland_generic_5y', 'peru_generic_5y']),
                get_data(category='macro',
                         fields=['south_africa_generic_1y', 'russia_generic_1y', 'br_generic_1y', 'mexico_generic_1y',
                                 'india_generic_1y', 'indonesia_generic_1y', 'turkey_generic_1y']),
                get_data(category='macro',
                         fields=['south_africa_generic_5y', 'russia_generic_5y', 'br_generic_5y', 'mexico_generic_5y',
                                 'india_generic_5y', 'indonesia_generic_5y', 'turkey_generic_1y']),
            ]
        )

        display_chart_with_expander(
            "Taxa de Juros (BR)",
            ["Curva Pré", "Curva IPCA", "Curva Implícita", "DI Futures"],
            [
                get_data(category='macro', fields=['br_pre_1y', 'br_pre_2y', 'br_pre_3y', 'br_pre_5y', 'br_pre_10y']),
                get_data(category='macro',
                         fields=['br_ipca_1y', 'br_ipca_2y', 'br_ipca_3y', 'br_ipca_5y', 'br_ipca_10y', 'br_ipca_35y']),
                get_data(category='macro',
                         fields=['br_implicita_1y', 'br_implicita_2y', 'br_implicita_3y', 'br_implicita_5y',
                                 'br_implicita_10y']),
                get_yield_curve('br_di_curve')
            ]
        )

    elif selected_category == "Commodities":
        display_chart_with_expander(
            "CRB e Fretes",
            ["Índice CRB", "Índice CRB (% 12 meses)", "DI Futures"],
            [
                get_data(category='commodity', fields=['crb_index', 'crb_fats_oils_index', 'crb_food_index', 'crb_livestock_index', 'crb_metals_index', 'crb_raw_industrials_index', 'crb_textiles_index']),
                get_data(category='commodity', fields=['crb_index']).pct_change(252).dropna(),
                get_data(category='commodity', fields=['baltic_dry_index', 'shanghai_containerized_freight_index'])
            ],
            connect_gaps=True
        )

