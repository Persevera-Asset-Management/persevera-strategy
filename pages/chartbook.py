import pandas as pd
import numpy as np
import os
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from st_files_connection import FilesConnection

import utils

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
# fs = utils.get_fs_connection("consolidado-indicators.parquet")


def get_data(fs, fields: list):
    df = pd.read_parquet(fs, filters=[('code', 'in', fields)])
    df = df.pivot_table(index='date', columns='code', values='value')
    df = df.filter(fields)
    return df


def get_data_old(fields: list):
    df = pd.read_parquet(os.path.join(DATA_PATH, "consolidado-indicators.parquet"),
                         filters=[('code', 'in', fields)])
    df = df.pivot_table(index='date', columns='code', values='value')
    df = df.filter(fields)
    return df


def code_to_name(df):
    de_para = pd.read_excel(os.path.join(DATA_PATH, "cadastro-base.xlsx"), sheet_name="indicators")
    de_para = de_para.dropna(subset='name')
    de_para = de_para.set_index('code')['name'].to_dict()
    return de_para


def get_cohort(assets: list, benchmark: str):
    individual_assets = get_data(fs, fields=assets)
    individual_assets = individual_assets.filter(assets)
    cohort = individual_assets.iloc[:, 0] / individual_assets.iloc[:, 1]
    cohort.name = ' / '.join(individual_assets.columns)

    df_benchmark = get_data(fs, fields=[benchmark])
    df = cohort.to_frame().merge(df_benchmark, left_index=True, right_index=True, how='left')
    df = df.dropna()
    return df


@st.cache_data
def get_index_data(category: str, codes: list, field: str):
    df = pd.read_parquet(os.path.join(DATA_PATH, f"indicators-index_{category}.parquet"))
    df = df.query('code == @codes')
    df = df.pivot_table(index='date', columns='code', values=field)
    df = df.filter(codes)
    return df


@st.cache_data
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

    df['Mediana (30D)'] = median
    df['Mínima (30D)'] = min
    df['Máxima (30D)'] = max
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
        legend=dict(title=None, yanchor="top", orientation="h", font=dict(size=14)),
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
        yaxis_title=None, xaxis_title=None,
        yaxis=dict(autorange=True, fixedrange=False, griddash="dash"),
        legend=dict(title=None, yanchor="top", orientation="h", font=dict(size=14)),
        showlegend=True,
        barmode="group"
    )
    return fig


def create_area_chart(data, title):
    fig = px.area(data)
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
        legend=dict(title=None, yanchor="top", orientation="h", font=dict(size=14)),
        showlegend=True,
        hovermode="x unified",
    )
    fig.update_traces(hovertemplate="%{y}")
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
        yaxis2=dict(autorange=True, fixedrange=False, showgrid=False),
        legend=dict(title=None, yanchor="top", orientation="h", font=dict(size=14)),
        showlegend=True,
        hovermode="x unified",
    )
    fig.update_traces(connectgaps=connect_gaps, hovertemplate="%{y}")
    return fig


def get_performance_table(df):
    time_frames = {
        'Last': df.iloc[-1],
        'WTD': df.groupby(pd.Grouper(level='date', freq="1W-FRI")).last().pct_change().iloc[-1],
        'MTD': df.groupby(pd.Grouper(level='date', freq="1M")).last().pct_change().iloc[-1],
        '1M': df.groupby(pd.Grouper(level='date', freq="1D")).last().pct_change(1 * 21).iloc[-1],
        'YTD': df.groupby(pd.Grouper(level='date', freq="Y")).last().pct_change().iloc[-1],
        '1Y': df.groupby(pd.Grouper(level='date', freq="1D")).last().pct_change(12 * 21).iloc[-1],
        '2Y': df.groupby(pd.Grouper(level='date', freq="1D")).last().pct_change(2 * 12 * 21).iloc[-1],
    }
    df = pd.DataFrame(time_frames)
    return df


def format_table(df):
    cols_format = list(df.columns)
    cols_format.remove("Last")

    return df.style \
        .bar(align='zero', subset=cols_format, color=['#FCC0CB', '#90EE90']) \
        .format({'Last': '{:,.2f}'.format,
                 'WTD': '{:,.2%}'.format,
                 'MTD': '{:,.2%}'.format,
                 '1M': '{:,.2%}'.format,
                 'YTD': '{:,.2%}'.format,
                 '1Y': '{:,.2%}'.format,
                 '2Y': '{:,.2%}'.format})


def scale_to_100(date, df):
    idx = df.loc[date:].iloc[0].name
    return 100 * (df / df.loc[idx, :])


def show_chartbook():
    st.header("Chartbook")
    fs = utils.get_fs_connection("consolidado-indicators.parquet")

    def display_chart_with_expander(expander_title, chart_titles, chart_types, datasets, connect_gaps=False):
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
                    # Converte código para nome
                    de_para = code_to_name(dataset)
                    if isinstance(dataset, pd.Series):
                        dataset = dataset.to_frame().rename(columns=de_para)
                    else:
                        dataset = dataset.rename(columns=de_para)

                    if chart_type == 'line':
                        col.plotly_chart(create_line_chart(dataset, title, connect_gaps), use_container_width=True)
                    elif chart_type == 'line_two_yaxis':
                        col.plotly_chart(create_two_yaxis_line_chart(dataset, title, connect_gaps),
                                         use_container_width=True)
                    elif chart_type == 'bar':
                        col.plotly_chart(create_bar_chart(dataset, title), use_container_width=True)
                    elif chart_type == 'area':
                        col.plotly_chart(create_area_chart(dataset, title), use_container_width=True)

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

                    # Converte código para nome
                    de_para = code_to_name(dataset)
                    table = table.rename(de_para)

                    col.dataframe(format_table(table), use_container_width=True)
                    # col.write(format_table(table).to_html(escape=True), unsafe_allow_html=True)

    menu_options = {
        "Estados Unidos": "globe",
        "Brasil": "globe-americas",
        "Juros": "clipboard2-pulse",
        "Commodities": "tree",
        "Moedas": "graph-up",
        "Mercados": "pie-chart",
        "Posicionamento": "broadcast-pin",
        "Tendência": "graph-up",
        "Cohorts": "arrow-left-right",
    }
    selected_category = option_menu(
        menu_title=None,
        options=[*menu_options.keys()],
        icons=[*menu_options.values()],
        orientation="horizontal"
    )

    if selected_category == "Estados Unidos":
        display_chart_with_expander(
            "PIB 🅴 🆂",
            ["PIB", "PIB (% YoY)", "PIB (% QoQ)"],
            ["line", "bar", "bar"],
            [
                get_data(fs, fields=["us_gdp_index"]),
                get_data(fs, fields=["us_gdp_yoy"]),
                get_data(fs, fields=["us_gdp_qoq"]),
            ]
        )

        display_chart_with_expander(
            "Taxas Referenciais 🆂",
            ["Curva Pré (Treasuries)", "Curva Inflação (TIPS)", "Curva Implícita (Breakeven)", "Curva de Juros",
             "Inclinações"],
            ["line", "line", "line", "line", "line"],
            [
                get_data(fs, fields=["us_generic_2y", "us_generic_5y", "us_generic_10y", "us_generic_30y"]),
                get_data(fs, fields=["us_generic_inflation_5y", "us_generic_inflation_10y", "us_generic_inflation_20y",
                                 "us_generic_inflation_30y"]),
                get_data(fs, 
                    fields=["us_breakeven_2y", "us_breakeven_5y", "us_breakeven_10y", "usd_inflation_swap_fwd_5y5y"]),
                get_yield_curve(contract="us_fed_funds_curve"),
                get_data(fs, fields=["us_2y10y_steepness", "us_5y10y_steepness", "us_5y30y_steepness"]),
            ],
            connect_gaps=True
        )

        display_chart_with_expander(
            "Taxas Corporativas 🆂",
            ["IG Spreads", "IG Taxas", "HY Spreads", "HY Taxas"],
            ["line", "line", "line", "line"],
            [
                get_data(fs, fields=["us_corporate_ig_5y_spread", "us_corporate_ig_10y_spread"]),
                get_data(fs, fields=["us_corporate_ig_5y_yield", "us_corporate_ig_10y_yield"]),
                get_data(fs, fields=["us_corporate_hy_5y_spread", "us_corporate_hy_10y_spread"]),
                get_data(fs, fields=["us_corporate_hy_5y_yield", "us_corporate_hy_10y_yield"])
            ]
        )

        display_chart_with_expander(
            "Inflação 🅴 🆂",
            ["Índices de Inflação (Consumidor)", "Índices de Inflação (Produtor)",
             "Projeção de Inflação (University of Michigan)", "PCE: Riscos de Inflação (Probabilidades)",
             "CPI Grupos"],
            ["line", "line", "line", "area", "line"],
            [
                get_data(fs, 
                    fields=["us_cpi_yoy", "us_cpi_core_yoy", "us_pce_yoy", "us_pce_core_yoy", "us_supercore_cpi_yoy"]),
                get_data(fs, fields=["us_ppi_yoy"]),
                get_data(fs, fields=["us_university_michigan_expected_inflation_fwd_12m_yoy"]),
                get_data(fs, fields=['us_pce_probability_deflation', 'us_pce_probability_between_0_15',
                                 'us_pce_probability_between_15_25', 'us_pce_probability_above_25']),
                get_data(fs, fields=["us_cpi_apparel_index", "us_cpi_education_and_communication_index", "us_cpi_food_index",
                                 "us_cpi_housing_index", "us_cpi_medical_care_index","us_cpi_other_goods_and_services_index",
                                 "us_cpi_recreation_index", "us_cpi_transportation_index"]).pct_change(12).dropna() * 100,
            ]
        )

        display_chart_with_expander(
            "Produção Industrial 🅴",
            ["Produção Industrial", "Produção Industrial (% YoY)", "Produção Industrial (% MoM)"],
            ["line", "bar", "bar"],
            [
                get_data(fs, fields=["us_industrial_production_index"]),
                get_data(fs, fields=["us_industrial_production_yoy"]),
                get_data(fs, fields=["us_industrial_production_mom"]),
            ]
        )

        display_chart_with_expander(
            "Varejo 🅴",
            ["Advance Retail Sales", "Advance Retail Sales (% LTM)", "Advance Retail Sales (% MoM)",
             "Advance Retail Sales (% YoY)"],
            ["line", "line", "bar", "bar"],
            [
                get_data(fs, fields=["us_advance_retail_sales_total", "us_advance_retail_sales_ex_auto_total"]),
                get_data(fs, fields=["us_advance_retail_sales_total", "us_advance_retail_sales_ex_auto_total"]).rolling(12).sum().pct_change(12).dropna() * 100,
                get_data(fs, fields=["us_advance_retail_sales_total_mom", "us_advance_retail_sales_ex_auto_mom"]),
                get_data(fs, fields=["us_advance_retail_sales_total_yoy", "us_advance_retail_sales_ex_auto_yoy"]),
            ]
        )

        display_chart_with_expander(
            "Habitação 🅴",
            ["Habitação", "Habitação (% YoY)", "Preços de Imóveis", "Preços de Imóveis (% YoY)"],
            ["line", "bar", "line", "bar"],
            [
                get_data(fs, fields=["us_new_home_sales_index", "us_housing_starts_index",
                                 "us_building_permits_index"]),
                get_data(fs, fields=["us_new_home_sales_yoy", "us_housing_starts_yoy",
                                 "us_building_permits_yoy"]),
                get_data(fs, fields=["us_case_shiller_home_price_national", "us_case_shiller_home_price_20_city_index"]),
                get_data(fs, fields=["us_case_shiller_home_price_national_yoy", "us_case_shiller_home_price_20_city_yoy"]),
            ]
        )

        display_chart_with_expander(
            "Crédito 🅴",
            ["Inadimplência"],
            ["line"],
            [
                get_data(fs, fields=["us_delinquency_rates_consumer_loans", "us_delinquency_rates_credit_cards",
                                 "us_delinquency_rates_business_loans"]),
            ]
        )

        display_chart_with_expander(
            "Sentimento 🅴",
            ["Institute for Supply Management (ISM)", "ISM Manufacturing", "ISM Services",
             "Sentimento do Consumidor (University of Michigan)",
             "Índice de Surpresas Econômicas", "Índice de Sentimento de Pequenas Empresas (NFIB)"],
            ["line", "line", "line", "line", "line_two_yaxis", "line"],
            [
                get_data(fs, fields=["us_ism_manufacting", "us_ism_services"]),
                get_data(fs, fields=["us_ism_manufacturing_new_orders", "us_ism_manufacturing_inventories",
                                 "us_ism_manufacturing_prices_paid", "us_ism_manufacturing_employment"]),
                get_data(fs, fields=["us_ism_services_new_orders", "us_ism_services_prices_paid",
                                 "us_ism_services_employment"]),
                get_data(fs, fields=["us_university_michigan_consumer_sentiment_index",
                                 "us_university_michigan_consumer_expectations_index"]),
                get_data(fs, fields=["us_citi_economic_surprise_index", "us_bloomberg_economic_surprise_index"]),
                get_data(fs, fields=["us_nfib_small_business_optimism_index"]),
            ],
            connect_gaps=True,
        )

        display_chart_with_expander(
            "Atividade Econômica 🅴",
            ["Índice de Condições Financeiras", "Índice de Stress Financeiro"],
            ["line", "line"],
            [
                get_data(fs, fields=["us_fed_national_fci", "us_fed_adjusted_national_fci"]),
                get_data(fs, fields=["us_fed_financial_stress_index"]),
            ]
        )

        display_chart_with_expander(
            "Emprego 🅴",
            ["Pedidos de Seguro-Desemprego", "Taxa de Desemprego", "Non-Farm Payroll (MoM)", "Non-Farm Payroll (% YoY)",
             "Ganho Médio por Hora", "Ganho Médio por Hora (% YoY)", "Abertura de Vagas (JOLTS)",
             "Saídas Voluntárias (JOLTS)",
             "Número de vagas abertas por desempregado"],
            ["line", "line", "bar", "bar", "line", "line", "line", "line", "line"],
            [
                get_data(fs, fields=["us_initial_jobless_claims", "us_initial_jobless_claims_4wma",
                                 "us_continuing_jobless_claims"]),
                get_data(fs, fields=["us_unemployment_rate", "us_unemployment_rate_u6"]),
                get_data(fs, fields=["us_adp_nonfarm_employment"]).diff().merge(
                    get_data(fs, fields=["us_employees_nonfarm_payrolls_mom"]), left_index=True, right_index=True,
                    how='outer'),
                get_data(fs, fields=["us_adp_nonfarm_employment_yoy", "us_employees_nonfarm_payrolls_yoy"]),
                get_data(fs, fields=["us_average_hourly_earnings"]),
                get_data(fs, fields=["us_average_hourly_earnings_yoy"]),
                get_data(fs, fields=["us_jolts_hiring_rate", "us_jolts_job_openings_rate"]),
                get_data(fs, fields=["us_jolts_quits_rate"]),
                get_data(fs, fields=['us_unemployed_level_to_job_openings', 'us_job_openings_total_non_farm']).eval(
                    'us_job_openings_total_non_farm / us_unemployed_level_to_job_openings').dropna().rename(
                    'job_openings_to_unemployment_level').to_frame()
            ],
            connect_gaps=True,
        )

    elif selected_category == "Brasil":
        display_chart_with_expander(
            "PIB 🅴 🆂",
            ["PIB", "PIB (% YoY)", "PIB (% QoQ)"],
            ["line", "bar", "bar"],
            [
                get_data(fs, fields=["br_gdp_index"]),
                get_data(fs, fields=["br_gdp_yoy"]),
                get_data(fs, fields=["br_gdp_qoq"]),
            ]
        )

        display_chart_with_expander(
            "Taxas Referenciais 🆂",
            ["Curva Pré", "Curva IPCA", "Curva Implícita", "Curva de Juros"],
            ["line", "line", "line", "line"],
            [
                get_data(fs, fields=["br_pre_1y", "br_pre_2y", "br_pre_3y", "br_pre_5y", "br_pre_10y"]),
                get_data(fs, fields=["br_ipca_1y", "br_ipca_2y", "br_ipca_3y", "br_ipca_5y", "br_ipca_10y", "br_ipca_35y"]),
                get_data(fs, fields=["br_breakeven_1y", "br_breakeven_2y", "br_breakeven_3y", "br_breakeven_5y",
                                 "br_breakeven_10y"]),
                get_yield_curve("br_di_curve")
            ],
            connect_gaps=True
        )

        display_chart_with_expander(
            "Inflação 🅴 🆂",
            ["IPCA (% YoY)", "Expectativa de Inflação (Focus)", "IPCA Grupos (% YoY)", "IPCA Grupos (% YoY)",
             "IPCA Grupos (% YoY)", "Outros Índices (% YoY)"],
            ["line", "line", "line", "line", "line", "line"],
            [
                get_data(fs, fields=["br_ipca_yoy"]).merge(
                    get_data(fs, fields=["br_ipca_yoy", "br_ipca_target_inflation_rate"]).ffill().drop(
                        columns="br_ipca_yoy"), left_index=True, right_index=True, how='left').assign(
                    LimiteSuperior=lambda x: x["br_ipca_target_inflation_rate"] + 1.5,
                    LimiteInferior=lambda x: x["br_ipca_target_inflation_rate"] - 1.5),
                get_data(fs, fields=["br_focus_ipca_median_fwd_12m_yoy", "br_focus_ipca_median_smooth_fwd_12m_yoy", "br_ipca_target_inflation_rate"]).ffill().dropna(
                    subset='br_focus_ipca_median_fwd_12m_yoy').assign(
                    Meta=lambda x: x["br_ipca_target_inflation_rate"].shift(-252)).ffill().assign(
                    LimiteSuperior=lambda x: x["br_ipca_target_inflation_rate"] + 1.5,
                    LimiteInferior=lambda x: x["br_ipca_target_inflation_rate"] - 1.5).drop(
                    columns="br_ipca_target_inflation_rate"),
                get_data(fs, fields=["br_ipca_yoy", "br_ipca_non_regulated_yoy", "br_ipca_regulated_yoy"]),
                get_data(fs, 
                    fields=["br_ipca_yoy", "br_ipca_services_yoy", "br_ipca_durable_yoy", "br_ipca_semi_durable_yoy",
                            "br_ipca_non_durable_yoy"]),
                get_data(fs, fields=["br_ipca_yoy", "br_ipca_food_beverages_yoy", "br_ipca_housing_yoy",
                                 "br_ipca_household_goods_yoy", "br_ipca_clothing_yoy", "br_ipca_transport_yoy",
                                 "br_ipca_health_yoy", "br_ipca_personal_expenses_yoy", "br_ipca_education_yoy",
                                 "br_ipca_communications_yoy"]),
                get_data(fs, fields=["br_ipca_yoy", "br_ipa10_yoy", "br_incc10_yoy", "br_igpm_yoy", "br_cpi_fipe_yoy"]),
            ]
        )

        display_chart_with_expander(
            "Produção Industrial 🅴",
            ["Produção Industrial", "Produção Industrial (% LTM)", "Produção Industrial (% YoY)",
             "Produção Industrial (% MoM)"],
            ["line", "line", "bar", "bar"],
            [
                get_data(fs, fields=["br_industrial_production"]),
                get_data(fs, fields=["br_industrial_production_12m_yoy"]),
                get_data(fs, fields=["br_industrial_production_yoy"]),
                get_data(fs, fields=["br_industrial_production_mom"]),
            ]
        )

        display_chart_with_expander(
            "Contas Públicas 🅴",
            ["Dívida (% do PIB)", "Resultado Fiscal (% do PIB)"],
            ["line", "line"],
            [
                get_data(fs, fields=["br_bcb_gross_gov_debt_to_gdp", "br_bcb_net_gov_debt_to_gdp",
                                 "br_bcb_net_public_sector_debt_to_gdp"]),
                get_data(fs, fields=["br_bcb_primary_result_12m_to_gdp", "br_bcb_nominal_result_12m_to_gdp"]),
            ]
        )

        display_chart_with_expander(
            "Balança Comercial 🅴 🆂",
            ["Termos de Troca (Citi)", "Termos de Troca (MDIC)", "Exportações vs Importações",
             "Exportações vs Importações (YTD)", "Exportações vs Importações (LTM)",
             "Saldo da Balança Comercial (LTM)"],
            ["line_two_yaxis", "line_two_yaxis", "line", "bar", "line", "bar"],
            [
                get_data(fs, fields=["br_citi_terms_of_trade_index", "br_current_account_to_gdp"]),
                get_data(fs, fields=["br_mdic_terms_of_trade_index", "br_current_account_to_gdp"]),
                get_data(fs, fields=["br_trade_balance_fob_exports", "br_trade_balance_fob_imports"]),
                get_data(fs, fields=["br_trade_balance_fob_exports", "br_trade_balance_fob_imports"]).resample("Y").sum(),
                get_data(fs, fields=["br_trade_balance_fob_exports", "br_trade_balance_fob_imports"]).rolling(
                    12).sum().dropna(),
                get_data(fs, fields=["br_trade_balance_fob_t12"]),
            ],
            connect_gaps=True
        )

        display_chart_with_expander(
            "Serviços (PMS) 🅴",
            ["Volume de Serviços", "Volume de Serviços (12 meses)", "Volume de Serviços (% YoY)",
             "Volume de Serviços (% MoM)", "Evolução por Atividade"],
            ["line", "line", "bar", "bar", "line"],
            [
                get_data(fs, fields=["br_pms_services_volume_total_index"]),
                get_data(fs, fields=["br_pms_services_volume_total_index"]),
                get_data(fs, fields=["br_pms_services_volume_total_yoy"]),
                get_data(fs, fields=["br_pms_services_volume_total_mom"]),
                get_data(fs, fields=["br_pms_services_volume_total_index", "br_pms_services_volume_administrative_index",
                                 "br_pms_services_volume_transport_index",
                                 "br_pms_services_volume_individual_and_family_index",
                                 "br_pms_services_volume_information_communication_index",
                                 "br_pms_services_volume_others_index"]),
            ]
        )

        display_chart_with_expander(
            "Varejo (PMC) 🅴",
            ["Volume de Vendas", "Volume de Vendas (% YoY)", "Evolução por Atividade"],
            ["line", "bar", "line"],
            [
                get_data(fs, fields=["br_pmc_retail_sales_volume_total_index",
                                 "br_pmc_retail_sales_volume_total_amplified_index"]),
                get_data(fs, 
                    fields=["br_pmc_retail_sales_volume_total_yoy", "br_pmc_retail_sales_volume_total_amplified_yoy"]),
                get_data(fs, fields=["br_pmc_retail_sales_volume_total_index",
                                 "br_pmc_retail_sales_volume_fuels_lubrificants_index",
                                 "br_pmc_retail_sales_volume_food_beverage_index",
                                 "br_pmc_retail_sales_volume_supermarkets_index",
                                 "br_pmc_retail_sales_volume_textiles_clothing_index",
                                 "br_pmc_retail_sales_volume_furniture_index",
                                 "br_pmc_retail_sales_volume_pharma_cosmetics_index",
                                 "br_pmc_retail_sales_volume_books_magazines_index",
                                 "br_pmc_retail_sales_volume_office_materials_index",
                                 "br_pmc_retail_sales_volume_others_index"]),
            ]
        )

        display_chart_with_expander(
            "Sentimento 🅴",
            ["Índice de Confiança do Consumidor", "Índice de Confiança Empresarial", "Índice de Confiança Industrial",
             "Índice de Incerteza Econômica"],
            ["line", "line", "line", "line"],
            [
                get_data(fs, fields=["br_fgv_consumer_confidence_current_situation_index",
                                 "br_fgv_consumer_confidence_expectations_index", "br_fgv_consumer_confidence_index"]),
                get_data(fs, fields=["br_fgv_business_confidence_current_situation_index",
                                 "br_fgv_business_confidence_expectations_index", "br_fgv_business_confidence_index"]),
                get_data(fs, fields=["br_fgv_industrial_confidence_current_situation_index",
                                 "br_fgv_industrial_confidence_expectations_index",
                                 "br_fgv_industrial_confidence_index"]),
                get_data(fs, fields=["br_fgv_economic_uncertainty_index"]),
            ]
        )

        display_chart_with_expander(
            "Atividade Econômica 🅴",
            ["IBC-Br", "IBC-Br (% YoY)", "IBC-Br (% QoQ)"],
            ["line", "bar", "bar"],
            [
                get_data(fs, fields=["br_ibcbr_index"]),
                get_data(fs, fields=["br_ibcbr_yoy"]),
                get_data(fs, fields=["br_ibcbr_qoq"]),
            ]
        )

        display_chart_with_expander(
            "Emprego 🅴",
            ["Criação de Empregos Formais (MoM)", "Criação de Empregos Formais (LTM)", "Taxa de Desemprego"],
            ["bar", "bar", "line"],
            [
                get_data(fs, fields=["br_caged_registered_employess_total"]).diff(),
                get_data(fs, fields=["br_caged_registered_employess_total"]).diff().rolling(12).sum(),
                get_data(fs, fields=["br_pnad_unemployment_rate"]),
            ]
        )

        display_chart_with_expander(
            "Crédito 🅴",
            ["Saldo da Carteira de Crédito (Total)", "Saldo da Carteira de Crédito (Abertura)",
             "Saldo da Carteira de Crédito (Porte PJ)", "Taxa Média de Juros das Operações",
             "Inadimplência da Carteira de Crédito"],
            ["line", "line", "line", "line", "line"],
            [
                get_data(fs, fields=["br_bcb_credit_outstanding_total", "br_bcb_credit_outstanding_pf",
                                 "br_bcb_credit_outstanding_pj"]),
                get_data(fs, fields=["br_bcb_nonearmarked_credit_outstanding_pj", "br_bcb_earmarked_credit_outstanding_pj",
                                 "br_bcb_nonearmarked_credit_outstanding_pf",
                                 "br_bcb_earmarked_credit_outstanding_pf"]),
                get_data(fs, fields=["br_bcb_credit_outstanding_total", "br_bcb_credit_outstanding_msme",
                                 "br_bcb_credit_outstanding_corporate"]),
                get_data(fs, fields=["br_bcb_average_interest_rate_total", "br_bcb_average_interest_rate_pf",
                                 "br_bcb_average_interest_rate_pj", "br_selic_target"]),
                get_data(fs, fields=["br_bcb_past_due_loans_pf", "br_bcb_past_due_loans_pj"]),
            ],
            connect_gaps=True
        )

        display_chart_with_expander(
            "Tráfego 🅴",
            ["Fluxo Pedagiado nas Estradas", "Fluxo Pedagiado nas Estradas (% YoY)"],
            ["line", "bar"],
            [
                get_data(fs, fields=["br_abcr_traffic_heavy_vehicles", "br_abcr_traffic_light_vehicles"]),
                get_data(fs, fields=["br_abcr_traffic_heavy_vehicles_yoy", "br_abcr_traffic_light_vehicles_yoy"]),
            ]
        )

    elif selected_category == "Juros":
        display_chart_with_expander(
            "Desenvolvidos 🆂",
            ["Taxa de 1 ano", "Taxa de 5 anos", "Taxa de 1 ano", "Taxa de 5 anos"],
            ["line", "line", "line", "line"],
            [
                get_data(fs, fields=["germany_generic_1y", "spain_generic_1y", "france_generic_1y", "italy_generic_1y",
                                 "japan_generic_1y",
                                 "switzerland_generic_1y", "sweden_generic_1y"]),
                get_data(fs, fields=["germany_generic_5y", "spain_generic_5y", "france_generic_5y", "italy_generic_5y",
                                 "japan_generic_5y",
                                 "switzerland_generic_5y", "sweden_generic_5y"]),
                get_data(fs, fields=["new_zealand_generic_1y", "australia_generic_1y", "canada_generic_1y",
                                 "norway_generic_1y",
                                 "us_generic_1y", "uk_generic_1y"]),
                get_data(fs, fields=["new_zealand_generic_5y", "australia_generic_5y", "canada_generic_5y",
                                 "norway_generic_5y",
                                 "us_generic_5y", "uk_generic_5y"]),
            ]
        )

        display_chart_with_expander(
            "Emergentes 🆂",
            ["Taxa de 1 ano", "Taxa de 5 anos", "Taxa de 1 ano", "Taxa de 5 anos"],
            ["line", "line", "line", "line"],
            [
                get_data(fs, fields=["china_generic_1y", "chile_generic_1y", "colombia_generic_1y", "hungary_generic_1y",
                                 "poland_generic_1y", "peru_generic_1y"]),
                get_data(fs, fields=["china_generic_5y", "chile_generic_5y", "colombia_generic_5y", "hungary_generic_5y",
                                 "poland_generic_5y", "peru_generic_5y"]),
                get_data(fs, fields=["south_africa_generic_1y", "russia_generic_1y", "br_generic_1y", "mexico_generic_1y",
                                 "india_generic_1y", "indonesia_generic_1y", "turkey_generic_1y"]),
                get_data(fs, fields=["south_africa_generic_5y", "russia_generic_5y", "br_generic_5y", "mexico_generic_5y",
                                 "india_generic_5y", "indonesia_generic_5y", "turkey_generic_5y"]),
            ]
        )

    elif selected_category == "Commodities":
        display_table_with_expander(
            "Performance 🆂",
            ["Energia", "Metais"],
            [
                get_data(fs, fields=["crude_oil_wti", "crude_oil_brent", "gasoline", "usda_diesel", "natural_gas",
                                 "thermal_coal"]).ffill(limit=2),
                get_data(fs, fields=["gold", "silver", "lme_aluminum", "lme_copper", "lme_nickel_cash", "sgx_iron_ore_62",
                                 "platinum", "palladium", "lme_zinc_spot", "coking_coal"]).fillna(method="ffill",
                                                                                                  limit=2),
            ]
        )

        display_chart_with_expander(
            "Commodity Research Bureau (CRB) 🆂",
            ["Índice CRB (2019 = 100)", "Índice CRB (% 12 meses)"],
            ["line", "line"],
            [
                scale_to_100(date="2019", df=get_data(fs, 
                    fields=["crb_index", "crb_fats_oils_index", "crb_food_index", "crb_livestock_index",
                            "crb_metals_index", "crb_raw_industrials_index", "crb_textiles_index"])),
                get_data(fs, fields=["crb_index"]).pct_change(252).dropna(),
            ],
            connect_gaps=True
        )

        display_chart_with_expander(
            "Fretes 🆂",
            ["Índices de Custo de Frete"],
            ["line"],
            [
                get_data(fs, fields=["baltic_dry_index", "shanghai_containerized_freight_index"])
            ],
            connect_gaps=True
        )

        display_chart_with_expander(
            "Combustível 🅴",
            ["Atacado (2019 = 100)", "Varejo (2019 = 100)"],
            ["line", "line"],
            [
                scale_to_100(date="2019", df=get_data(fs, fields=["crude_oil_brent", "crude_oil_wti", "gasoline",
                                                              "usda_diesel"])),
                scale_to_100(date="2019", df=get_data(fs, fields=["br_anp_gasoline_retail", "br_anp_diesel_retail",
                                                              "br_anp_hydrated_ethanol_retail", "br_anp_lpg_retail"])),
            ],
            connect_gaps=True
        )

        display_chart_with_expander(
            "Brasil 🅴",
            ["Índice de Commodities Brasil (2019 = 100)", "Agrícolas (2019 = 100)", "Pecuárias (2019 = 100)"],
            ["line", "line", "line"],
            [
                scale_to_100(date="2019", df=get_data(fs, 
                    fields=["br_icb_composite", "br_icb_agriculture", "br_icb_energy", "br_icb_metal"])),
                scale_to_100(date="2019", df=get_data(fs, 
                    fields=["br_cepea_paddy_rice", "br_cepea_soft_wheat", "br_cepea_corn_wholesale",
                            "br_cepea_soybean_wholesale", "br_cepea_sugar", "br_cepea_cotton_feather",
                            "br_cepea_arabica_coffee"]).ffill(limit=63)),
                scale_to_100(date="2019", df=get_data(fs, 
                    fields=["br_cepea_chilled_whole_broiler", "br_cepea_pork", "br_cepea_beef_carcass",
                            "br_cepea_beef_forequarter", "br_cepea_beef_hindquarter", "br_cepea_beef_thin_flank",
                            "br_cepea_fed_cattle"])),
            ],
            connect_gaps=True
        )

    elif selected_category == "Moedas":
        display_table_with_expander(
            "Performance 🆂",
            ["Desenvolvidos", "Emergentes"],
            [
                get_data(fs, 
                    fields=["twd_usd", "dxy_index", "eur_usd", "jpy_usd", "gbp_usd", "chf_usd", "cad_usd",
                            "aud_usd", "nok_usd", "sek_usd"]).fillna(method="ffill", limit=2),
                get_data(fs, 
                    fields=["brl_usd", "mxn_usd", "clp_usd", "zar_usd", "try_usd", "cnh_usd"]).fillna(method="ffill",
                                                                                                      limit=2),
            ]
        )

    elif selected_category == "Mercados":
        display_chart_with_expander(
            "EPS 🅼",
            ["S&P 500", "Ibovespa"],
            ["line", "line"],
            [
                get_index_data(category='valuation', codes=["us_sp500"], field="earnings_per_share_fwd"),
                get_index_data(category='valuation', codes=["br_ibovespa"], field="earnings_per_share_fwd"),
            ]
        )

        display_chart_with_expander(
            "P/E 🅼",
            ["Desenvolvidos", "Desenvolvidos (vs. S&P 500)", "Emergentes", "Emergentes (vs. S&P 500)"],
            ["line", "line", "line", "line"],
            [
                get_index_data(
                    category='valuation',
                    codes=["us_sp500", "us_russell2000", "us_nasdaq100", "germany_dax40", "japan_nikkei225", "uk_ukx"],
                    field="price_to_earnings_fwd"),
                get_index_data(
                    category='valuation',
                    codes=["us_sp500", "us_russell2000", "us_nasdaq100", "germany_dax40", "japan_nikkei225", "uk_ukx"],
                    field="price_to_earnings_fwd").apply(lambda x: x / x["us_sp500"], axis=1).drop(columns="us_sp500"),
                get_index_data(
                    category='valuation',
                    codes=["br_ibovespa", "china_csi300", "south_africa_top40", "mexico_bmv", "chile_ipsa",
                           "india_nifty50", "indonesia_jci"],
                    field="price_to_earnings_fwd"),
                get_index_data(
                    category='valuation',
                    codes=["us_sp500", "br_ibovespa", "china_csi300", "south_africa_top40", "mexico_bmv", "chile_ipsa",
                           "india_nifty50", "indonesia_jci"],
                    field="price_to_earnings_fwd").apply(lambda x: x / x["us_sp500"], axis=1).drop(
                    columns="us_sp500").dropna(how="all"),
            ],
            connect_gaps=True
        )

        display_chart_with_expander(
            "Volatilidade Implícita 🅼",
            ["S&P 500", "Ibovespa"],
            ["line_two_yaxis", "line_two_yaxis"],
            [
                get_index_data(category='options', codes=["us_sp500"],
                               field="implied_volatility_100_moneyness_1m").assign(
                    percentile=get_index_data(category='options', codes=["us_sp500"],
                                              field="implied_volatility_100_moneyness_1m").rank(pct=True) * 100),
                get_index_data(category='options', codes=["br_ibovespa"],
                               field="implied_volatility_100_moneyness_1m").assign(
                    percentile=get_index_data(category='options', codes=["br_ibovespa"],
                                              field="implied_volatility_100_moneyness_1m").rank(pct=True) * 100),
            ]
        )

    elif selected_category == "Posicionamento":
        display_chart_with_expander(
            "Treasuries 🅼",
            ["Treasury 2Y", "Treasury 5Y", "Treasury 10Y", "Treasury Bonds"],
            ["bar", "bar", "bar", "bar"],
            [
                get_data(fs, fields=["cftc_cbt_treasury_2y"]),
                get_data(fs, fields=["cftc_cbt_treasury_5y"]),
                get_data(fs, fields=["cftc_cbt_treasury_10y"]),
                get_data(fs, fields=["cftc_cbt_treasury_bonds"]),
            ]
        )

        display_chart_with_expander(
            "Commodities 🅼",
            ["Copper", "Gold", "Silver", "Crude Oil"],
            ["bar", "bar", "bar", "bar"],
            [
                get_data(fs, fields=["cftc_cmx_copper"]),
                get_data(fs, fields=["cftc_cmx_gold"]),
                get_data(fs, fields=["cftc_cmx_silver"]),
                get_data(fs, fields=["cftc_nyme_crude_oil"]),
            ]
        )

        display_chart_with_expander(
            "Moedas 🅼",
            ["AUD", "BRL", "CAD", "CHF", "EUR", "GBP", "JPY", "MXN", "NZD", "RUB", "ZAR"],
            ["bar", "bar", "bar", "bar", "bar", "bar", "bar", "bar", "bar", "bar", "bar"],
            [
                get_data(fs, fields=["cftc_cme_aud"]),
                get_data(fs, fields=["cftc_cme_brl"]),
                get_data(fs, fields=["cftc_cme_cad"]),
                get_data(fs, fields=["cftc_cme_chf"]),
                get_data(fs, fields=["cftc_cme_eur"]),
                get_data(fs, fields=["cftc_cme_gbp"]),
                get_data(fs, fields=["cftc_cme_jpy"]),
                get_data(fs, fields=["cftc_cme_mxn"]),
                get_data(fs, fields=["cftc_cme_nzd"]),
                get_data(fs, fields=["cftc_cme_rub"]),
                get_data(fs, fields=["cftc_cme_zar"]),
            ]
        )

        display_chart_with_expander(
            "Bolsas 🅼",
            ["S&P 500", "Nasdaq", "Nikkei", "Russell 2000"],
            ["bar", "bar", "bar", "bar"],
            [
                get_data(fs, fields=["cftc_cme_sp500"]),
                get_data(fs, fields=["cftc_cme_nasdaq"]),
                get_data(fs, fields=["cftc_cme_nikkei"]),
                get_data(fs, fields=["cftc_cme_russell2000"]),
            ]
        )

    elif selected_category == "Tendência":
        display_chart_with_expander(
            "Média Móvel 🅼",
            ["S&P 500", "Ibovespa"],
            ["line", "line"],
            [
                get_data(fs, fields=["us_sp500"]).assign(ma_200=lambda x: x['us_sp500'].rolling(200).mean(),
                                                     ma_50=lambda x: x['us_sp500'].rolling(50).mean()),
                get_data(fs, fields=["br_ibovespa"]).assign(ma_200=lambda x: x['br_ibovespa'].rolling(200).mean(),
                                                        ma_50=lambda x: x['br_ibovespa'].rolling(50).mean())
            ]
        )

    elif selected_category == "Cohorts":
        display_chart_with_expander(
            "Estados Unidos",
            ["SOXX vs SPY", "Discretionary vs Staples", "VIX3M vs VIX", "High Beta vs Low Volatility",
             "Utilities vs SPY"],
            ["line_two_yaxis", "line_two_yaxis", "line_two_yaxis", "line_two_yaxis", "line_two_yaxis"],
            [
                get_cohort(assets=["us_semiconductor_soxx", "us_sp500"], benchmark="us_sp500"),
                get_cohort(assets=["us_ew_discretionary_rspd", "us_ew_staples_rspd"], benchmark="us_sp500"),
                get_cohort(assets=["us_vix3m", "us_vix"], benchmark="us_sp500"),
                get_cohort(assets=["us_high_beta_sphb", "us_low_volatility_usmv"], benchmark="us_sp500"),
                get_cohort(assets=["us_utilities_xlu", "us_sp500"], benchmark="us_sp500"),
            ]
        )
