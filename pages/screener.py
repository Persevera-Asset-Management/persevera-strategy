import pandas as pd
import os
import streamlit as st
from streamlit_option_menu import option_menu

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')


def get_screen(fields: list, selected_sectors: list):
    zoo = pd.read_parquet(os.path.join(DATA_PATH, "factors-factor_zoo.parquet"), columns=fields)
    zoo = zoo.droplevel(1)
    sectors = pd.read_excel(os.path.join(DATA_PATH, "cadastro-base.xlsx"), sheet_name='equities')
    sectors = sectors[['code', 'name', 'sector_layer_1']].set_index('code')

    df = sectors.merge(zoo, left_index=True, right_index=True, how='right')
    if 'Todos' not in selected_sectors:
        df = df.query('sector_layer_1 == @selected_sectors')

    return df


def get_stock_data(code: str, fields: list):
    df = pd.read_parquet("https://persevera.s3.sa-east-1.amazonaws.com/factor_zoo.parquet",
                         filters=[("code", "==", code)], columns=fields)
    df.index = df.index.droplevel(0)
    return df


def create_line_chart(data, title, connect_gaps):
    # ESSA FUNÇÃO É IDENTICA AO DO CHARTBOOK
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


def show_screener():
    st.header("Screener")

    selected_category = option_menu(
        menu_title=None,
        options=["Geral", "Persevera MultiFactor Model (PMM)"],
        orientation="horizontal"
    )

    if selected_category == "Geral":

        # Single Name
        st.subheader("Setorial")
        variables_available = pd.read_parquet(os.path.join(DATA_PATH, "factors-factor_zoo.parquet")).columns
        stocks_available = sorted(pd.read_parquet(os.path.join(DATA_PATH, "factors-factor_zoo.parquet")).index.get_level_values(0))
        sectors_available = sorted(pd.read_excel(os.path.join(DATA_PATH, "cadastro-base.xlsx"), sheet_name='equities')['sector_layer_1'].dropna().unique())
        sectors_available.insert(0, 'Todos')

        cols = st.columns(2, gap='large')

        with cols[0]:
            selected_sectors = st.multiselect(label='Selecione os setores:',
                                              options=sectors_available,
                                              default=['Todos'])
            if 'Todos' in selected_sectors:
                selected_sectors = sectors_available

            liquidity_filter = st.number_input(label="Filtro de Liquidez",
                                               min_value=0, value=0, step=100000)

        with cols[1]:
            selected_variables = cols[1].multiselect(label='Selecione as variáveis:',
                                                     options=variables_available,
                                                     default=['price_close', 'market_cap', '21d_median_dollar_volume_traded'])

        data = get_screen(fields=selected_variables, selected_sectors=selected_sectors)
        data = data[data['21d_median_dollar_volume_traded'] >= liquidity_filter]
        st.dataframe(data)

        # Single Name
        st.subheader("Single Name")

        selected_stock = st.selectbox(label='Selecione a ação:',
                                      options=stocks_available,
                                      default="VALE3")

        cols_single_names = st.columns(2, gap='large')

        with cols_single_names[0]:
            data_price = get_stock_data(code=selected_stock, fields=['price_close'])
            st.plotly_chart(create_line_chart(data_price, f"Preço de Fechamento: {selected_stock}", connect_gaps=True), use_container_width=True)
