import pandas as pd
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')


def get_data(category: str, fields: list):
    df = pd.read_parquet(os.path.join(DATA_PATH, f"indicators-{category}.parquet"))
    df = df.query('code == @fields')
    df = df.pivot_table(index='date', columns='code', values='value')
    return df


def create_line_chart(data, title, connectgaps=False):
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
    )
    fig.update_traces(connectgaps=connectgaps)
    return fig


def show_others():
    st.header("Reunião de Estratégia")
    st.write(
        """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Mauris id diam 
        pharetra, dapibus est fermentum, laoreet diam. Integer vitae consequat augue:
        """
    )

    def display_chart_with_expander(expander_title, chart_titles, datasets):
        with st.expander(expander_title):
            cols = st.columns(len(chart_titles))
            for col, title, dataset in zip(cols, chart_titles, datasets):
                col.plotly_chart(create_line_chart(dataset, title), use_container_width=True)

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

    # Continue similar pattern for other sections...
