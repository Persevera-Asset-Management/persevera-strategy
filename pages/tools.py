import pandas as pd
import logging, os
import plotly.express as px
import streamlit as st


DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))


def get_copom_data(meeting):
    df = pd.read_parquet(os.path.join(DATA_PATH, "b3-copom_options.parquet"),
                         columns=['date', 'date_expiration', 'price', 'decision'],
                         filters=[('date_expiration', '==', meeting)])

    df = df.pivot_table(index='date', columns='decision', values='price')
    return df


def get_copom_meeting_dates():
    df = pd.read_parquet(os.path.join(DATA_PATH, "b3-copom_options.parquet"), columns=['date_expiration'])
    df = df['date_expiration'].unique()
    return df


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


def show_tools():
    st.header("Tools")

    st.subheader("Opções de Copom")
    meeting_date = st.selectbox(label="Selecione a data da reunião do Copom:",
                                options=get_copom_meeting_dates())

    df = get_copom_data(meeting_date)
    fig = px.line(df)
    st.plotly_chart(format_chart(figure=fig, connect_gaps=True), use_container_width=True)

    st.subheader("Selic Implícita")

    st.subheader("Valuation Pré")

    st.subheader("Valuation Forward")
