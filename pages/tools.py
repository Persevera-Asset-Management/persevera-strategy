import pandas as pd
import numpy as np
from datetime import datetime
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
    unique_dates = df['date_expiration'].unique()
    dates = np.flip(unique_dates)
    return dates


def format_chart(figure, connect_gaps=False):
    figure.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=False),
            type="date",
        ),
        yaxis_title=None, xaxis_title=None,
        yaxis=dict(autorange=True, fixedrange=False, griddash="dash"),
        legend=dict(title=None, yanchor="top", orientation="v"),
        showlegend=True,
        hovermode="x unified",
    )
    figure.update_traces(connectgaps=connect_gaps, hovertemplate="%{y}")
    return figure


def show_tools():
    st.header("Tools")

    st.subheader("Opções de Copom")
    cols = st.columns(2, gap='large')
    with cols[0]:
        meeting_date = st.selectbox(label="Selecione a data da reunião do Copom:",
                                    options=get_copom_meeting_dates(),
                                    index=(np.abs(get_copom_meeting_dates() - datetime.today())).argmin()
                                    )

    df = get_copom_data(meeting_date)
    fig = px.line(df, line_shape='spline')
    st.plotly_chart(format_chart(figure=fig, connect_gaps=True), use_container_width=True)

    st.subheader("Selic Implícita")

    st.subheader("Valuation Pré")

    st.subheader("Valuation Forward")
