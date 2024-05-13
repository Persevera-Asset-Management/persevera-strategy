from datetime import datetime
import streamlit as st


def show_quant_screening():
    st.header("Quant Screening")

    with st.form("factor_definition"):
        st.markdown("**Definição dos fatores**")
        holding_period = st.number_input("Holding period", value=1, min_value=1, max_value=30)
        freq = st.selectbox("Frequency", options=["D", "M"])
        start = st.date_input("Date", value=datetime(2008, 1, 1), format="YYYY-MM-DD")
        quantile = st.number_input("Quantile", value=5, min_value=1, max_value=5)

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write()
