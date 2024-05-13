from datetime import datetime
import streamlit as st


def show_quant_screening():
    st.header("Quant Screening")

    with st.form("factor_definition"):
        st.markdown("**Definição dos fatores**")

        cols = st.columns(2, gap='large')
        with cols[0]:
            holding_period = st.number_input("Holding period", value=1, min_value=1, max_value=30)
            freq = st.selectbox("Frequency", options=["D", "M"], index=1)
            start = st.date_input("Date", value=datetime(2008, 1, 1), format="YYYY-MM-DD")
            quantile = st.number_input("Quantile", value=5, min_value=1, max_value=5)

        with cols[1]:
            sector = st.selectbox("Sector Group", index=1, options=["sector_layer_0", "sector_layer_1", "sector_layer_2", "sector_layer_3"])
            container = st.container(border=True)
            with container:
                liquidity_thresh = st.number_input("Liq. Threshold", value=0.4, min_value=0., max_value=1., step=0.1)
                liquidity_lookback = st.selectbox("Liq. Lookback", options=["21", "63", "252"], index=0)
                size_segment = st.selectbox("Size", options=["ALL", "Large", "Mid", "Small"], index=0)


        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write()
