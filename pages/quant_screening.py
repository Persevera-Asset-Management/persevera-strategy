from datetime import datetime
import streamlit as st


def show_quant_screening():
    st.header("Quant Screening")

    with st.form("factor_definition"):
        st.markdown("**Definição dos fatores**")

        cols = st.columns(2, gap='large')
        with cols[0]:
            freq = st.selectbox("Frequency of rebalance", options=["D", "M"], index=1)
            holding_period = st.number_input(f"Holding period (in {freq})", value=1, min_value=1, max_value=30)
            start = st.date_input("Start date", value=datetime(2008, 1, 1), format="YYYY-MM-DD")
            quantile = st.number_input("Quantile", value=5, min_value=1, max_value=5)
            sector = st.selectbox("Sector Group", index=1, options=["sector_layer_0", "sector_layer_1", "sector_layer_2", "sector_layer_3"])

        with cols[1]:
            container = st.container(border=True)
            with container:
                liquidity_thresh = st.number_input("Liq. Threshold", value=0.4, min_value=0., max_value=1., step=0.1)
                liquidity_lookback = st.selectbox("Liq. Lookback", options=["21", "63", "252"], index=0)
                size_segment = st.selectbox("Size", options=["ALL", "Large", "Mid", "Small"], index=0)

            use_buckets = st.checkbox("Buckets", value=True)
            use_factor_relevance = st.checkbox("Factor Relevance", value=True)
            use_sector_score = st.checkbox("Sector Score", value=True)

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write()
