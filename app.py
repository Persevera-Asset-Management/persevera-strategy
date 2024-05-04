import os, logging
import streamlit as st
from streamlit_navigation_bar import st_navbar
import warnings
import pages as pg

warnings.simplefilter("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

st.set_page_config(
    page_title="Persevera",
    layout="wide",
    initial_sidebar_state="collapsed"
)

pages = ["Reunião de Estratégia", "Trinity", "Nemesis", "Risk Management", "Quant Screening", "Factor Tearsheet", "Chartbook"]
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(parent_dir, "logo.svg")

styles = {
    "nav": {
        "background-color": "lightgrey",
        "justify-content": "left",
    },
    "div": {
        "max-width": "50rem",
    },
    "img": {
        "padding-right": "14px",
    },
    "span": {
        "color": "black",
        "padding": "14px",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
    "active": {
        "color": "var(--text-color)",
        "background-color": "white",
        "font-weight": "normal",
        "padding": "14px",
    }
}
options = {
    "show_menu": False,
    "show_sidebar": False,
}

page = st_navbar(
    pages=pages,
    logo_path=logo_path,
    styles=styles,
    options=options,
)

functions = {
    "Home": pg.show_home,
    "Reunião de Estratégia": pg.show_reuniao_estrategia,
    "Trinity": pg.show_trinity,
    "Nemesis": pg.show_nemesis,
    "Risk Management": pg.show_risk_management,
    "Quant Screening": pg.show_quant_screening,
    "Factor Tearsheet": pg.show_factor_tearsheet,
    "Chartbook": pg.show_others,
}
go_to = functions.get(page)
if go_to:
    go_to()
