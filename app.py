import os
import logging
import warnings
import streamlit as st
from streamlit_navigation_bar import st_navbar
import pages as pg

# Ignore future warnings
warnings.simplefilter("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Define the pages
PAGES = ["Chartbook", "Fixed Income", "Fund Analysis", "Tools", "Factor Playground", "Screener"]

# Define paths
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(parent_dir, "logo.svg")
page_icon_logo_path = os.path.join(parent_dir, "assets/persevera_logo_page_icon.png")

# Streamlit configuration
st.set_page_config(
    page_title="Persevera",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Define navigation styles
NAV_STYLES = {
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

NAV_OPTIONS = {
    "show_menu": True,
    "show_sidebar": False,
}

# Display the navigation bar
selected_page = st_navbar(
    pages=PAGES,
    logo_path=logo_path,
    styles=NAV_STYLES,
    options=NAV_OPTIONS,
)

# Page functions mapping
PAGE_FUNCTIONS = {
    "Home": pg.show_home,
    "Chartbook": pg.show_chartbook,
    "Fixed Income": pg.show_fixed_income,
    "Fund Analysis": pg.show_fund_analysis,
    "Factor Playground": pg.show_factor_playground,
    "Screener": pg.show_screener,
    "Tools": pg.show_tools,
}


# Navigate to the selected page
def navigate_to_page(page_name):
    """Navigate to the selected page."""
    page_function = PAGE_FUNCTIONS.get(page_name)
    if page_function:
        page_function()
    else:
        st.error("Page not found")


navigate_to_page(selected_page)
