import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Heritage Housing Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Home page welcome
st.markdown(
    """
    <h2>Heritage Housing Sale Price Estimator</h2>
    <p>
    Navigate using the sidebar to estimate house prices,
    explore insights, or learn about this project.
    </p>
    """,
    unsafe_allow_html=True)
