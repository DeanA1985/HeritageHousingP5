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

# Footer with GitHub link
footer = """
<style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #f0f2f6;
        text-align: center;
        padding: 10px;
        font-size: 0.9em;
        color: #555;
        border-top: 1px solid #eaeaea;
    }
    .footer a {
        color: #0366d6;
        text-decoration: none;
        margin-left: 10px;
    }
</style>
<div class="footer">
    Heritage Housing App &copy; 2025 | Powered by Streamlit
    <a href="https://github.com/DeanA1985/HeritageHousingP5" target="_blank">
    GitHub Repo</a>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
