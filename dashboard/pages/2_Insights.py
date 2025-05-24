import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Page title
st.header("📊 Market Insights")

st.markdown("Explore how sale prices relate to home quality "
            "features in the dataset."
            )

# Load data
df = pd.read_csv("data/processed/cleaned_data.csv")

# Visualization
st.subheader("Average Sale Price by Overall Quality")

try:
    avg_price = df.groupby("OverallQual")["SalePrice"].mean().reset_index()
    fig = go.Figure(go.Bar(
        x=avg_price["OverallQual"],
        y=avg_price["SalePrice"],
        marker_color="indianred"
    ))
    fig.update_layout(
        title="Average Sale Price vs. Quality Rating",
        xaxis_title="Overall Quality (1–10)",
        yaxis_title="Average Sale Price ($)",
        plot_bgcolor="#f5f5f5"
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not display chart. Error: {e}")
