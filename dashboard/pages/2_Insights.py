import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import joblib
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# ---- Page Header ----
st.header("Explore Insights from Heritage Housing Data")

# ---- Load Data ----
df = pd.read_csv("data/processed/cleaned_data.csv")

# ---- SECTION 1: HISTORICAL DATA ----
st.subheader("1. Historical Data Insights")

st.markdown(
    """
    Explore the raw data used to train the models.
    Use the controls below to choose a feature and
    see how it relates to sale price.
    """
)

selected_hist_feature = st.selectbox(
    "Choose a feature from the dataset:", df.columns.drop("SalePrice")
)

# Plot depending on feature type
if pd.api.types.is_numeric_dtype(df[selected_hist_feature]):
    fig_hist = px.scatter(
        df,
        x=selected_hist_feature,
        y="SalePrice",
        trendline="ols",
        opacity=0.6,
        title=f"{selected_hist_feature} vs. Sale Price (Historical)"
    )
else:
    fig_hist = px.box(
        df,
        x=selected_hist_feature,
        y="SalePrice",
        title=f"{selected_hist_feature} vs. Sale Price (Historical)"
    )
st.plotly_chart(fig_hist, use_container_width=True)

# ---- SECTION 2: MODEL-DRIVEN INSIGHTS ----
st.subheader("2. Insights from Model Predictions")

model_choice = st.sidebar.radio(
    "Choose a model:", ["Random Forest", "Linear Regression"]
)

if model_choice == "Random Forest":
    model = joblib.load("models/random_forest_model.pkl")
    features = joblib.load("models/random_forest_features.pkl")
else:
    model = joblib.load("models/linear_regression_model.pkl")
    features = joblib.load("models/linear_regression_features.pkl")

X = df.reindex(columns=features, fill_value=0)
y = df["SalePrice"]
predicted_prices = model.predict(X)

selected_feature = st.selectbox(
    "Choose a model input feature to explore:", features
)

# Plot: Feature vs Predicted Price
st.markdown("### Feature vs. Predicted Sale Price")
fig1 = px.scatter(
    x=df[selected_feature],
    y=predicted_prices,
    labels={"x": selected_feature, "y": "Predicted Price"},
    trendline="ols",
    opacity=0.6,
    title=f"{selected_feature} vs. Predicted Sale Price"
)
st.plotly_chart(fig1, use_container_width=True)

# Plot: Actual vs. Predicted
st.markdown("### Actual vs. Predicted Sale Price")
fig2 = px.scatter(
    x=y,
    y=predicted_prices,
    labels={"x": "Actual Sale Price", "y": "Predicted Sale Price"},
    trendline="ols",
    opacity=0.6,
    title="Actual vs. Predicted Sale Price"
)
st.plotly_chart(fig2, use_container_width=True)

# Feature Importance
st.markdown("### Key Influential Features")
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": abs(
        model.feature_importances_
        if hasattr(model, "feature_importances_")
        else model.coef_)

}).sort_values(by="Importance", ascending=False)

top_features = importance_df.head(3)
for _, row in top_features.iterrows():
    feature = row["Feature"]
    importance = row["Importance"]
    st.success(
        f"**{feature}** is a top driver with importance {importance:.4f}"
    )

# ---- EXPORT SECTION ----
st.subheader("Export Insights and Visuals")

# CSV Export
st.markdown("### Download Top Features as CSV")
st.download_button(
    label="Download CSV",
    data=top_features.to_csv(index=False),
    file_name="top_feature_insights.csv",
    mime="text/csv"
)

# PNG Export
st.markdown("### Download Feature Importance Chart as PNG")
png_buffer = io.BytesIO()
fig_png = px.bar(
    top_features,
    x="Feature",
    y="Importance",
    title="Top Feature Importance"
)
pio.write_image(fig_png, png_buffer, format="png")
st.download_button(
    label="Download PNG",
    data=png_buffer.getvalue(),
    file_name="feature_importance.png",
    mime="image/png"
)

# PDF Export
st.markdown("### Download Insight Summary as PDF")
pdf_buffer = io.BytesIO()
c = canvas.Canvas(pdf_buffer, pagesize=letter)
c.setFont("Helvetica", 12)
c.drawString(50, 800, "Top 3 Influential Features:")
for i, row in top_features.iterrows():
    feature_text = f"{row['Feature']} - {round(row['Importance'], 4)}"
    c.drawString(50, 780 - (i * 20), feature_text)
c.save()
pdf_buffer.seek(0)
st.download_button(
    label="Download PDF",
    data=pdf_buffer,
    file_name="insight_summary.pdf",
    mime="application/pdf"
)

# ---- FEATURE GLOSSARY ----
with st.expander("See Feature Glossary (Full Descriptions)"):
    st.markdown(
        """
        - `GrLivArea`: Above Ground Living Area (sq ft)
        - `GarageArea`: Enclosed Garage Space (sq ft)
        - `TotalBsmtSF`: Total Basement Area (sq ft)
        - `1stFlrSF`: First Floor Area (sq ft)
        - `GarageYrBlt`: Year Garage was built
        - `OverallQual`: Overall Quality Rating (1–10)
        - `YearBuilt`: Year the house was originally constructed
        - `YearRemodAdd`: Year of remodel/addition
        - `MasVnrArea`: Masonry Veneer Area (sq ft)
        - `BsmtFinSF1`: Finished Basement Area Type 1 (sq ft)
        - `BsmtExposure`: Basement exposure level (None, Gd, Av, Mn)
        - `KitchenQual`: Kitchen Quality (Ex, Gd, TA, Fa)
        - `SalePrice`: Sale price of the property
        """
    )
