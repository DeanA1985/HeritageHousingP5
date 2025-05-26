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

st.markdown("""

This page allows you to explore how different property features
relate to house sale prices, based on historical data and model predictions.

You can:
- **Visualize** historical trends and correlations.
- **Predict** sale prices using trained models.
- **Analyze** which features most influence the price.
- **Export** insights as visuals or documents for sharing or further analysis.
""")

# ---- Load Data ----
df = pd.read_csv("data/processed/cleaned_data.csv")

# ---- SECTION 1: HISTORICAL DATA ----
st.subheader("1. Historical Data Insights")

st.markdown("""
### Historical Relationship Between Features and Sale Price

Use the dropdown below to select any feature (e.g., YearBuilt, GrLivArea,
GarageCars).
You'll see a scatter or box plot showing how that feature relates
to **actual sale prices**.

This is useful for understanding patterns in the original data — for example:
- Do larger houses tend to sell for more?
- Does overall quality increase price?
""")

selected_hist_feature = st.selectbox(
    "Choose a feature from the dataset:", df.columns.drop("SalePrice")
)

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

st.markdown("""
### Correlation Heatmap

This heatmap shows how strongly each feature correlates with Sale Price.
A **stronger correlation** (closer to 1 or -1) means the feature is more
linearly related to sale price.

This does **not** mean causation, but it helps identify key influencers
in the data.
""")

corr_hist = df.select_dtypes(include=["number"]).corr()
fig_corr = px.imshow(
    corr_hist,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu",
    title="Correlation Matrix"
)
st.plotly_chart(fig_corr, use_container_width=True)

# ---- SECTION 2: MODEL-DRIVEN INSIGHTS ----
st.subheader("2. Insights from Model Predictions")

st.markdown("""
## Insights from Model Predictions

This section lets you explore the behavior of your chosen machine learning
model — either Random Forest or Linear Regression.

The model predicts sale prices based on features. By comparing
predictions with actual values,you can evaluate model accuracy
and explore what the model has “learned” about the data.
""")

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

# Feature vs Predicted Price
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

# Predicted vs Actual Price
st.markdown("""
### Predicted vs. Actual Prices

This chart compares the model's **predicted sale prices**
against the real prices in the dataset. A good model should
show points close to the diagonal line — meaning predictions
closely match real values.

If there’s a wide spread, it might suggest:
- Model limitations
- Data issues
- Underfitting or overfitting
""")

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
st.markdown("""
### Feature Importance

Feature importance tells us which variables the model
relied on most when making predictions.

- In a **Random Forest**, it's based on how often and how effectively
a feature splits the data.
- In **Linear Regression**, it’s based on the absolute value of the
model coefficients.

This helps you understand what drives home value in the model's eyes.
""")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": abs(
        model.feature_importances_
        if hasattr(model, "feature_importances_")
        else model.coef_
    )
}).sort_values(by="Importance", ascending=False)

# Top 3 Influential Features
st.markdown("""
### Top 3 Influential Features

These are the features with the highest influence on the model’s predictions.
They are ranked by how much they contributed to the predicted sale prices.
""")

top_features = importance_df.head(3)
for _, row in top_features.iterrows():
    feature = row["Feature"]
    importance = row["Importance"]
    st.success(
        f"**{feature}** is a top driver with importance "
        f"{importance:.4f}"
    )

# ---- EXPORT SECTION ----
st.subheader("Export Insights and Visuals")

st.markdown("""
### Exporting Your Insights

You can download:
- A **CSV file** containing the top features and their importance.
- A **PNG chart** visualizing the ranking of features.
- A **PDF summary** with a plain-text summary of the top 3 features
and their scores.

These exports are perfect for reports, presentations,
or client-facing documentation.
""")

# CSV Export
st.download_button(
    label="Download CSV",
    data=top_features.to_csv(index=False),
    file_name="top_feature_insights.csv",
    mime="text/csv"
)

# PNG Export
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
    st.markdown("""
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
""")

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
