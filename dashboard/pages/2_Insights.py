import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import joblib
import io
from reportlab.pdfgen import canvas

# --- Page Header ---
st.header("Explore Insights from Heritage Housing Data")

# --- Load Data ---
df = pd.read_csv("data/processed/cleaned_data.csv")

# ===========================
# SECTION 1: HISTORICAL DATA
# ===========================
st.subheader("1. Historical Data Insights")

st.markdown(
    """
    Explore the raw data used to train the models.
    Use the controls below to choose a feature and
    see how it relates to sale price.
    """
)

selected_hist_feature = st.selectbox(
    "Choose a feature from the dataset:",
    df.columns.drop("SalePrice")
)

# Plot: Feature vs SalePrice (Historical)
fig_hist = px.scatter(
    df, x=selected_hist_feature, y="SalePrice",
    trendline="ols", opacity=0.6,
    title=f"{selected_hist_feature} vs. Sale Price (Historical)"
)
st.plotly_chart(fig_hist, use_container_width=True)

# Heatmap of Correlations
st.markdown("### Correlation Heatmap")
corr_hist = df.select_dtypes(include='number').corr()
fig_corr = px.imshow(corr_hist, text_auto=True, aspect="auto")
st.plotly_chart(fig_corr, use_container_width=True)

# Insight box
corr_strength = abs(
    corr_hist[selected_hist_feature]["SalePrice"]
)

insight_text = (
    f"**{selected_hist_feature}** shows a "
    f"{'strong' if corr_strength > 0.5 else 'moderate'} "
    "correlation with sale price."
)

# ===============================
# SECTION 2: MODEL-DRIVEN INSIGHTS
# ===============================
st.subheader("2. Insights from Model Predictions")

# --- Model Selection Toggle ---
st.sidebar.header("Model Settings")
model_choice = st.sidebar.radio(
    "Choose a model:",
    ["Random Forest", "Linear Regression"]
)

# Load model and features
if model_choice == "Random Forest":
    model = joblib.load("models/random_forest_model.pkl")
    features = joblib.load("models/random_forest_features.pkl")
else:
    model = joblib.load("models/linear_regression_model.pkl")
    features = joblib.load("models/linear_regression_features.pkl")

# Prepare data for prediction
X = df.reindex(columns=features, fill_value=0)
df["PredictedPrice"] = model.predict(X)

# Feature selection
selected_feature = st.selectbox(
    "Choose a model input feature to explore:",
    features
)


# Plot: Feature vs Predicted Price
st.markdown("### Feature vs. Predicted Sale Price")
fig1 = px.scatter(
    df, x=selected_feature, y="PredictedPrice",
    trendline="ols", opacity=0.6,
    title=f"{selected_feature} vs. Predicted Price"
)
st.plotly_chart(fig1, use_container_width=True)

# Plot: Actual vs. Predicted Price
st.markdown("### Actual vs. Predicted Sale Price")
fig2 = px.scatter(
    df, x="SalePrice", y="PredictedPrice",
    trendline="ols", opacity=0.6,
    labels={"SalePrice": "Actual", "PredictedPrice": "Predicted"},
    title="Actual vs. Predicted Sale Price"
)
st.plotly_chart(fig2, use_container_width=True)

# Feature Importance (Random Forest only)
if model_choice == "Random Forest":
    st.markdown("### Random Forest Feature Importance")
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig3 = px.bar(importance_df, x="Importance", y="Feature", orientation='h',
                  title="Feature Importance Ranking")
    st.plotly_chart(fig3, use_container_width=True)
else:
    coefs = model.coef_
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": abs(coefs)
    }).sort_values(by="Importance", ascending=False)

# Insight cards
st.markdown("### Key Influential Features")
top_features = importance_df.head(3)
for i, row in top_features.iterrows():
    feature = row['Feature']
    importance = row['Importance']
    st.success(
        f"**{feature}** is a top driver with importance {importance:.4f}"
    )


# =====================
# EXPORT SECTION
# =====================
with st.expander("⬇️ Export Insights and Visuals"):
    # CSV Export
    st.markdown("#### Download Top Features as CSV")
    insight_export_df = pd.DataFrame({
        "Feature": top_features["Feature"],
        "Importance": top_features["Importance"]
    })
    st.download_button(
        label="Download CSV",
        data=insight_export_df.to_csv(index=False),
        file_name="top_feature_insights.csv",
        mime="text/csv"
    )

    # PNG Export (if RF)
    if model_choice == "Random Forest":
        st.markdown("#### Download Feature Importance Chart as PNG")
        png_buffer = io.BytesIO()
        pio.write_image(fig3, png_buffer, format="png")
        st.download_button(
            label="Download PNG",
            data=png_buffer.getvalue(),
            file_name="feature_importance.png",
            mime="image/png"
        )

    # PDF Export
    st.markdown("#### Download Insight Summary as PDF")
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer)
    c.setFont("Helvetica", 12)
    c.drawString(50, 800, "Top 3 Influential Features")
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
