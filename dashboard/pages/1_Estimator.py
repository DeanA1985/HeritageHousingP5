import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# --- Page Title ---
st.header("🏡 Estimate House Sale Price")

# --- Intro ---
st.markdown(
    """
Use the sliders to enter property characteristics, and choose a model to
predict the estimated house sale price.

**Available Models:**
- **Random Forest:** Better for capturing complex patterns in the data.
- **Linear Regression:** Simpler and more interpretable.
"""
)

# --- Model Choice (Toggle) ---
model_choice = st.toggle(
    "Use Linear Regression (turn off for Random Forest)",
    value=False
)
model_path = (
    "models/linear_regression_model.pkl"
    if model_choice else
    "models/random_forest_model.pkl"
)
model_label = (
    "Linear Regression"
    if model_choice else
    "Random Forest"
)

model = joblib.load(model_path)
# Load the corresponding feature list
feature_path = (
    "models/linear_regression_features.pkl"
    if model_choice else
    "models/random_forest_features.pkl"
)
expected_features = joblib.load(feature_path)
st.info(f"**Current Model:** {model_label}")

# --- Input Sliders ---
st.markdown("### Enter Property Characteristics")
col1, col2 = st.columns(2)

with col1:
    GrLivArea = st.slider(
        "Above Ground Living Area (sq ft)", 500, 4000, 1500,
        help="Finished living area above ground level."
    )
    GarageArea = st.slider(
        "Garage Area (sq ft)", 0, 1000, 300,
        help="Enclosed garage space."
    )
    TotalBsmtSF = st.slider(
        "Total Basement Area (sq ft)", 0, 2000, 800,
        help="Total basement square footage."
    )
    FirstFlrSF = st.slider(
        "First Floor Area (sq ft)", 0, 2000, 1000,
        help="First floor square footage."
    )
    GarageYrBlt = st.slider(
        "Garage Year Built", 1900, 2023, 2005,
        help="Year the garage was built."
    )

with col2:
    OverallQual = st.slider(
        "Overall Quality Rating", 1, 10, 5,
        help="1 = very poor, 10 = excellent quality of materials and finish."
    )
    YearBuilt = st.slider(
        "Year Built", 1900, 2023, 1980,
        help="Year the house was originally constructed."
    )
    YearRemodAdd = st.slider(
        "Year Remodeled", 1950, 2023, 2000,
        help="Most recent year the house was remodeled or added to."
    )
    MasVnrArea = st.slider(
        "Masonry Veneer Area (sq ft)", 0, 1000, 200,
        help="Masonry veneer (brick/stone) on exterior walls."
    )
    BsmtFinSF1 = st.slider(
        "Finished Basement SF Type 1", 0, 2000, 500,
        help="Finished basement area of type 1."
    )

# --- Create Aligned DataFrame ---
input_data = pd.DataFrame({
    "GrLivArea": GrLivArea,
    "GarageArea": GarageArea,
    "TotalBsmtSF": TotalBsmtSF,
    "1stFlrSF": FirstFlrSF,
    "GarageYrBlt": GarageYrBlt,
    "OverallQual": OverallQual,
    "YearBuilt": YearBuilt,
    "YearRemodAdd": YearRemodAdd,
    "MasVnrArea": MasVnrArea,
    "BsmtFinSF1": BsmtFinSF1
}, index=[0])


# Reindex input data to match expected feature order
input_data = input_data.reindex(columns=expected_features, fill_value=0)


# --- Prediction ---
if st.button("📊 Predict Sale Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"**Estimated Sale Price: ${prediction:,.0f}**")

    # Save downloadable report
    report = input_data.copy()
    report["PredictedSalePrice"] = round(prediction, 2)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.download_button(
        label="⬇️ Download Prediction Report",
        data=report.to_csv(index=False),
        file_name=f"prediction_report_{timestamp}.csv",
        mime="text/csv"
    )
