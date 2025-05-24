import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load model
model = joblib.load("models/random_forest_model.pkl")

# Page title
st.header("🏠 Estimate House Sale Price")

# Instructions
st.markdown("Use the sliders to input property characteristics and get an estimated price.")

# Input sliders in columns
col1, col2 = st.columns(2)

with col1:
    GrLivArea = st.slider("Above Ground Living Area (sq ft)", 500, 4000, 1500)
    GarageArea = st.slider("Garage Area (sq ft)", 0, 1000, 300)
    TotalBsmtSF = st.slider("Total Basement Area (sq ft)", 0, 2000, 800)
    FirstFlrSF = st.slider("1st Floor Area (sq ft)", 0, 2000, 1000)
    GarageYrBlt = st.slider("Garage Year Built", 1900, 2023, 2005)

with col2:
    OverallQual = st.slider("Overall Quality Rating", 1, 10, 5, help="1 = very poor, 10 = excellent")
    YearBuilt = st.slider("Year Built", 1900, 2023, 1980)
    YearRemodAdd = st.slider("Year Remodeled", 1950, 2023, 2000)
    MasVnrArea = st.slider("Masonry Veneer Area (sq ft)", 0, 1000, 200)
    BsmtFinSF1 = st.slider("Finished Basement SF Type 1", 0, 2000, 500)

# Create DataFrame
input_data = pd.DataFrame({
    "GrLivArea": [GrLivArea],
    "GarageArea": [GarageArea],
    "TotalBsmtSF": [TotalBsmtSF],
    "1stFlrSF": [FirstFlrSF],
    "GarageYrBlt": [GarageYrBlt],
    "OverallQual": [OverallQual],
    "YearBuilt": [YearBuilt],
    "YearRemodAdd": [YearRemodAdd],
    "MasVnrArea": [MasVnrArea],
    "BsmtFinSF1": [BsmtFinSF1]
})

# Predict
if st.button("Predict Sale Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Sale Price: **${prediction:,.2f}**")

    # Save downloadable report
    report = input_data.copy()
    report["PredictedSalePrice"] = round(prediction, 2)
    filename = f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    st.download_button("Download Prediction Report", report.to_csv(index=False), file_name=filename)