from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from train_model import DATA_PATH, MODEL_PATH, train_model


ROOT = Path(__file__).resolve().parent


st.set_page_config(
    page_title="Crop Yield Prediction",
    layout="wide",
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        train_model()
    return joblib.load(MODEL_PATH)


def predict_yield(model_bundle, input_data):
    row = pd.DataFrame([input_data])
    predicted_yield = model_bundle["pipeline"].predict(row)[0]
    total_production = predicted_yield * input_data["Area_ha"]
    return round(float(predicted_yield), 2), round(float(total_production), 2)


data = load_data()
model_bundle = load_model()
metrics = model_bundle["metrics"]

st.title("Crop Yield Prediction")
st.caption("A Streamlit machine learning app using Python, Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn.")

metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
metric_col_1.metric("Dataset Rows", metrics["rows"])
metric_col_2.metric("MAE", metrics["mae"])
metric_col_3.metric("R2 Score", metrics["r2_score"])

st.divider()

form_col, result_col = st.columns([1.2, 1])

with form_col:
    st.subheader("Enter Crop Details")

    state = st.selectbox("State", sorted(data["State"].unique()))
    crop = st.selectbox("Crop", sorted(data["Crop"].unique()))
    season = st.selectbox("Season", sorted(data["Season"].unique()))
    soil_type = st.selectbox("Soil Type", sorted(data["Soil_Type"].unique()))

    input_col_1, input_col_2 = st.columns(2)
    with input_col_1:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=700.0, step=10.0)
        fertilizer = st.number_input("Fertilizer (kg/ha)", min_value=0.0, value=120.0, step=5.0)
        area = st.number_input("Area (ha)", min_value=0.1, value=2.5, step=0.1)
    with input_col_2:
        temperature = st.number_input("Temperature (C)", value=28.0, step=0.5)
        pesticide = st.number_input("Pesticide (kg/ha)", min_value=0.0, value=2.0, step=0.1)

    submitted = st.button("Predict Yield", type="primary", use_container_width=True)

input_data = {
    "State": state,
    "Crop": crop,
    "Season": season,
    "Soil_Type": soil_type,
    "Rainfall_mm": rainfall,
    "Temperature_C": temperature,
    "Fertilizer_kg_per_ha": fertilizer,
    "Pesticide_kg_per_ha": pesticide,
    "Area_ha": area,
}

with result_col:
    st.subheader("Prediction Result")

    if submitted:
        predicted_yield, total_production = predict_yield(model_bundle, input_data)
        st.success("Prediction completed")
        st.metric("Predicted Yield", f"{predicted_yield} ton/ha")
        st.metric("Estimated Production", f"{total_production} tons")
    else:
        st.info("Fill the inputs and click Predict Yield.")

    st.write("Selected Input")
    st.dataframe(pd.DataFrame([input_data]), use_container_width=True, hide_index=True)

st.divider()

chart_col_1, chart_col_2 = st.columns(2)

with chart_col_1:
    st.subheader("Average Yield by Crop")
    crop_yield = data.groupby("Crop", as_index=False)["Yield_ton_per_ha"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=crop_yield, x="Yield_ton_per_ha", y="Crop", ax=ax, color="#2f7d4c")
    ax.set_xlabel("Average Yield (ton/ha)")
    ax.set_ylabel("Crop")
    st.pyplot(fig)

with chart_col_2:
    st.subheader("Rainfall vs Yield")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=data,
        x="Rainfall_mm",
        y="Yield_ton_per_ha",
        hue="Season",
        ax=ax,
    )
    ax.set_xlabel("Rainfall (mm)")
    ax.set_ylabel("Yield (ton/ha)")
    st.pyplot(fig)

with st.expander("View Dataset"):
    st.dataframe(data, use_container_width=True, hide_index=True)
