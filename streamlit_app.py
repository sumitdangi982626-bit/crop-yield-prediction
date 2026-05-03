from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from train_model import DATA_PATH, MODEL_PATH, train_model


ROOT = Path(__file__).resolve().parent


st.set_page_config(
    page_title="Crop Yield Studio",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --leaf: #2f7d4c;
        --leaf-dark: #1f5d39;
        --mint: #e9f8ee;
        --sun: #f7c948;
        --peach: #ffebe0;
        --sky: #e7f3ff;
        --soil: #7a5632;
        --ink: #17251d;
        --muted: #627267;
        --line: #d8eadf;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(247, 201, 72, 0.30), transparent 26%),
            radial-gradient(circle at 88% 10%, rgba(102, 187, 106, 0.24), transparent 28%),
            linear-gradient(135deg, #fbfff8 0%, #eef9f1 46%, #fff8ec 100%);
        color: var(--ink);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(47, 125, 76, 0.16);
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }

    p, label, span {
        color: var(--ink);
    }

    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stMarkdownContainer"] p {
        color: var(--ink);
    }

    .hero-wrap {
        border: 1px solid rgba(47, 125, 76, 0.18);
        border-radius: 22px;
        padding: 30px;
        background:
            linear-gradient(120deg, rgba(255, 255, 255, 0.94), rgba(233, 248, 238, 0.92)),
            repeating-linear-gradient(90deg, transparent 0 22px, rgba(47, 125, 76, 0.035) 22px 23px);
        box-shadow: 0 18px 50px rgba(31, 93, 57, 0.12);
        position: relative;
        overflow: hidden;
    }

    .hero-wrap:after {
        content: "🌿";
        position: absolute;
        right: 30px;
        top: 20px;
        font-size: 70px;
        opacity: 0.18;
    }

    .kicker {
        display: inline-flex;
        gap: 8px;
        align-items: center;
        border: 1px solid rgba(47, 125, 76, 0.20);
        border-radius: 999px;
        padding: 7px 13px;
        background: #ffffff;
        color: var(--leaf-dark);
        font-size: 13px;
        font-weight: 800;
    }

    .hero-title {
        margin: 14px 0 8px;
        font-size: 48px;
        line-height: 1.04;
        font-weight: 900;
    }

    .hero-subtitle {
        max-width: 760px;
        margin: 0;
        color: var(--muted);
        font-size: 17px;
        line-height: 1.55;
    }

    .plant-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 20px;
    }

    .plant-chip {
        border-radius: 999px;
        padding: 9px 13px;
        background: #ffffff;
        border: 1px solid rgba(47, 125, 76, 0.18);
        color: var(--leaf-dark);
        font-weight: 750;
        font-size: 13px;
    }

    .section-card {
        border: 1px solid rgba(47, 125, 76, 0.16);
        border-radius: 18px;
        padding: 22px;
        background: rgba(255, 255, 255, 0.88);
        box-shadow: 0 12px 34px rgba(31, 93, 57, 0.10);
    }

    .result-card {
        border-radius: 22px;
        padding: 24px;
        background: linear-gradient(145deg, #ffffff, #ecf8ef);
        border: 1px solid rgba(47, 125, 76, 0.18);
        box-shadow: 0 16px 40px rgba(31, 93, 57, 0.12);
    }

    .result-number {
        margin: 8px 0 4px;
        color: var(--leaf-dark);
        font-size: 46px;
        line-height: 1;
        font-weight: 950;
    }

    .result-copy {
        color: var(--muted);
        font-size: 15px;
        line-height: 1.5;
    }

    .tiny-label {
        color: var(--leaf-dark);
        font-size: 12px;
        font-weight: 850;
        text-transform: uppercase;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(47, 125, 76, 0.16);
        border-radius: 16px;
        padding: 16px 18px;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 10px 26px rgba(31, 93, 57, 0.08);
    }

    div[data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-weight: 750;
    }

    div[data-testid="stMetricValue"] {
        color: var(--leaf-dark);
        font-weight: 900;
    }

    .stButton > button {
        border: 0;
        border-radius: 999px;
        background: linear-gradient(90deg, #2f7d4c, #7cb342);
        color: white;
        font-weight: 850;
        box-shadow: 0 12px 24px rgba(47, 125, 76, 0.22);
        min-height: 46px;
    }

    .stButton > button:hover {
        border: 0;
        color: white;
        transform: translateY(-1px);
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        border-radius: 14px;
        border-color: var(--line);
        background-color: #ffffff;
        color: var(--ink);
        opacity: 1;
    }

    div[data-baseweb="select"] span,
    div[data-testid="stNumberInput"] input,
    input::placeholder {
        color: var(--ink);
        -webkit-text-fill-color: var(--ink);
    }

    div[data-testid="stNumberInput"] button {
        border-radius: 12px;
        border: 1px solid rgba(47, 125, 76, 0.18);
        background: #f3fbf5;
        color: var(--leaf-dark);
    }

    div[data-testid="stNumberInput"] button:hover {
        background: #dff3e5;
        color: var(--leaf-dark);
    }

    div[data-testid="stNumberInput"] button:focus {
        box-shadow: 0 0 0 3px rgba(47, 125, 76, 0.16);
    }

    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
    }

    .input-summary {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
    }

    .input-pill {
        border: 1px solid rgba(47, 125, 76, 0.14);
        border-radius: 14px;
        padding: 11px 12px;
        background: rgba(255, 255, 255, 0.86);
    }

    .input-pill span {
        display: block;
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
    }

    .input-pill strong {
        display: block;
        margin-top: 4px;
        color: var(--ink);
        font-size: 14px;
        overflow-wrap: anywhere;
    }

    @media (max-width: 700px) {
        .hero-wrap {
            padding: 22px;
        }

        .hero-title {
            font-size: 36px;
        }

        .input-summary {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
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

st.markdown(
    """
    <section class="hero-wrap">
        <div class="kicker">🌱 AI crop lab</div>
        <h1 class="hero-title">Crop Yield Studio</h1>
        <p class="hero-subtitle">
            Predict crop yield with a clean ML dashboard built for quick experiments, lab demos,
            and those "let me test one more input" moments.
        </p>
        <div class="plant-row">
            <span class="plant-chip">🌾 Yield prediction</span>
            <span class="plant-chip">🌦️ Rainfall aware</span>
            <span class="plant-chip">🧪 Fertilizer inputs</span>
            <span class="plant-chip">📊 Report-ready charts</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.write("")

metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
metric_col_1.metric("Dataset Rows", metrics["rows"])
metric_col_2.metric("Model Error", metrics["mae"])
metric_col_3.metric("Accuracy Score", metrics["r2_score"])

st.write("")

form_col, result_col = st.columns([1.2, 1])

with form_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Build Your Crop Scenario")

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

    submitted = st.button("Grow The Prediction", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

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
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<div class="tiny-label">Prediction result</div>', unsafe_allow_html=True)
    if submitted:
        predicted_yield, total_production = predict_yield(model_bundle, input_data)
        st.markdown(
            f"""
            <div class="result-number">{predicted_yield} ton/ha</div>
            <p class="result-copy">
                Estimated total production: <strong>{total_production} tons</strong>.
                This scenario is ready for your project explanation or report screenshot.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(predicted_yield / 10, 1.0), text="Yield strength")
    else:
        st.markdown(
            """
            <div class="result-number">Ready</div>
            <p class="result-copy">
                Fill the crop details and click the prediction button to see the estimated yield.
            </p>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown("#### Your Current Inputs")
    st.markdown(
        f"""
        <div class="input-summary">
            <div class="input-pill"><span>State</span><strong>{state}</strong></div>
            <div class="input-pill"><span>Crop</span><strong>{crop}</strong></div>
            <div class="input-pill"><span>Season</span><strong>{season}</strong></div>
            <div class="input-pill"><span>Soil</span><strong>{soil_type}</strong></div>
            <div class="input-pill"><span>Rainfall</span><strong>{rainfall:g} mm</strong></div>
            <div class="input-pill"><span>Temperature</span><strong>{temperature:g} C</strong></div>
            <div class="input-pill"><span>Fertilizer</span><strong>{fertilizer:g} kg/ha</strong></div>
            <div class="input-pill"><span>Pesticide</span><strong>{pesticide:g} kg/ha</strong></div>
            <div class="input-pill"><span>Area</span><strong>{area:g} ha</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

chart_col_1, chart_col_2 = st.columns(2)

with chart_col_1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Crop Leaderboard")
    crop_yield = data.groupby("Crop", as_index=False)["Yield_ton_per_ha"].mean()
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#ffffff")
    sns.barplot(data=crop_yield, x="Yield_ton_per_ha", y="Crop", ax=ax, color="#56a764")
    ax.set_xlabel("Average Yield (ton/ha)")
    ax.set_ylabel("Crop")
    ax.set_facecolor("#ffffff")
    ax.grid(axis="x", color="#dcebdd")
    sns.despine(ax=ax)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

with chart_col_2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Rainfall Glow-Up")
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#ffffff")
    sns.scatterplot(
        data=data,
        x="Rainfall_mm",
        y="Yield_ton_per_ha",
        hue="Season",
        palette=["#2f7d4c", "#f7a440", "#3c91e6"],
        s=82,
        ax=ax,
    )
    ax.set_xlabel("Rainfall (mm)")
    ax.set_ylabel("Yield (ton/ha)")
    ax.set_facecolor("#ffffff")
    ax.grid(color="#dcebdd")
    sns.despine(ax=ax)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Peek at the dataset"):
    st.dataframe(data, use_container_width=True, hide_index=True)
