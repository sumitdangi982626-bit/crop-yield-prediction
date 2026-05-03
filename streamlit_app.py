from pathlib import Path
from datetime import datetime

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

    h1 a, h2 a, h3 a, h4 a {
        display: none;
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
        content: "plant lab";
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

    .hero-wrap:after {
        content: "plant lab";
        color: var(--leaf-dark);
        font-size: 30px;
        font-weight: 900;
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

    .suggestion-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin-top: 10px;
    }

    .suggestion-card {
        border: 1px solid rgba(47, 125, 76, 0.16);
        border-radius: 16px;
        padding: 14px;
        background: rgba(255, 255, 255, 0.86);
        box-shadow: 0 10px 24px rgba(31, 93, 57, 0.07);
    }

    .suggestion-card strong {
        display: block;
        color: var(--leaf-dark);
        font-size: 14px;
        margin-bottom: 5px;
    }

    .suggestion-card span {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.45;
    }

    .lab-note {
        border-left: 5px solid var(--leaf);
        border-radius: 14px;
        padding: 14px 16px;
        background: rgba(255, 255, 255, 0.82);
        color: var(--muted);
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

        .suggestion-grid {
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


def generate_suggestions(input_data, predicted_yield):
    suggestions = []

    if input_data["Rainfall_mm"] < 500:
        suggestions.append(
            (
                "Water strategy",
                "Rainfall is on the lower side. Add irrigation planning, mulching, or drought-tolerant crop choices.",
            )
        )
    elif input_data["Rainfall_mm"] > 1000:
        suggestions.append(
            (
                "Drainage check",
                "Rainfall is high. Make sure the field has drainage so roots do not stay waterlogged.",
            )
        )
    else:
        suggestions.append(
            (
                "Rainfall fit",
                "Rainfall is in a balanced range for many seasonal crops. Keep monitoring during flowering and grain filling.",
            )
        )

    if input_data["Temperature_C"] > 32:
        suggestions.append(
            (
                "Heat risk",
                "Temperature is high. Consider heat-tolerant varieties and avoid water stress during peak heat.",
            )
        )
    elif input_data["Temperature_C"] < 20:
        suggestions.append(
            (
                "Cool weather",
                "Temperature is low. Rabi crops may perform better than heat-loving crops in this condition.",
            )
        )

    if input_data["Fertilizer_kg_per_ha"] < 90:
        suggestions.append(
            (
                "Nutrient boost",
                "Fertilizer input is low. A soil test can help plan balanced NPK application.",
            )
        )
    elif input_data["Fertilizer_kg_per_ha"] > 160:
        suggestions.append(
            (
                "Avoid overuse",
                "Fertilizer input is high. Too much fertilizer can increase cost and damage soil health.",
            )
        )

    if input_data["Pesticide_kg_per_ha"] > 2.5:
        suggestions.append(
            (
                "Pest management",
                "Pesticide use is high. Try integrated pest management and regular field scouting.",
            )
        )

    if input_data["Soil_Type"] in ["Sandy", "Red"]:
        suggestions.append(
            (
                "Soil care",
                "This soil can benefit from compost or organic matter to improve water and nutrient holding capacity.",
            )
        )
    elif input_data["Soil_Type"] in ["Alluvial", "Loamy"]:
        suggestions.append(
            (
                "Soil advantage",
                "This soil type is generally crop-friendly. Focus on balanced fertilizer and timely irrigation.",
            )
        )

    if predicted_yield < 2.5:
        suggestions.append(
            (
                "Yield improvement",
                "Predicted yield is modest. Test another crop, improve irrigation, or adjust fertilizer levels.",
            )
        )
    elif predicted_yield >= 5:
        suggestions.append(
            (
                "Strong scenario",
                "Predicted yield is strong. This is a good scenario to include in your project report.",
            )
        )

    return suggestions[:6]


def build_prediction_report(record, suggestions):
    lines = [
        "Crop Yield Studio - Prediction Report",
        f"Generated: {record['Timestamp']}",
        "",
        "Selected Scenario",
        f"State: {record['State']}",
        f"Crop: {record['Crop']}",
        f"Season: {record['Season']}",
        f"Soil type: {record['Soil_Type']}",
        f"Rainfall: {record['Rainfall_mm']} mm",
        f"Temperature: {record['Temperature_C']} C",
        f"Fertilizer: {record['Fertilizer_kg_per_ha']} kg/ha",
        f"Pesticide: {record['Pesticide_kg_per_ha']} kg/ha",
        f"Area: {record['Area_ha']} ha",
        "",
        "Prediction",
        f"Predicted yield: {record['Predicted_Yield_ton_per_ha']} ton/ha",
        f"Estimated production: {record['Estimated_Production_tons']} tons",
        "",
        "Suggestions",
    ]

    for title, detail in suggestions:
        lines.append(f"- {title}: {detail}")

    return "\n".join(lines)


data = load_data()
model_bundle = load_model()
metrics = model_bundle["metrics"]

if "prediction_lab" not in st.session_state:
    st.session_state.prediction_lab = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

st.markdown(
    """
    <section class="hero-wrap">
        <div class="kicker">AI crop lab</div>
        <h1 class="hero-title">Crop Yield Studio</h1>
        <p class="hero-subtitle">
            Predict crop yield with a clean ML dashboard built for quick experiments, lab demos,
            and those "let me test one more input" moments.
        </p>
        <div class="plant-row">
            <span class="plant-chip">Yield prediction</span>
            <span class="plant-chip">Rainfall aware</span>
            <span class="plant-chip">Fertilizer inputs</span>
            <span class="plant-chip">Report-ready charts</span>
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
    if submitted:
        predicted_yield, total_production = predict_yield(model_bundle, input_data)
        suggestions = generate_suggestions(input_data, predicted_yield)
        prediction_record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **input_data,
            "Predicted_Yield_ton_per_ha": predicted_yield,
            "Estimated_Production_tons": total_production,
        }
        st.session_state.last_prediction = {
            "record": prediction_record,
            "suggestions": suggestions,
        }
        st.session_state.prediction_lab.append(prediction_record)
        st.markdown(
            f"""
            <div class="result-card">
            <div class="tiny-label">Prediction result</div>
            <div class="result-number">{predicted_yield} ton/ha</div>
            <p class="result-copy">
                Estimated total production: <strong>{total_production} tons</strong>.
                This scenario is ready for your project explanation or report screenshot.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(predicted_yield / 10, 1.0), text="Yield strength")
    else:
        last_prediction = st.session_state.last_prediction
        if last_prediction:
            record = last_prediction["record"]
            predicted_yield = record["Predicted_Yield_ton_per_ha"]
            total_production = record["Estimated_Production_tons"]
            st.markdown(
                f"""
                <div class="result-card">
                <div class="tiny-label">Latest prediction</div>
                <div class="result-number">{predicted_yield} ton/ha</div>
                <p class="result-copy">
                    Estimated total production: <strong>{total_production} tons</strong>.
                    Your latest result stays here while you explore downloads and charts.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(predicted_yield / 10, 1.0), text="Yield strength")
        else:
            st.markdown(
                """
                <div class="result-card">
                <div class="tiny-label">Prediction result</div>
                <div class="result-number">Ready</div>
                <p class="result-copy">
                    Fill the crop details and click the prediction button to see the estimated yield.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not submitted and not st.session_state.last_prediction:
        pass
    elif st.session_state.last_prediction:
        active_suggestions = st.session_state.last_prediction["suggestions"]
        st.markdown("#### Smart Suggestions")
        st.markdown(
            "<div class='suggestion-grid'>"
            + "".join(
                f"<div class='suggestion-card'><strong>{title}</strong><span>{detail}</span></div>"
                for title, detail in active_suggestions
            )
            + "</div>",
            unsafe_allow_html=True,
        )

        report_text = build_prediction_report(
            st.session_state.last_prediction["record"],
            active_suggestions,
        )
        st.download_button(
            "Download Latest Report",
            data=report_text,
            file_name="crop_yield_prediction_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown(
        """
        <div class="lab-note">
            Every prediction you run is saved in the Prediction Lab below, so you can compare scenarios and download the results.
        </div>
        """,
        unsafe_allow_html=True,
    )

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

st.subheader("Prediction Lab")
if st.session_state.prediction_lab:
    lab_df = pd.DataFrame(st.session_state.prediction_lab)
    st.dataframe(lab_df, use_container_width=True, hide_index=True)

    download_col_1, download_col_2, download_col_3 = st.columns(3)
    with download_col_1:
        st.download_button(
            "Download Lab CSV",
            data=lab_df.to_csv(index=False),
            file_name="crop_yield_prediction_lab.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_col_2:
        st.download_button(
            "Download Dataset CSV",
            data=data.to_csv(index=False),
            file_name="crop_yield_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_col_3:
        if st.button("Clear Lab", use_container_width=True):
            st.session_state.prediction_lab = []
            st.session_state.last_prediction = None
            st.rerun()
else:
    st.info("Run a prediction to add your first experiment to the lab.")

chart_col_1, chart_col_2 = st.columns(2)

with chart_col_1:
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

with chart_col_2:
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

with st.expander("Peek at the dataset"):
    st.dataframe(data, use_container_width=True, hide_index=True)


