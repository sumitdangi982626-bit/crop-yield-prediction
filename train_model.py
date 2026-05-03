from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "crop_yield.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "crop_yield_model.joblib"


def train_model():
    data = pd.read_csv(DATA_PATH)

    target = "Yield_ton_per_ha"
    x = data.drop(columns=[target])
    y = data[target]

    categorical_features = ["State", "Crop", "Season", "Soil_Type"]
    numerical_features = [
        "Rainfall_mm",
        "Temperature_C",
        "Fertilizer_kg_per_ha",
        "Pesticide_kg_per_ha",
        "Area_ha",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numerical", StandardScaler(), numerical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        min_samples_leaf=1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    metrics = {
        "mae": round(mean_absolute_error(y_test, predictions), 3),
        "r2_score": round(r2_score(y_test, predictions), 3),
        "rows": len(data),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metrics": metrics}, MODEL_PATH)

    return metrics


if __name__ == "__main__":
    result = train_model()
    print("Model trained successfully")
    print(f"Rows used: {result['rows']}")
    print(f"MAE: {result['mae']}")
    print(f"R2 score: {result['r2_score']}")
