# Crop Yield Prediction Streamlit Application

A simple Python machine learning project that predicts crop yield using CSV data, pandas, NumPy, scikit-learn, and a Streamlit web interface.

## Tech Stack

- Python
- Pandas and NumPy for data handling
- Matplotlib and Seaborn for visualization
- Scikit-learn for machine learning
- CSV for dataset storage
- Streamlit for the web app

## Project Structure

```text
.
|-- streamlit_app.py
|-- train_model.py
|-- requirements.txt
|-- data/
|   `-- crop_yield.csv
`-- models/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run streamlit_app.py
```

## Easy Open

Double-click `Open Crop Yield App.bat` to start the app and open it in your browser.

The `Open Crop Yield App.url` shortcut opens the app link, but it works only when Streamlit is already running.

Open the app at:

```text
http://localhost:8501
```

## Notes

The included dataset is a small sample dataset for demonstration. Replace `data/crop_yield.csv` with a real dataset for better predictions.
