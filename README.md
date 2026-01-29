---
title: Game LTV Predictor
emoji: 🎮
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.31.0
app_file: ui/app.py
pinned: false
---

# Game Retention & LTV Prediction System

A comprehensive tool for predicting user retention, Life-Time Value (LTV), and ROAS Payback for the gaming industry.

## Features

1.  **Retention Prediction**:
    - Input: Daily retention rates (D1, D3, D7, etc.).
    - Models: Weibull, Log-Normal.
    - Output: D30+ forecasts, 80% Confidence Intervals, Metrics (R², MAPE).

2.  **LTV Prediction**:
    - Input: Daily ARPU or Cumulative LTV.
    - Models: Power Law, Logarithmic, Retention-Based (Probabilistic Projection).
    - Output: 90/180/365-day LTV, Sensitivity Analysis.

3.  **ROAS Payback**:
    - Input: CPI, Spend, Installs.
    - Output: Payback Period, ROAS Curve, Breakeven Analysis.

4.  **Data Management**:
    - Import/Export configuration via JSON.
    - Manual data editing in UI.

## Installation & Usage

### Local Running

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the application:
    ```bash
    streamlit run ui/app.py
    ```

3.  Open your browser at `http://localhost:8501`.

### Docker

1.  Build the image:
    ```bash
    docker build -t game-ltv-predictor .
    ```

2.  Run the container:
    ```bash
    docker run -p 8501:8501 game-ltv-predictor
    ```

## Project Structure

- `models/`: Core prediction logic (Retention, LTV, ROAS).
- `ui/`: Streamlit web application.
- `tests/`: Unit tests (pytest).
- `data/`: Sample data storage.

## Testing

Run unit tests:
```bash
python -m pytest tests
```

## Deployment Guide (Streamlit Cloud)

Since this application requires a Python backend (for calculations using NumPy, SciPy, Scikit-learn), it **cannot** be hosted on GitHub Pages (which only supports static HTML/JS).

The recommended way to deploy this for free using your GitHub account is **Streamlit Cloud**.

### Steps to Deploy:

1.  **Push code to GitHub**:
    - Create a new repository on GitHub.
    - Push all files in this folder to the repository.
    ```bash
    git init
    git add .
    git commit -m "Initial commit"
    git branch -M main
    git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    git push -u origin main
    ```

2.  **Deploy on Streamlit Cloud**:
    - Go to [share.streamlit.io](https://share.streamlit.io/).
    - Sign in with your GitHub account.
    - Click **"New app"**.
    - Select your GitHub repository.
    - **Main file path**: Enter `ui/app.py`.
    - Click **"Deploy"**.

3.  **Done!**
    - Your app will be live at `https://YOUR-REPO-NAME.streamlit.app`.
