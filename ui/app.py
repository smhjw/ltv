import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.retention import RetentionModel
from models.ltv import LTVModel
from models.roas import ROASCalculator

st.set_page_config(page_title="Game LTV & Retention Predictor", layout="wide")

st.title("Game Retention & LTV Prediction System")

# Sidebar for Navigation
page = st.sidebar.selectbox("Module", ["Retention Prediction", "LTV Prediction", "ROAS Payback", "Data Management"])

# Session State Initialization
if 'retention_data' not in st.session_state:
    st.session_state.retention_data = pd.DataFrame({
        'Day': [1, 2, 3, 7, 14, 30],
        'Retention': [0.50, 0.40, 0.35, 0.25, 0.20, 0.15]
    })
if 'ltv_data' not in st.session_state:
    st.session_state.ltv_data = pd.DataFrame({
        'Day': [1, 2, 3, 7, 14, 30],
        'LTV': [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    })
if 'predicted_retention' not in st.session_state:
    st.session_state.predicted_retention = None
if 'predicted_ltv' not in st.session_state:
    st.session_state.predicted_ltv = None
if 'roas_params' not in st.session_state:
    st.session_state.roas_params = {'cpi': 2.0}

def plot_with_interval(days, mean, lower, upper, title, y_axis_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=mean, mode='lines', name='Prediction', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=days, y=upper, mode='lines', name='Upper Bound', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=days, y=lower, mode='lines', name='Lower Bound', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,255,0.2)', showlegend=True))
    fig.update_layout(title=title, xaxis_title="Days", yaxis_title=y_axis_title)
    return fig

# --- Retention Module ---
if page == "Retention Prediction":
    st.header("1. Retention Prediction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Data")
        edited_df = st.data_editor(st.session_state.retention_data, num_rows="dynamic")
        st.session_state.retention_data = edited_df
        
        model_type = st.selectbox("Model Type", ["weibull", "lognormal"])
        prediction_days = st.number_input("Predict up to Day", value=90, min_value=30)
        
        if st.button("Run Retention Prediction"):
            try:
                days = edited_df['Day'].values
                rates = edited_df['Retention'].values
                
                # Validation
                if len(days) < 3:
                    st.error("Need at least 3 data points.")
                else:
                    model = RetentionModel(model_type=model_type)
                    model.fit(days, rates)
                    metrics = model.get_metrics()
                    
                    st.success("Model Fitted!")
                    st.write("Metrics:", metrics)
                    
                    if metrics['MAPE'] > 10:
                        st.warning("MAPE > 10%. Consider checking data or switching model.")
                    
                    # Predict
                    future_days = np.arange(1, prediction_days + 1)
                    pred, lower, upper = model.predict_with_interval(future_days)
                    
                    st.session_state.predicted_retention = {
                        'model': model,
                        'days': future_days,
                        'pred': pred
                    }
                    
                    # Plot
                    fig = plot_with_interval(future_days, pred, lower, upper, "Retention Curve", "Retention Rate")
                    # Add actuals
                    fig.add_trace(go.Scatter(x=days, y=rates, mode='markers', name='Actual', marker=dict(color='red')))
                    
                    st.session_state.retention_fig = fig
            except Exception as e:
                st.error(f"Error: {str(e)}")

    with col2:
        if 'retention_fig' in st.session_state:
            st.plotly_chart(st.session_state.retention_fig, use_container_width=True)

# --- LTV Module ---
elif page == "LTV Prediction":
    st.header("2. LTV Prediction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Data")
        edited_df = st.data_editor(st.session_state.ltv_data, num_rows="dynamic")
        st.session_state.ltv_data = edited_df
        
        model_type = st.selectbox("LTV Model", ["power_law", "logarithmic", "retention_based"])
        prediction_days = st.number_input("Predict up to Day", value=365, min_value=30)
        
        if st.button("Run LTV Prediction"):
            try:
                days = edited_df['Day'].values
                vals = edited_df['LTV'].values
                
                ret_model = None
                if model_type == 'retention_based':
                    if st.session_state.predicted_retention:
                        ret_model = st.session_state.predicted_retention['model']
                    else:
                        st.error("Please run Retention Prediction first for 'retention_based' model.")
                        st.stop()
                
                model = LTVModel(model_type=model_type)
                model.fit(days, vals, retention_model=ret_model)
                
                future_days = np.arange(1, prediction_days + 1)
                pred, lower, upper = model.predict_with_interval(future_days)
                
                st.session_state.predicted_ltv = {
                    'days': future_days,
                    'pred': pred
                }
                
                fig = plot_with_interval(future_days, pred, lower, upper, "LTV Curve", "Cumulative LTV")
                fig.add_trace(go.Scatter(x=days, y=vals, mode='markers', name='Actual', marker=dict(color='red')))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Key Metrics
                st.write("Predicted LTV:")
                st.write({
                    "D90": np.interp(90, future_days, pred),
                    "D180": np.interp(180, future_days, pred),
                    "D365": np.interp(365, future_days, pred)
                })
                
                # Sensitivity (if retention based)
                if model_type == 'retention_based':
                    sens = model.sensitivity_analysis(future_days)
                    st.subheader("Sensitivity Analysis (Retention ±20%)")
                    sens_df = pd.DataFrame(sens)
                    sens_df['Day'] = future_days
                    st.line_chart(sens_df.set_index('Day'))

            except Exception as e:
                st.error(f"Error: {str(e)}")

# --- ROAS Module ---
elif page == "ROAS Payback":
    st.header("3. ROAS Payback Analysis")
    
    if st.session_state.predicted_ltv is None:
        st.warning("Please run LTV Prediction first.")
    else:
        cpi = st.number_input("CPI (Cost Per Install)", value=st.session_state.roas_params.get('cpi', 2.0))
        st.session_state.roas_params['cpi'] = cpi
        
        ltv_days = st.session_state.predicted_ltv['days']
        ltv_vals = st.session_state.predicted_ltv['pred']
        
        roas_calc = ROASCalculator(cpi, ltv_days, ltv_vals)
        roas_curve = roas_calc.calculate_roas()
        payback_day = roas_calc.get_payback_period()
        
        st.metric("Payback Period (Days)", f"{payback_day if payback_day else '> ' + str(max(ltv_days))}")
        st.write(roas_calc.get_metrics_at_days([90, 180, 365]))
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ltv_days, y=roas_curve * 100, mode='lines', name='ROAS %'))
        fig.add_hline(y=100, line_dash="dash", line_color="green", annotation_text="Breakeven")
        fig.update_layout(title="ROAS Curve", xaxis_title="Days", yaxis_title="ROAS %")
        st.plotly_chart(fig, use_container_width=True)

# --- Data Management ---
elif page == "Data Management":
    st.header("4. Data Management")
    
    st.subheader("Import/Export")
    
    # Export
    data_export = {
        "retention": st.session_state.retention_data.to_dict(orient='records'),
        "ltv": st.session_state.ltv_data.to_dict(orient='records')
    }
    st.download_button("Download Config JSON", data=json.dumps(data_export, indent=2), file_name="config.json")
    
    # Import
    uploaded_file = st.file_uploader("Upload Config JSON", type="json")
    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            st.session_state.retention_data = pd.DataFrame(data['retention'])
            st.session_state.ltv_data = pd.DataFrame(data['ltv'])
            st.success("Data loaded!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            
    st.subheader("CSV Import Helpers")
    col_csv1, col_csv2 = st.columns(2)
    with col_csv1:
        st.write("Upload Retention CSV (Columns: Day, Retention)")
        ret_csv = st.file_uploader("Retention CSV", type="csv")
        if ret_csv:
            try:
                df = pd.read_csv(ret_csv)
                if 'Day' in df.columns and 'Retention' in df.columns:
                    st.session_state.retention_data = df[['Day', 'Retention']]
                    st.success("Retention data loaded!")
                else:
                    st.error("CSV must have 'Day' and 'Retention' columns.")
            except Exception as e:
                st.error(f"Error: {e}")

    with col_csv2:
        st.write("Upload LTV CSV (Columns: Day, LTV)")
        ltv_csv = st.file_uploader("LTV CSV", type="csv")
        if ltv_csv:
            try:
                df = pd.read_csv(ltv_csv)
                if 'Day' in df.columns and 'LTV' in df.columns:
                    st.session_state.ltv_data = df[['Day', 'LTV']]
                    st.success("LTV data loaded!")
                else:
                    st.error("CSV must have 'Day' and 'LTV' columns.")
            except Exception as e:
                st.error(f"Error: {e}")

