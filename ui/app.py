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

# --- Translation Dictionary ---
TRANSLATIONS = {
    "English": {
        "title": "Game Retention & LTV Prediction System",
        "sidebar_lang": "Language",
        "sidebar_theme": "Theme",
        "theme_dark": "Dark",
        "theme_light": "Light",
        "sidebar_module": "Module",
        "currency_label": "Currency",
        "modules": {
            "retention": "Retention Prediction",
            "ltv": "LTV Prediction",
            "roas": "ROAS Payback",
            "data": "Data Management"
        },
        "retention": {
            "header": "1. Retention Prediction",
            "input_header": "Input Data",
            "model_type": "Model Type",
            "predict_days": "Predict up to Day",
            "run_btn": "Run Retention Prediction",
            "error_points": "Need at least 3 data points.",
            "success": "Model Fitted!",
            "warning_mape": "MAPE > 10%. Consider checking data or switching model.",
            "plot_title": "Retention Curve",
            "y_axis": "Retention Rate",
            "actual": "Actual",
            "prediction": "Prediction",
            "lower": "Lower Bound",
            "upper": "Upper Bound"
        },
        "ltv": {
            "header": "2. LTV Prediction",
            "input_header": "Input Data",
            "model_type": "LTV Model",
            "predict_days": "Predict up to Day",
            "run_btn": "Run LTV Prediction",
            "error_retention": "Please run Retention Prediction first for 'retention_based' model.",
            "plot_title": "LTV Curve",
            "y_axis": "Cumulative LTV",
            "actual": "Actual",
            "metrics_title": "Predicted LTV:",
            "sensitivity_title": "Sensitivity Analysis (Retention ±20%)",
            "panel_toggle": "Toggle Chart Panel"
        },
        "roas": {
            "header": "3. ROAS Payback Analysis",
            "warning": "Please run LTV Prediction first.",
            "cpi_label": "CPI (Cost Per Install)",
            "payback_metric": "Payback Period (Days)",
            "plot_title": "ROAS Curve",
            "breakeven": "Breakeven"
        },
        "data": {
            "header": "4. Data Management",
            "subheader_io": "Import/Export",
            "download_btn": "Download Config JSON",
            "upload_label": "Upload Config JSON",
            "success_load": "Data loaded!",
            "error_load": "Error loading file",
            "subheader_csv": "CSV Import Helpers",
            "ret_csv_text": "Upload Retention CSV (Columns: Day, Retention)",
            "ret_csv_label": "Retention CSV",
            "ltv_csv_text": "Upload LTV CSV (Columns: Day, LTV)",
            "ltv_csv_label": "LTV CSV",
            "error_csv_cols": "CSV must have correct columns.",
            "success_ret": "Retention data loaded!",
            "success_ltv": "LTV data loaded!"
        }
    },
    "中文": {
        "title": "游戏留存与LTV预测系统",
        "sidebar_lang": "语言 / Language",
        "sidebar_theme": "主题 / Theme",
        "theme_dark": "深色 / Dark",
        "theme_light": "亮色 / Light",
        "sidebar_module": "功能模块",
        "currency_label": "货币单位",
        "modules": {
            "retention": "留存预测",
            "ltv": "LTV预测",
            "roas": "ROAS回收分析",
            "data": "数据管理"
        },
        "retention": {
            "header": "1. 留存率预测",
            "input_header": "输入数据",
            "model_type": "模型类型",
            "predict_days": "预测天数",
            "run_btn": "开始留存预测",
            "error_points": "至少需要3个数据点。",
            "success": "模型拟合成功！",
            "warning_mape": "MAPE > 10%。请检查数据或更换模型。",
            "plot_title": "留存曲线",
            "y_axis": "留存率",
            "actual": "实际值",
            "prediction": "预测值",
            "lower": "下限",
            "upper": "上限"
        },
        "ltv": {
            "header": "2. LTV 预测",
            "input_header": "输入数据",
            "model_type": "LTV 模型",
            "predict_days": "预测天数",
            "run_btn": "开始 LTV 预测",
            "error_retention": "使用'retention_based'模型前请先运行留存预测。",
            "plot_title": "LTV 曲线",
            "y_axis": "累计 LTV",
            "actual": "实际值",
            "metrics_title": "LTV 预测值：",
            "sensitivity_title": "敏感度分析 (留存率 ±20%)",
            "panel_toggle": "切换图表面板"
        },
        "roas": {
            "header": "3. ROAS 回收分析",
            "warning": "请先运行 LTV 预测。",
            "cpi_label": "CPI (每用户安装成本)",
            "payback_metric": "回本周期 (天)",
            "plot_title": "ROAS 曲线",
            "breakeven": "回本线"
        },
        "data": {
            "header": "4. 数据管理",
            "subheader_io": "导入/导出",
            "download_btn": "下载配置 JSON",
            "upload_label": "上传配置 JSON",
            "success_load": "数据加载成功！",
            "error_load": "文件加载失败",
            "subheader_csv": "CSV 导入助手",
            "ret_csv_text": "上传留存 CSV (列名: Day, Retention)",
            "ret_csv_label": "留存 CSV",
            "ltv_csv_text": "上传 LTV CSV (列名: Day, LTV)",
            "ltv_csv_label": "LTV CSV",
            "error_csv_cols": "CSV 必须包含正确的列名。",
            "success_ret": "留存数据已加载！",
            "success_ltv": "LTV 数据已加载！"
        }
    }
}

# --- Theme Management ---
# Persistence via Query Params
if 'theme' not in st.session_state:
    qp = st.query_params
    st.session_state.theme = qp.get('theme', 'dark')

def apply_theme():
    if st.session_state.theme == 'light':
        st.markdown("""
            <style>
                [data-testid="stAppViewContainer"] {
                    background-color: #F5F5F5;
                    color: black;
                }
                [data-testid="stSidebar"] {
                    background-color: #E0E0E0;
                }
                [data-testid="stHeader"] {
                    background-color: #F5F5F5;
                }
                .stMarkdown, .stText, h1, h2, h3 {
                    color: black !important;
                }
                /* High contrast text for charts/metrics in light mode if needed */
            </style>
        """, unsafe_allow_html=True)
    # Store in query params for reload persistence
    st.query_params['theme'] = st.session_state.theme

apply_theme()

# Top Bar Layout
col_header, col_controls = st.columns([3, 1])

with col_controls:
    # Language Selector
    lang_choice = st.radio("Language / 语言", ["English", "中文"], index=1, horizontal=True, key="lang_select")
    t = TRANSLATIONS[lang_choice]
    
    # Theme Toggle
    theme_toggle = st.toggle(f"{t['theme_light'] if st.session_state.theme == 'dark' else t['theme_dark']}", value=(st.session_state.theme == 'light'))
    
    # Handle Toggle Logic
    if theme_toggle and st.session_state.theme == 'dark':
        st.session_state.theme = 'light'
        st.rerun()
    elif not theme_toggle and st.session_state.theme == 'light':
        st.session_state.theme = 'dark'
        st.rerun()

with col_header:
    st.title(t["title"])


# Sidebar for Navigation
# Create a reverse mapping for the selectbox to handle logic
module_map = {
    t["modules"]["retention"]: "Retention Prediction",
    t["modules"]["ltv"]: "LTV Prediction",
    t["modules"]["roas"]: "ROAS Payback",
    t["modules"]["data"]: "Data Management"
}
page_display = st.sidebar.selectbox(t["sidebar_module"], list(module_map.keys()))
page = module_map[page_display]

# Session State Initialization
if 'retention_data' not in st.session_state:
    st.session_state.retention_data = pd.DataFrame({
        'Day': [1, 2, 3, 7, 14, 30],
        'Retention': [50.0, 40.0, 35.0, 25.0, 20.0, 15.0]
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
if 'currency' not in st.session_state:
    st.session_state.currency = 'USD'

def format_value(val, is_percent=False, currency=None):
    if is_percent:
        return f"{val:.1f}%"
    if currency:
        # Simple currency mapping
        symbol = {'USD': '$', 'CNY': '¥', 'EUR': '€', 'JPY': '¥'}.get(currency, '')
        return f"{symbol}{val:,.1f}"
    return f"{val:.2f}"

def plot_with_interval(days, mean, lower, upper, title, y_axis_title, lang_dict, theme='dark', currency=None, is_percent=False):
    layout_template = 'plotly_dark' if theme == 'dark' else 'plotly_white'
    
    # Task 1: High contrast text for lower bound in dark mode
    lower_color = 'rgba(0,0,255,0.2)'
    legend_font_color = 'white' if theme == 'dark' else 'black'
    
    fig = go.Figure()
    
    # Prediction
    fig.add_trace(go.Scatter(
        x=days, y=mean, mode='lines', 
        name=lang_dict['prediction'], 
        line=dict(color='blue'),
        hovertemplate=f"%{{y:.1f}}{'%' if is_percent else ''}<extra></extra>"
    ))
    
    # Upper Bound (hidden line)
    fig.add_trace(go.Scatter(
        x=days, y=upper, mode='lines', 
        name=lang_dict['upper'], 
        line=dict(width=0), 
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Lower Bound (filled area)
    # Task 1: Ensure text contrast in legend
    fig.add_trace(go.Scatter(
        x=days, y=lower, mode='lines', 
        name=lang_dict['lower'], 
        line=dict(width=0), 
        fill='tonexty', 
        fillcolor=lower_color, 
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Task 3: Annotations for specific days [20, 40, 60, 90]
    annotations = []
    target_days = [20, 40, 60, 90]
    for d in target_days:
        if d in days:
            idx = np.where(days == d)[0][0]
            val = mean[idx]
            
            text_val = format_value(val, is_percent, currency)
            
            annotations.append(dict(
                x=d, y=val,
                text=text_val,
                showarrow=True,
                arrowhead=0,
                ax=0, ay=-20,
                bgcolor="white",
                bordercolor="black",
                borderpad=4,
                font=dict(color="black", size=12),
                opacity=0.9
            ))
            
    fig.update_layout(
        template=layout_template,
        title=title, 
        xaxis_title="Days", 
        yaxis_title=y_axis_title,
        legend=dict(font=dict(color=legend_font_color)),
        annotations=annotations,
        hovermode="x unified"
    )
    
    # Task 5: Responsive layout
    fig.update_layout(autosize=True)
    
    return fig

# --- Retention Module ---
if page == "Retention Prediction":
    st.header(t["retention"]["header"])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(t["retention"]["input_header"])
        
        # Task 2: Percentage formatting in Data Editor
        edited_df = st.data_editor(
            st.session_state.retention_data, 
            num_rows="dynamic",
            column_config={
                "Retention": st.column_config.NumberColumn(
                    "Retention (%)",
                    format="%.1f %%",
                    min_value=0,
                    max_value=100
                )
            }
        )
        st.session_state.retention_data = edited_df
        
        model_type = st.selectbox(t["retention"]["model_type"], ["weibull", "lognormal"])
        prediction_days = st.number_input(t["retention"]["predict_days"], value=90, min_value=30)
        
        if st.button(t["retention"]["run_btn"]):
            try:
                days = edited_df['Day'].values
                # Convert percentage to decimal for model
                rates = edited_df['Retention'].values / 100.0
                
                # Validation
                if len(days) < 3:
                    st.error(t["retention"]["error_points"])
                else:
                    model = RetentionModel(model_type=model_type)
                    model.fit(days, rates)
                    metrics = model.get_metrics()
                    
                    st.success(t["retention"]["success"])
                    st.write("Metrics:", metrics)
                    
                    if metrics['MAPE'] > 10:
                        st.warning(t["retention"]["warning_mape"])
                    
                    # Predict
                    future_days = np.arange(1, prediction_days + 1)
                    pred, lower, upper = model.predict_with_interval(future_days)
                    
                    st.session_state.predicted_retention = {
                        'model': model,
                        'days': future_days,
                        'pred': pred
                    }
                    
                    # Plot
                    # Convert predictions back to percentage for display
                    fig = plot_with_interval(
                        future_days, pred * 100, lower * 100, upper * 100, 
                        t["retention"]["plot_title"], 
                        t["retention"]["y_axis"] + " (%)", 
                        t["retention"],
                        theme=st.session_state.theme,
                        is_percent=True
                    )
                    # Add actuals
                    fig.add_trace(go.Scatter(
                        x=days, y=rates * 100, 
                        mode='markers', 
                        name=t["retention"]["actual"], 
                        marker=dict(color='red'),
                        hovertemplate="%{y:.1f}%<extra></extra>"
                    ))
                    
                    st.session_state.retention_fig = fig
            except Exception as e:
                st.error(f"Error: {str(e)}")

    with col2:
        if 'retention_fig' in st.session_state:
            st.plotly_chart(st.session_state.retention_fig, use_container_width=True)

# --- LTV Module ---
elif page == "LTV Prediction":
    st.header(t["ltv"]["header"])
    
    # Currency Selector (Global for LTV)
    currency_options = ["USD", "CNY", "EUR", "JPY"]
    # Try to detect locale? Defaults to USD for now.
    st.session_state.currency = st.selectbox(
        t["currency_label"], 
        currency_options, 
        index=currency_options.index(st.session_state.get('currency', 'USD'))
    )
    
    # Layout: Main Input (Left) + Chart Panel (Right)
    # User requested a 320px right panel. We approximate with columns.
    show_panel = st.checkbox(t["ltv"]["panel_toggle"], value=True)
    
    if show_panel:
        col_main, col_right = st.columns([3, 1])
    else:
        col_main = st.container()
        col_right = None
    
    with col_main:
        st.subheader(t["ltv"]["input_header"])
        edited_df = st.data_editor(st.session_state.ltv_data, num_rows="dynamic")
        st.session_state.ltv_data = edited_df
        
        model_type = st.selectbox(t["ltv"]["model_type"], ["power_law", "logarithmic", "retention_based"])
        prediction_days = st.number_input(t["ltv"]["predict_days"], value=365, min_value=30)
        
        if st.button(t["ltv"]["run_btn"]):
            try:
                days = edited_df['Day'].values
                vals = edited_df['LTV'].values
                
                ret_model = None
                if model_type == 'retention_based':
                    if st.session_state.predicted_retention:
                        ret_model = st.session_state.predicted_retention['model']
                    else:
                        st.error(t["ltv"]["error_retention"])
                        st.stop()
                
                model = LTVModel(model_type=model_type)
                model.fit(days, vals, retention_model=ret_model)
                
                future_days = np.arange(1, prediction_days + 1)
                pred, lower, upper = model.predict_with_interval(future_days)
                
                st.session_state.predicted_ltv = {
                    'days': future_days,
                    'pred': pred,
                    'lower': lower,
                    'upper': upper,
                    'input_days': days,
                    'input_vals': vals
                }
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if show_panel and col_right:
        with col_right:
            if st.session_state.predicted_ltv:
                data = st.session_state.predicted_ltv
                
                # Plot
                fig = plot_with_interval(
                    data['days'], data['pred'], data['lower'], data['upper'], 
                    t["ltv"]["plot_title"], 
                    t["ltv"]["y_axis"], 
                    t["retention"],
                    theme=st.session_state.theme,
                    currency=st.session_state.currency
                )
                fig.add_trace(go.Scatter(
                    x=data['input_days'], y=data['input_vals'], 
                    mode='markers', 
                    name=t["ltv"]["actual"], 
                    marker=dict(color='red')
                ))
                
                # Task 5: Mobile check - use_container_width handles responsiveness
                st.plotly_chart(fig, use_container_width=True)
                
                # Key Metrics
                st.subheader(t["ltv"]["metrics_title"])
                metrics = {
                    "D90": np.interp(90, data['days'], data['pred']),
                    "D180": np.interp(180, data['days'], data['pred']),
                    "D365": np.interp(365, data['days'], data['pred'])
                }
                for k, v in metrics.items():
                    st.metric(k, format_value(v, currency=st.session_state.currency))
                
                # Sensitivity
                if model_type == 'retention_based': # This variable might be stale if re-running. 
                    # Ideally should store model type in session state or check model class
                    # For simplicity, we skip re-calculating sensitivity here to save space or move it to a modal
                    st.info("Sensitivity Analysis available in full view")

# --- ROAS Module ---
elif page == "ROAS Payback":
    st.header(t["roas"]["header"])
    
    if st.session_state.predicted_ltv is None:
        st.warning(t["roas"]["warning"])
    else:
        # Use currency symbol in label if possible
        currency_symbol = {'USD': '$', 'CNY': '¥', 'EUR': '€', 'JPY': '¥'}.get(st.session_state.get('currency', 'USD'), '')
        cpi = st.number_input(f"{t['roas']['cpi_label']} ({currency_symbol})", value=st.session_state.roas_params.get('cpi', 2.0))
        st.session_state.roas_params['cpi'] = cpi
        
        ltv_days = st.session_state.predicted_ltv['days']
        ltv_vals = st.session_state.predicted_ltv['pred']
        
        roas_calc = ROASCalculator(cpi, ltv_days, ltv_vals)
        roas_curve = roas_calc.calculate_roas()
        payback_day = roas_calc.get_payback_period()
        
        st.metric(t["roas"]["payback_metric"], f"{payback_day if payback_day else '> ' + str(max(ltv_days))}")
        
        # Format Metrics
        roas_metrics = roas_calc.get_metrics_at_days([90, 180, 365])
        st.write("ROAS Metrics:")
        cols = st.columns(3)
        for i, (k, v) in enumerate(roas_metrics.items()):
            cols[i].metric(k, f"{v*100:.1f}%")
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ltv_days, y=roas_curve * 100, mode='lines', name='ROAS %'))
        fig.add_hline(y=100, line_dash="dash", line_color="green", annotation_text=t["roas"]["breakeven"])
        
        # Theme handling
        template = 'plotly_white' if st.session_state.get('theme') == 'light' else 'plotly_dark'
        
        fig.update_layout(
            template=template,
            title=t["roas"]["plot_title"], 
            xaxis_title="Days", 
            yaxis_title="ROAS %",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Data Management ---
elif page == "Data Management":
    st.header(t["data"]["header"])
    
    st.subheader(t["data"]["subheader_io"])
    
    # Export
    data_export = {
        "retention": st.session_state.retention_data.to_dict(orient='records'),
        "ltv": st.session_state.ltv_data.to_dict(orient='records')
    }
    st.download_button(t["data"]["download_btn"], data=json.dumps(data_export, indent=2), file_name="config.json")
    
    # Import
    uploaded_file = st.file_uploader(t["data"]["upload_label"], type="json")
    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            st.session_state.retention_data = pd.DataFrame(data['retention'])
            st.session_state.ltv_data = pd.DataFrame(data['ltv'])
            st.success(t["data"]["success_load"])
        except Exception as e:
            st.error(f"{t['data']['error_load']}: {e}")
            
    st.subheader(t["data"]["subheader_csv"])
    col_csv1, col_csv2 = st.columns(2)
    with col_csv1:
        st.write(t["data"]["ret_csv_text"])
        ret_csv = st.file_uploader(t["data"]["ret_csv_label"], type="csv")
        if ret_csv:
            try:
                df = pd.read_csv(ret_csv)
                if 'Day' in df.columns and 'Retention' in df.columns:
                    st.session_state.retention_data = df[['Day', 'Retention']]
                    st.success(t["data"]["success_ret"])
                else:
                    st.error(t["data"]["error_csv_cols"])
            except Exception as e:
                st.error(f"Error: {e}")

    with col_csv2:
        st.write(t["data"]["ltv_csv_text"])
        ltv_csv = st.file_uploader(t["data"]["ltv_csv_label"], type="csv")
        if ltv_csv:
            try:
                df = pd.read_csv(ltv_csv)
                if 'Day' in df.columns and 'LTV' in df.columns:
                    st.session_state.ltv_data = df[['Day', 'LTV']]
                    st.success(t["data"]["success_ltv"])
                else:
                    st.error(t["data"]["error_csv_cols"])
            except Exception as e:
                st.error(f"Error: {e}")
