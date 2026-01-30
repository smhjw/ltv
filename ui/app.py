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
            "channel_roi": "Channel ROI Monitor",
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
        "channel_roi": {
            "header": "4. Channel ROI Monitor",
            "input_header": "Channel Data Input",
            "upload_text": "Upload Channel Data (CSV: Channel, CPI, Original_LTV)",
            "sample_btn": "Load Sample Data",
            "audit_params": "Audit Parameters",
            "share_ratio": "Platform Share Ratio",
            "safety_margin": "Safety Margin",
            "run_audit": "Run Audit & Monitor",
            "report_title": "Channel ROI Monitor Report",
            "status_breakeven": "Breakeven",
            "status_loss": "Unrecovered",
            "warn_normal": "Normal",
            "warn_safe": "Safe",
            "warn_high": "High Risk",
            "warn_abnormal": "Abnormal",
            "push_btn": "Push to WeCom",
            "push_success": "Notification Pushed to: Ops, Finance, UA Team"
        },
        "data": {
            "header": "5. Data Management",
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
            "channel_roi": "渠道回本监控",
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
        "channel_roi": {
            "header": "4. 渠道回本监控",
            "input_header": "渠道数据录入",
            "upload_text": "上传渠道数据 (CSV列名: Channel, CPI, Original_LTV)",
            "sample_btn": "加载示例数据",
            "audit_params": "审计参数设置",
            "share_ratio": "平台分成比例 (开发者实得)",
            "safety_margin": "安全阈值 (Fluctuation Buffer)",
            "run_audit": "运行审计监控",
            "report_title": "渠道回本监控日报",
            "status_breakeven": "已回本",
            "status_loss": "未回本",
            "warn_normal": "正常",
            "warn_safe": "安全",
            "warn_high": "高风险",
            "warn_abnormal": "数据异常",
            "push_btn": "推送至企业微信",
            "push_success": "已推送预警信息至：投放负责人、财务、运营"
        },
        "data": {
            "header": "5. 数据管理",
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
    # Import Fonts
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Roboto:wght@400;500&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Inter', 'Roboto', "Helvetica Neue", sans-serif;
            }
            
            /* Responsive Helper Classes (Approximation) */
            @media (max-width: 768px) {
                /* Force columns to stack on mobile if not already */
                [data-testid="column"] {
                    width: 100% !important;
                    flex: 1 1 auto !important;
                    min-width: 100% !important;
                }
            }
        </style>
    """, unsafe_allow_html=True)

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
                .stMarkdown, .stText, h1, h2, h3, p, div, span {
                    color: black !important;
                }
                /* Reset specific chart colors if needed */
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
    t["modules"]["channel_roi"]: "Channel ROI Monitor",
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

if 'channel_roi_data' not in st.session_state:
    st.session_state.channel_roi_data = None
if 'channel_audit_report' not in st.session_state:
    st.session_state.channel_audit_report = None

def format_value(val, is_percent=False, currency=None):
    if is_percent:
        return f"{val:.1f}%"
    if currency:
        # Simple currency mapping
        symbol = {'USD': '$', 'CNY': '¥', 'EUR': '€', 'JPY': '¥'}.get(currency, '')
        return f"{symbol}{val:,.1f}"
    return f"{val:.2f}"

def plot_with_interval(days, mean, lower, upper, title, y_axis_title, lang_dict, theme='dark', currency=None, is_percent=False, cpi_line=None):
    layout_template = 'plotly_dark' if theme == 'dark' else 'plotly_white'
    
    # Task 2: Visual Channels & Color Strategy
    # Dark Theme: Neon Cyan #4DFFFF or Amber #FFB74D for lower bounds
    # Light Theme: Darker contrast color
    if theme == 'dark':
        ci_color_fill = 'rgba(77, 255, 255, 0.2)' # Neon Cyan with transparency
        ci_line_color = '#4DFFFF'
        main_line_color = '#29B6F6' # Light Blue
        text_color = '#E0E0E0'
        grid_color = '#333333'
    else:
        ci_color_fill = 'rgba(255, 167, 38, 0.2)' # Orange/Amber
        ci_line_color = '#FF9800'
        main_line_color = '#1976D2' # Dark Blue
        text_color = '#333333'
        grid_color = '#E0E0E0'

    font_family = "Inter, Roboto, 'Helvetica Neue', sans-serif"
    
    fig = go.Figure()
    
    # Prediction (Mean)
    fig.add_trace(go.Scatter(
        x=days, y=mean, mode='lines', 
        name=lang_dict['prediction'], 
        line=dict(color=main_line_color, width=3),
        hovertemplate=f"<b>%{{y:.1f}}{'%' if is_percent else ''}</b><extra></extra>"
    ))
    
    # CPI Line (if provided) for LTV/ROI integration
    if cpi_line is not None:
        fig.add_hline(
            y=cpi_line, 
            line_dash="dash", 
            line_color="#FF5252", 
            annotation_text=f"CPI (Cost)", 
            annotation_position="top left",
            annotation_font=dict(color="#FF5252")
        )

    # Upper Bound (Invisible for fill, but we add a stroke as requested)
    # "Task 1: ...increase 2px semi-transparent stroke" - Applying to bounds
    fig.add_trace(go.Scatter(
        x=days, y=upper, mode='lines', 
        name=lang_dict['upper'], 
        line=dict(width=2, color=ci_line_color),
        opacity=0.5,
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Lower Bound (Filled)
    fig.add_trace(go.Scatter(
        x=days, y=lower, mode='lines', 
        name=lang_dict['lower'], 
        line=dict(width=2, color=ci_line_color), 
        fill='tonexty', 
        fillcolor=ci_color_fill, 
        opacity=0.5,
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Task 3: Annotations for specific days (Every 20 days)
    annotations = []
    # Dynamic interval: Every 20 days within the range
    max_day = int(max(days))
    target_days = [d for d in range(20, max_day + 1, 20)]
    
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
                ax=0, ay=-25,
                bgcolor="white",
                bordercolor="black",
                borderpad=4,
                font=dict(color="black", size=12, family=font_family),
                opacity=0.9
            ))
            
    fig.update_layout(
        template=layout_template,
        title=dict(text=title, font=dict(size=18, family=font_family, color=text_color)),
        xaxis=dict(
            title=dict(text="Days", font=dict(color=text_color)), 
            gridcolor=grid_color,
            tickfont=dict(family=font_family, color=text_color, size=12)
        ),
        yaxis=dict(
            title=dict(text=y_axis_title, font=dict(color=text_color)), 
            gridcolor=grid_color,
            tickfont=dict(family=font_family, color=text_color, size=12)
        ),
        legend=dict(
            orientation="h", 
            yanchor="bottom", y=1.02, 
            xanchor="right", x=1,
            font=dict(family=font_family, size=13, color=text_color),
            itemwidth=30  # Spacing
        ),
        annotations=annotations,
        hovermode="x unified",
        height=500, # Task 1: Height >= 500px
        margin=dict(l=40, r=40, t=80, b=40),
    )
    
    # Task 5: Responsive layout
    # fig.update_layout(autosize=True) # Streamlit handles this with use_container_width=True
    
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

# --- LTV & ROAS Integrated Module ---
elif page == "LTV Prediction":
    # Renamed header to reflect integration
    st.header(t["ltv"]["header"] + " & ROI Analysis")
    
    # Layout: Left Input (25-30%) + Right Chart (70-75%)
    col_left, col_right = st.columns([1, 3])
    
    with col_left:
        st.subheader(t["ltv"]["input_header"])
        
        # Currency Selector
        currency_options = ["USD", "CNY", "EUR", "JPY"]
        st.session_state.currency = st.selectbox(
            t["currency_label"], 
            currency_options, 
            index=currency_options.index(st.session_state.get('currency', 'USD'))
        )
        
        # Data Editor
        edited_df = st.data_editor(st.session_state.ltv_data, num_rows="dynamic", use_container_width=True)
        st.session_state.ltv_data = edited_df
        
        # LTV Model Parameters
        model_type = st.selectbox(t["ltv"]["model_type"], ["power_law", "logarithmic", "retention_based"])
        prediction_days = st.number_input(t["ltv"]["predict_days"], value=365, min_value=30)
        
        # ROI / CPI Integration
        st.divider()
        st.markdown("### ROI / Cost Parameters")
        currency_symbol = {'USD': '$', 'CNY': '¥', 'EUR': '€', 'JPY': '¥'}.get(st.session_state.get('currency', 'USD'), '')
        cpi_input = st.number_input(f"CPI ({currency_symbol})", value=st.session_state.roas_params.get('cpi', 2.0))
        st.session_state.roas_params['cpi'] = cpi_input
        
        if st.button(t["ltv"]["run_btn"], type="primary", use_container_width=True):
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

    with col_right:
        if st.session_state.predicted_ltv:
            data = st.session_state.predicted_ltv
            
            # --- ROI Calculation ---
            cpi = st.session_state.roas_params['cpi']
            # Find payback day (first day where LTV >= CPI)
            payback_day = None
            for d, val in zip(data['days'], data['pred']):
                if val >= cpi:
                    payback_day = d
                    break
            
            # Key Metrics (Integrated LTV & ROI)
            metrics = {
                "D90 LTV": np.interp(90, data['days'], data['pred']),
                "D180 LTV": np.interp(180, data['days'], data['pred']),
                "D365 LTV": np.interp(365, data['days'], data['pred']),
                "Payback Day": payback_day if payback_day else f"> {max(data['days'])}",
                "D90 ROI": (np.interp(90, data['days'], data['pred']) / cpi * 100) if cpi > 0 else 0
            }
            
            # Display Metrics
            m_cols = st.columns(len(metrics))
            for i, (k, v) in enumerate(metrics.items()):
                if "ROI" in k:
                     m_cols[i].metric(k, f"{v:.1f}%")
                elif "Day" in k and isinstance(v, (int, float)):
                     m_cols[i].metric(k, f"{int(v)} Days")
                elif "Day" in k: # String case
                     m_cols[i].metric(k, v)
                else:
                     m_cols[i].metric(k, format_value(v, currency=st.session_state.currency))

            # Plot (Integrated LTV Curve + CPI Line)
            fig = plot_with_interval(
                data['days'], data['pred'], data['lower'], data['upper'], 
                t["ltv"]["plot_title"] + " vs CPI (ROI Analysis)", 
                t["ltv"]["y_axis"], 
                t["retention"],
                theme=st.session_state.theme,
                currency=st.session_state.currency,
                cpi_line=cpi
            )
            
            # Add Actuals
            fig.add_trace(go.Scatter(
                x=data['input_days'], y=data['input_vals'], 
                mode='markers', 
                name=t["ltv"]["actual"], 
                marker=dict(color='#FF5252', size=8, line=dict(width=1.5, color='white'), opacity=0.9)
            ))
            
            # High resolution & Responsive
            st.plotly_chart(fig, use_container_width=True, config={'responsive': True, 'displayModeBar': False})
            
            # Optional: Detailed ROI Table
            with st.expander("Detailed ROI / ROAS Data", expanded=False):
                roi_data = []
                for d in [30, 60, 90, 180, 365]:
                    if d <= max(data['days']):
                        val = np.interp(d, data['days'], data['pred'])
                        roi = (val / cpi * 100) if cpi > 0 else 0
                        roi_data.append({"Day": d, "LTV": val, "ROI (%)": roi})
                st.table(pd.DataFrame(roi_data).style.format({"LTV": "{:.2f}", "ROI (%)": "{:.1f}%"}))

        else:
            st.info("👈 Please enter data and run prediction / 请在左侧输入数据并运行预测")

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

# --- Channel ROI Monitor Module ---
elif page == "Channel ROI Monitor":
    st.header(t["channel_roi"]["header"])
    
    col_input, col_params = st.columns([1, 1])
    
    with col_input:
        st.subheader(t["channel_roi"]["input_header"])
        
        # File Uploader
        uploaded_file = st.file_uploader(t["channel_roi"]["upload_text"], type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                # Basic Validation
                req_cols = ['Channel', 'CPI', 'Original_LTV']
                if all(col in df.columns for col in req_cols):
                    st.session_state.channel_roi_data = df
                else:
                    st.error(f"CSV missing columns: {req_cols}")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
        
        # Sample Data Button
        if st.button(t["channel_roi"]["sample_btn"]):
            st.session_state.channel_roi_data = pd.DataFrame({
                'Channel': ['FB_Ads', 'Google_UAC', 'TikTok', 'Organic', 'Unity_Ads', 'IronSource'],
                'CPI': [25.0, 30.0, 15.0, 0.0, 40.0, 10.0],
                'Original_LTV': [40.0, 35.0, 10.0, 15.0, 20.0, 100.0]
            })
            
        if st.session_state.channel_roi_data is not None:
            st.dataframe(st.session_state.channel_roi_data, hide_index=True)
            
    with col_params:
        st.subheader(t["channel_roi"]["audit_params"])
        
        # Parameters
        share_ratio = st.slider(t["channel_roi"]["share_ratio"], 0.0, 1.0, 0.7, 0.05)
        safety_margin = st.slider(t["channel_roi"]["safety_margin"], 1.0, 1.5, 1.2, 0.05)
        
        # Run Audit
        if st.button(t["channel_roi"]["run_audit"], type="primary"):
            if st.session_state.channel_roi_data is not None:
                df = st.session_state.channel_roi_data.copy()
                
                # 1. Calculate Actual LTV
                df['Actual_LTV'] = df['Original_LTV'] * share_ratio
                
                # 2. ROI Calculation (Handle CPI=0)
                df['ROI'] = df.apply(lambda x: x['Actual_LTV'] / x['CPI'] if x['CPI'] > 0 else (999.0 if x['Actual_LTV'] > 0 else 0), axis=1)
                
                # 3. Status Determination
                def get_status(row):
                    if row['ROI'] > 1.0:
                        return t["channel_roi"]["status_breakeven"]
                    return t["channel_roi"]["status_loss"]
                
                df['Status'] = df.apply(get_status, axis=1)
                
                # 4. Warning Level
                def get_warning(row):
                    # Anomaly Check
                    if row['CPI'] > 0:
                        ratio = row['Actual_LTV'] / row['CPI']
                        if ratio < 0.5:
                            return t["channel_roi"]["warn_abnormal"] # < 0.5 High risk/Abnormal
                        if ratio > 3.0:
                            return t["channel_roi"]["warn_abnormal"] # > 3 Suspicious
                            
                    # Safety Check
                    if row['ROI'] >= safety_margin:
                        return t["channel_roi"]["warn_safe"]
                    elif row['ROI'] >= 1.0:
                        return t["channel_roi"]["warn_normal"]
                    else:
                        return t["channel_roi"]["warn_high"] # Loss
                        
                df['Warning_Level'] = df.apply(get_warning, axis=1)
                
                st.session_state.channel_audit_report = df
                st.success("Audit Completed!")
            else:
                st.warning("Please load data first.")

    # Report Section
    if st.session_state.channel_audit_report is not None:
        st.divider()
        st.subheader(t["channel_roi"]["report_title"])
        
        report = st.session_state.channel_audit_report
        
        # Styling Logic
        def highlight_rows(row):
            styles = [''] * len(row)
            # High Risk / Loss -> Red background
            if row['Warning_Level'] == t["channel_roi"]["warn_high"]:
                styles = ['background-color: #ffcccc; color: black'] * len(row)
            # Abnormal -> Yellow background
            elif row['Warning_Level'] == t["channel_roi"]["warn_abnormal"]:
                 styles = ['background-color: #fff3cd; color: black'] * len(row)
            # Safe -> Greenish text
            elif row['Warning_Level'] == t["channel_roi"]["warn_safe"]:
                 styles = ['color: green'] * len(row)
            return styles

        st.dataframe(
            report.style.apply(highlight_rows, axis=1)
            .format({
                "CPI": "¥{:.2f}", 
                "Original_LTV": "¥{:.2f}", 
                "Actual_LTV": "¥{:.2f}",
                "ROI": "{:.2f}x"
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Actions
        col_act1, col_act2 = st.columns([1, 4])
        with col_act1:
            csv = report.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV",
                csv,
                "channel_roi_report.csv",
                "text/csv"
            )
        with col_act2:
            if st.button(t["channel_roi"]["push_btn"]):
                # Filter bad channels for notification
                bad_channels = report[report['Warning_Level'].isin([t["channel_roi"]["warn_high"], t["channel_roi"]["warn_abnormal"]])]
                msg = t["channel_roi"]["push_success"]
                if not bad_channels.empty:
                    msg += f"\n\n🚨 Alert Channels ({len(bad_channels)}): {', '.join(bad_channels['Channel'].tolist())}"
                st.info(msg)

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
