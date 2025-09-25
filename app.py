
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from permetrics.regression import RegressionMetric
from sklearn.metrics import r2_score
import shap

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    layout="wide", 
    page_title="Soil Swelling Prediction Tool",
    page_icon="⚙️",
    initial_sidebar_state="expanded"
)
st.title("⚙️ Soil Swelling Potential Prediction Tool")

# Enhanced CSS for Chrome optimization, modern UI, and fixed tabs
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #2E4053;
        --secondary-color: #34495E;
        --accent-color: #3498DB;
        --success-color: #27AE60;
        --warning-color: #F39C12;
        --danger-color: #E74C3C;
        --light-bg: #F8F9FA;
        --border-color: #E9ECEF;
        --shadow: 0 2px 4px rgba(0,0,0,0.1);
        --shadow-hover: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 16px;
        line-height: 1.6;
        color: #2C3E50;
    }
    
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
        margin: 0 auto;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }
    
    .main .block-container > div {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: var(--shadow-hover);
        margin-bottom: 2rem;
    }
    
    h1 {
        font-size: 3rem;
        font-weight: 700;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        font-size: 2.2rem;
        font-weight: 600;
        color: var(--secondary-color);
        border-bottom: 3px solid var(--accent-color);
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    
    h3 {
        font-size: 1.8rem;
        font-weight: 500;
        color: var(--secondary-color);
        margin: 1.5rem 0 1rem 0;
    }
    
    h4 {
        font-size: 1.4rem;
        font-weight: 500;
        color: var(--secondary-color);
        margin: 1rem 0;
    }
    
    .css-1d391kg {
        background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
        border-radius: 0 20px 20px 0;
    }
    
    .sidebar .sidebar-content {
        padding: 2rem 1rem;
        color: white;
    }
    
    .sidebar h2, .sidebar h3 {
        color: white !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-color), #2980B9);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2980B9, var(--accent-color));
        box-shadow: var(--shadow-hover);
        transform: translateY(-2px);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--success-color), #2ECC71);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow);
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #2ECC71, var(--success-color));
        box-shadow: var(--shadow-hover);
        transform: translateY(-2px);
    }
    
    .stMetric {
        background: var(--light-bg);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid var(--accent-color);
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        box-shadow: var(--shadow-hover);
        transform: translateY(-2px);
    }
    
    .stMetric > div > div:first-child {
        font-size: 1.1rem;
        font-weight: 500;
        color: var(--secondary-color);
    }
    
    .stMetric > div > div:nth-child(2) {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    .stAlert {
        border-radius: 12px;
        border: none;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div,
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid var(--border-color);
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus,
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        position: sticky !important;
        top: 0;
        z-index: 1000;
        background: var(--light-bg);
        padding: 0.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.75rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-color), #2980B9);
        color: white;
    }
    
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: var(--shadow);
        margin: 1rem 0;
    }
    
    .streamlit-expanderHeader {
        background: var(--light-bg);
        border-radius: 12px;
        padding: 1rem;
        border: 2px solid var(--border-color);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: #E8F4FD;
        border-color: var(--accent-color);
    }
    
    .stFileUploader {
        border: 2px dashed var(--accent-color);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(52, 152, 219, 0.05);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        background: rgba(52, 152, 219, 0.1);
        border-color: #2980B9;
    }
    
    .stCheckbox > label,
    .stRadio > label {
        font-weight: 500;
        color: var(--secondary-color);
    }
    
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        h1 {
            font-size: 2.2rem;
        }
        
        h2 {
            font-size: 1.8rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }
    }
    
    .success-badge {
        background: linear-gradient(135deg, var(--success-color), #2ECC71);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: var(--shadow);
    }
    
    .warning-badge {
        background: linear-gradient(135deg, var(--warning-color), #E67E22);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: var(--shadow);
    }
    
    .danger-badge {
        background: linear-gradient(135deg, var(--danger-color), #C0392B);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: var(--shadow);
    }
    
    .info-badge {
        background: linear-gradient(135deg, var(--accent-color), #2980B9);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: var(--shadow);
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 2s ease-in-out infinite;
    }
    
    * {
        transition: all 0.3s ease;
    }
    
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--light-bg);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--accent-color), #2980B9);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #2980B9, var(--accent-color));
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Caching: Load Models and Preprocessors
# -----------------------------
@st.cache_resource
def load_models_and_preprocessors():
    try:
        scaler_noimp = joblib.load("models/scaler.pkl")
        median_imputer = joblib.load("models/median_imputer.pkl")
        scaler_median = joblib.load("models/scaler_median.pkl")
        knn_imputer = joblib.load("models/knn_imputer.pkl")
        scaler_knn = joblib.load("models/scaler_knn.pkl")

        models_no_imp = {
            "XGBoost": joblib.load("models/xg_reg_model_without_imputation.pkl"),
            "LightGBM": joblib.load("models/lgbm_reg_model_without_imputation.pkl"),
            "CatBoost": joblib.load("models/cat_reg_model_without_imputation.pkl"),
        }
        models_med_imp = {
            "XGBoost": joblib.load("models/xg_reg_model_imputation_median.pkl"),
            "LightGBM": joblib.load("models/lgbm_reg_model_imputation_median.pkl"),
            "CatBoost": joblib.load("models/cat_reg_model_imputation_median.pkl"),
        }
        models_knn_imp = {
            "XGBoost": joblib.load("models/xg_reg_model_imputation_knn.pkl"),
            "LightGBM": joblib.load("models/lgbm_reg_model_imputation_knn.pkl"),
            "CatBoost": joblib.load("models/cat_reg_model_imputation_knn.pkl"),
        }

        explainers_no_imp = {
            "XGBoost": shap.TreeExplainer(models_no_imp["XGBoost"]),
            "LightGBM": shap.TreeExplainer(models_no_imp["LightGBM"]),
            "CatBoost": shap.TreeExplainer(models_no_imp["CatBoost"]),
        }
        explainers_med_imp = {
            "XGBoost": shap.TreeExplainer(models_med_imp["XGBoost"]),
            "LightGBM": shap.TreeExplainer(models_med_imp["LightGBM"]),
            "CatBoost": shap.TreeExplainer(models_med_imp["CatBoost"]),
        }
        explainers_knn_imp = {
            "XGBoost": shap.TreeExplainer(models_knn_imp["XGBoost"]),
            "LightGBM": shap.TreeExplainer(models_knn_imp["LightGBM"]),
            "CatBoost": shap.TreeExplainer(models_knn_imp["CatBoost"]),
        }

        return {
            "scalers": {"no_imp": scaler_noimp, "median": scaler_median, "knn": scaler_knn},
            "imputers": {"median": median_imputer, "knn": knn_imputer},
            "models": {"no_imp": models_no_imp, "median": models_med_imp, "knn": models_knn_imp},
            "explainers": {"no_imp": explainers_no_imp, "median": explainers_med_imp, "knn": explainers_knn_imp}
        }
    except FileNotFoundError:
        st.error("⚠️ **Error**: Model files not found. Please ensure the 'models' directory and all necessary .pkl files are in the same directory as this script.")
        st.stop()

with st.spinner('Loading models and preprocessors...'):
    assets = load_models_and_preprocessors()

# -----------------------------
# Helper Functions
# -----------------------------
def compute_A30(y_true, y_pred):
    evaluator = RegressionMetric(y_true.tolist(), y_pred.tolist())
    return evaluator.A30()

def plot_real_vs_pred(y_true, y_pred, model_name):
    swelling_potential_categories = {
        "Low (0-15)": "#34A853",
        "Medium (15-25)": "#F9AB00",
        "High (25-35)": "#EA4335",
        "Very High (>35)": "#6A0DAD"
    }

    fig, ax = plt.subplots(figsize=(8, 8))
    max_val = 200
    x_line = np.linspace(0, max_val, 100)
    
    df_plot = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})

    scatter_handles = {}
    for i in range(len(y_pred)):
        pi_value = y_pred[i]
        color = "#009de6"
        label = None
        if pi_value >= 35:
            label = "Very High (>35)"
            color = swelling_potential_categories[label]
        elif pi_value >= 25:
            label = "High (25-35)"
            color = swelling_potential_categories[label]
        elif pi_value >= 15:
            label = "Medium (15-25)"
            color = swelling_potential_categories[label]
        else:
            label = "Low (0-15)"
            color = swelling_potential_categories[label]
        
        if label not in scatter_handles:
            scatter_handles[label] = ax.scatter(y_true[i], y_pred[i], color=color, edgecolors="black", alpha=0.8, s=70, label=label)
        else:
            ax.scatter(y_true[i], y_pred[i], color=color, edgecolors="black", alpha=0.8, s=70)

    y_true_filtered = df_plot[df_plot['y_true'] >= 15]['y_true']
    y_pred_filtered = df_plot[df_plot['y_true'] >= 15]['y_pred']

    if len(y_true_filtered) > 0:
        A30_val = compute_A30(y_true_filtered, y_pred_filtered)
    else:
        A30_val = 0.0

    R2_val = r2_score(y_true, y_pred)

    handles = [scatter_handles[label] for label in swelling_potential_categories.keys() if label in scatter_handles]
    labels = [label for label in swelling_potential_categories.keys() if label in scatter_handles]
    
    ax.plot(x_line, x_line, 'k-', linewidth=2, label="Perfect Fit")
    ax.plot(x_line, 0.7 * x_line, 'k--', linewidth=2, label="±30% Tolerance")
    ax.plot(x_line, 1.3 * x_line, 'k--', linewidth=2)
    ax.fill_between(x_line, 0.7 * x_line, 1.3 * x_line, color='gray', alpha=0.2)
    
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Experimental Swelling Potential [%]", fontsize=16)
    ax.set_ylabel("Predicted Swelling Potential [%]", fontsize=16)

    metrics_text = f"$R^2$ = {R2_val:.2f}\nA30-index (for SP $\\geq$ 15%) = {A30_val:.2f}"
    ax.text(0.95, 0.05, metrics_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', edgecolor='black', linewidth=1.5, boxstyle="round,pad=0.3"))

    ax.legend(handles=handles, title=f"Predicted Swelling Potential\n({model_name})", loc="upper left", fontsize=14, title_fontsize=14, frameon=True, edgecolor='black')
    
    ax.grid(True, which='major', linestyle='-', linewidth=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5)
    ax.minorticks_on()

    return fig

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
        workbook = writer.book
        worksheet = writer.sheets['Predictions']
        header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top'})
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
    processed_data = output.getvalue()
    return processed_data

# -----------------------------
# Sidebar: Enhanced Model Selection
# -----------------------------
st.sidebar.markdown("## 🤖 Model Configuration")

st.sidebar.markdown("### Select Prediction Model")
model_name = st.sidebar.radio(
    "Choose Model",
    ["XGBoost", "LightGBM", "CatBoost"],
    format_func=lambda x: {
        "XGBoost": "🚀 XGBoost",
        "LightGBM": "⚡ LightGBM",
        "CatBoost": "🎯 CatBoost"
    }[x],
    horizontal=True,
    label_visibility="collapsed"
)

st.sidebar.markdown("### Select Imputation Strategy")
scenario = st.sidebar.radio(
    "Imputation Strategy",
    ["Without Imputation", "Median Imputation", "KNN Imputation"],
    format_func=lambda x: {
        "Without Imputation": "🔵 Native",
        "Median Imputation": "🟢 Median",
        "KNN Imputation": "🟣 KNN"
    }[x],
    horizontal=True,
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current Configuration")
st.sidebar.markdown(f"""
<div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;'>
    <p><strong>Model:</strong> {model_name}</p>
    <p><strong>Imputation:</strong> {scenario}</p>
</div>
""", unsafe_allow_html=True)

scenario_map = {
    "Without Imputation": "no_imp",
    "Median Imputation": "median",
    "KNN Imputation": "knn"
}
scenario_key = scenario_map[scenario]

# -----------------------------
# Input Features Information
# -----------------------------
inputs_info = {
    "G": {"range": (2.2, 3.0), "unit": "-", "description": "Specific Gravity"}, 
    "Mc [%]": {"range": (1.3, 85.9), "unit": "%", "description": "Moisture Content"}, 
    "e": {"range": (0.4, 1.8), "unit": "-", "description": "Void Ratio"},
    "γd [kN/m3]": {"range": (8.1, 21.9), "unit": "kN/m³", "description": "Dry Unit Weight"}, 
    "LL [%]": {"range": (19, 255), "unit": "%", "description": "Liquid Limit"}, 
    "PL [%]": {"range": (4, 83), "unit": "%", "description": "Plastic Limit"},
    "PI [%]": {"range": (1, 225), "unit": "%", "description": "Plasticity Index"}, 
    "A": {"range": (0.2, 2.02), "unit": "-", "description": "Activity Index"}, 
    "Cc [%]": {"range": (2.5, 93), "unit": "%", "description": "Clay Content"},
    "Sc [%]": {"range": (12, 69.1), "unit": "%", "description": "Silt Content"}, 
    "OMC [%]": {"range": (11.5, 32), "unit": "%", "description": "Optimum Moisture Content"}, 
    "MDD [kN/m3]": {"range": (12.5, 19.4), "unit": "kN/m³", "description": "Maximum Dry Density"},
}
feature_names = list(inputs_info.keys())

# -----------------------------
# Main Panel with Enhanced Tabs
# -----------------------------
tab_single_pred, tab_batch_existing, tab_batch_unseen, tab_about = st.tabs([
    "🔮 Single Prediction",
    "📊 Batch (Existing)",
    "📈 Batch (Unseen)",
    "ℹ️ About"
])

# --- Enhanced Single Prediction Tab ---
with tab_single_pred:
    st.markdown("# 🔮 Single Soil Sample Prediction")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
                padding: 2rem; border-radius: 15px; border-left: 5px solid #667eea; margin-bottom: 2rem;">
        <h3>🎯 Interactive Prediction Interface</h3>
        <p>Input your soil parameters below to get instant swelling potential predictions with detailed explanations 
        of how each feature contributes to the final result.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for sample selection
    if 'selected_sample_index' not in st.session_state:
        st.session_state.selected_sample_index = None
    if 'sample_values' not in st.session_state:
        st.session_state.sample_values = {key: round(float(np.mean(inputs_info[key]["range"])), 2) for key in feature_names}
        st.session_state.missing_values = {key: False for key in feature_names}

    # Load dataset for sample selection
    try:
        default_data = pd.read_excel("data-with-output.xlsx")
        # Create display labels with row number (1-based index for user)
        sample_labels = [f"Sample {i+1}" for i in range(len(default_data))]
        st.markdown("## 🧪 Test with Sample from Dataset")
        selected_sample_label = st.selectbox(
            "Select a sample from the dataset",
            ["Manual Input"] + sample_labels,
            key="sample_select"
        )
        
        if selected_sample_label != "Manual Input":
            selected_index = sample_labels.index(selected_sample_label)
            st.session_state.selected_sample_index = selected_index
            sample = default_data.loc[selected_index, feature_names]
            st.session_state.sample_values = {key: sample[key] if not pd.isna(sample[key]) else np.nan for key in feature_names}
            st.session_state.missing_values = {key: pd.isna(sample[key]) for key in feature_names}
            st.success(f"Loaded values for {selected_sample_label}")
        else:
            st.session_state.selected_sample_index = None
            st.session_state.sample_values = {key: round(float(np.mean(inputs_info[key]["range"])), 2) for key in feature_names}
            st.session_state.missing_values = {key: False for key in feature_names}
            
    except FileNotFoundError:
        st.warning("Default dataset 'data-with-output.xlsx' not found. Manual input required.")
        default_data = None

    st.markdown("## 📝 Soil Property Inputs")
    
    with st.expander("🔧 **Configure Soil Feature Values**", expanded=True):
        prop_tab1, prop_tab2, prop_tab3 = st.tabs(["🏗️ Physical Properties", "💧 Moisture & Density", "📊 Index Properties"])
        
        inputs = {}
        
        with prop_tab1:
            col1, col2 = st.columns(2)
            physical_props = ["G", "e", "γd [kN/m3]", "MDD [kN/m3]"]
            for i, key in enumerate(physical_props):
                info = inputs_info[key]
                with col1 if i % 2 == 0 else col2:
                    is_missing = st.checkbox(
                        f"Missing: {info['description']}",
                        value=st.session_state.missing_values[key],
                        key=f"missing_{key}"
                    )
                    if is_missing:
                        inputs[key] = np.nan
                        st.markdown(f"<span class='warning-badge'>{key} set to Missing</span>", unsafe_allow_html=True)
                    else:
                        default_value = st.session_state.sample_values[key]
                        if pd.isna(default_value):
                            default_value = round(float(np.mean(info["range"])), 2)
                        inputs[key] = st.number_input(
                            label=f"{info['description']} ({key})",
                            min_value=float(info["range"][0]),
                            max_value=float(info["range"][1]),
                            value=default_value,
                            step=0.1,
                            format="%.2f",
                            key=f"input_{key}",
                            help=f"Range: {info['range'][0]} - {info['range'][1]} {info['unit']}"
                        )
        
        with prop_tab2:
            col1, col2 = st.columns(2)
            moisture_props = ["Mc [%]", "OMC [%]"]
            for i, key in enumerate(moisture_props):
                info = inputs_info[key]
                with col1 if i % 2 == 0 else col2:
                    is_missing = st.checkbox(
                        f"Missing: {info['description']}",
                        value=st.session_state.missing_values[key],
                        key=f"missing_{key}"
                    )
                    if is_missing:
                        inputs[key] = np.nan
                        st.markdown(f"<span class='warning-badge'>{key} set to Missing</span>", unsafe_allow_html=True)
                    else:
                        default_value = st.session_state.sample_values[key]
                        if pd.isna(default_value):
                            default_value = round(float(np.mean(info["range"])), 2)
                        inputs[key] = st.number_input(
                            label=f"{info['description']} ({key})",
                            min_value=float(info["range"][0]),
                            max_value=float(info["range"][1]),
                            value=default_value,
                            step=0.1,
                            format="%.2f",
                            key=f"input_{key}",
                            help=f"Range: {info['range'][0]} - {info['range'][1]} {info['unit']}"
                        )
        
        with prop_tab3:
            col1, col2 = st.columns(2)
            index_props = ["LL [%]", "PL [%]", "PI [%]", "A", "Cc [%]", "Sc [%]"]
            for i, key in enumerate(index_props):
                info = inputs_info[key]
                with col1 if i % 2 == 0 else col2:
                    is_missing = st.checkbox(
                        f"Missing: {info['description']}",
                        value=st.session_state.missing_values[key],
                        key=f"missing_{key}"
                    )
                    if is_missing:
                        inputs[key] = np.nan
                        st.markdown(f"<span class='warning-badge'>{key} set to Missing</span>", unsafe_allow_html=True)
                    else:
                        default_value = st.session_state.sample_values[key]
                        if pd.isna(default_value):
                            default_value = round(float(np.mean(info["range"])), 2)
                        inputs[key] = st.number_input(
                            label=f"{info['description']} ({key})",
                            min_value=float(info["range"][0]),
                            max_value=float(info["range"][1]),
                            value=default_value,
                            step=0.1,
                            format="%.2f",
                            key=f"input_{key}",
                            help=f"Range: {info['range'][0]} - {info['range'][1]} {info['unit']}"
                        )

    def get_swelling_category(value):
        if value >= 35:
            return "Very High", "#6A0DAD", "Extreme swelling potential - Special design considerations required"
        elif value >= 25:
            return "High", "#EA4335", "High swelling potential - Significant structural impact"
        elif value >= 15:
            return "Medium", "#F9AB00", "Moderate swelling potential - Engineering precautions needed"
        else:
            return "Low", "#34A853", "Low swelling potential - Minimal structural impact"

    if st.button("🔍 **Generate Prediction & Analysis**", type="primary", use_container_width=True):
        missing_count = sum(1 for value in inputs.values() if pd.isna(value))
        
        if missing_count >= 3:
            st.error("❌ **Prediction Error:** The model cannot provide a reliable estimate with 3 or more missing features. Please provide more data.")
        else:
            with st.spinner("Processing soil data and generating predictions..."):
                # Ensure input order matches feature_names
                input_df = pd.DataFrame([inputs])[feature_names]
                input_df_unscaled = input_df.copy()
                
                # Processing logic identical to batch
                if scenario_key == "no_imp":
                    X_processed = assets["scalers"]["no_imp"].transform(input_df)
                elif scenario_key == "median":
                    X_imputed = assets["imputers"]["median"].transform(input_df)
                    X_processed = assets["scalers"]["median"].transform(X_imputed)
                    input_df_unscaled = pd.DataFrame(X_imputed, columns=input_df.columns)
                else:  # KNN
                    X_imputed = assets["imputers"]["knn"].transform(input_df)
                    X_processed = assets["scalers"]["knn"].transform(X_imputed)
                    input_df_unscaled = pd.DataFrame(X_imputed, columns=input_df.columns)
                    
                model = assets["models"][scenario_key][model_name]
                prediction = model.predict(X_processed)[0]
                
                # Debug: Compare with batch prediction for the same sample
                if st.session_state.selected_sample_index is not None and default_data is not None:
                    batch_df = default_data[feature_names].iloc[[st.session_state.selected_sample_index]]
                    if scenario_key == "no_imp":
                        batch_processed = assets["scalers"]["no_imp"].transform(batch_df)
                    elif scenario_key == "median":
                        batch_imputed = assets["imputers"]["median"].transform(batch_df)
                        batch_processed = assets["scalers"]["median"].transform(batch_imputed)
                    else:
                        batch_imputed = assets["imputers"]["knn"].transform(batch_df)
                        batch_processed = assets["scalers"]["knn"].transform(batch_imputed)
                    batch_prediction = model.predict(batch_processed)[0]
                    st.info(f"Debug: Single Prediction: {prediction:.2f}% | Batch Prediction (Sample {st.session_state.selected_sample_index + 1}): {batch_prediction:.2f}%")
                
                category, color, description = get_swelling_category(prediction)
        
        st.markdown("---")
        st.markdown("# 📊 Prediction Results")
        
        result_col1, result_col2 = st.columns([2, 3])
        
        with result_col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}15, {color}25);
                        padding: 2rem; border-radius: 20px; border-left: 5px solid {color};
                        text-align: center; margin-bottom: 1rem;">
                <h2 style="color: {color}; margin: 0;">Swelling Potential</h2>
                <h1 style="color: #2E4053; margin: 0.5rem 0; font-size: 3rem;">{prediction:.2f}%</h1>
                <div style="background: {color}; color: white; padding: 0.5rem 1rem; 
                           border-radius: 20px; display: inline-block; font-weight: bold;">
                    {category}
                </div>
                <p style="color: #34495E; margin-top: 1rem; font-style: italic;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with result_col2:
            st.markdown(f"""
            <div style="background: #F8F9FA; padding: 1.5rem; border-radius: 15px; 
                       border: 2px solid #E9ECEF;">
                <h4>🤖 Model Configuration</h4>
                <p><strong>Model:</strong> <span class="info-badge">{model_name}</span></p>
                <p><strong>Imputation:</strong> <span class="info-badge">{scenario}</span></p>
                <p><strong>Features Processed:</strong> {len([k for k, v in inputs.items() if not pd.isna(v)])}/{len(inputs)}</p>
                <p><strong>Missing Values:</strong> {len([k for k, v in inputs.items() if pd.isna(v)])}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("# 💡 Feature Impact Analysis (SHAP)")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(52, 152, 219, 0.1), rgba(41, 128, 185, 0.1));
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;">
            <p><strong>🔍 Understanding the Prediction:</strong> This waterfall plot shows how each soil property 
            contributes to the final prediction. Red bars push the prediction higher, blue bars push it lower, 
            starting from the average prediction (base value). Feature values shown are unscaled.</p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            explainer = assets["explainers"][scenario_key][model_name]
            shap_values = explainer.shap_values(X_processed)
            expected_value = explainer.expected_value

            plt.clf()
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(14, 10))
            fig.patch.set_facecolor('white')
            
            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=expected_value,
                data=input_df_unscaled.iloc[0].values,
                feature_names=feature_names
            )

            shap.waterfall_plot(explanation, show=False, max_display=12)
            
            ax.set_title(f'Feature Contributions to Swelling Potential Prediction\n'
                        f'Model: {model_name} | Scenario: {scenario}', 
                        fontsize=18, fontweight='bold', pad=20)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button(
                label="💾 Download SHAP Analysis",
                data=buf.getvalue(),
                file_name=f"shap_analysis_{model_name.lower()}.png",
                mime="image/png"
            )
            
            plt.close(fig)
            
        except Exception as e:
            st.error(f"Error generating SHAP analysis: {str(e)}")
            st.info("SHAP analysis may not be available for this configuration. The prediction above is still valid.")

# --- Enhanced Batch Prediction (Existing) Tab ---
with tab_batch_existing:
    st.markdown("# 📊 Batch Prediction with Performance Evaluation")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(39, 174, 96, 0.1), rgba(46, 204, 113, 0.1));
                padding: 2rem; border-radius: 15px; border-left: 5px solid #27AE60; margin-bottom: 2rem;">
        <h3>🎯 Model Performance Assessment</h3>
        <p>Upload your dataset with known swelling potential values to evaluate model accuracy and generate 
        comprehensive performance visualizations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        default_data_with_output = pd.read_excel("data-with-output.xlsx")
        st.success("✅ Default validation dataset loaded successfully")
        
        st.markdown("## 📈 Dataset Overview")
        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
        with overview_col1:
            st.metric("Total Samples", len(default_data_with_output))
        with overview_col2:
            st.metric("Features", len(feature_names))
        with overview_col3:
            sp_mean = default_data_with_output["SP [%]"].mean()
            st.metric("Avg. SP", f"{sp_mean:.2f}%")
        with overview_col4:
            missing_pct = default_data_with_output[feature_names].isnull().sum().sum() / (len(default_data_with_output) * len(feature_names)) * 100
            st.metric("Missing Data", f"{missing_pct:.2f}%")
            
    except FileNotFoundError:
        st.error("⚠️ Default data file 'data-with-output.xlsx' not found")
        default_data_with_output = None
    
    if default_data_with_output is not None:
        template_df_existing = default_data_with_output.copy()
        excel_template_existing = to_excel(template_df_existing)
        
        st.markdown("## 📥 Sample Data")
        download_col1, download_col2 = st.columns([1, 2])
        with download_col1:
            st.download_button(
                label="📥 Download Sample Dataset",
                data=excel_template_existing,
                file_name="sample_data_with_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        with download_col2:
            st.info("Use this sample dataset to test the application or as a template for your own data")

        st.markdown("## 📂 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Select your Excel file containing the 12 soil features plus SP [%] column",
            type=["xlsx"],
            key="existing_output_upload",
            help="File must include all 12 input features and a 'SP [%]' column for comparison"
        )
        
        if uploaded_file is None:
            df = template_df_existing.copy()
            st.info("📊 Displaying results using the default validation dataset. Upload your file to analyze custom data.")
        else:
            try:
                df = pd.read_excel(uploaded_file)
                st.success(f"✅ Successfully loaded {len(df)} samples from uploaded file")
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                df = None

        if df is not None:
            required_cols_with_output = feature_names + ["SP [%]"]
            missing_cols = [col for col in required_cols_with_output if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Please ensure your file contains all required columns as shown in the sample data")
            else:
                X_batch = df[feature_names].copy()
                
                missing_info = []
                for col in feature_names:
                    missing_count = X_batch[col].isnull().sum()
                    if missing_count > 0:
                        missing_info.append(f"{col}: {missing_count}")
                
                if missing_info and scenario_key != "no_imp":
                    st.warning(f"🔧 Applying {scenario} to handle missing values in: {', '.join(missing_info)}")
                elif missing_info and scenario_key == "no_imp":
                    st.info(f"🤖 Model handling missing values natively: {', '.join(missing_info)}")

                with st.spinner("Generating predictions and performance analysis..."):
                    if scenario_key == "no_imp":
                        X_batch_processed = assets["scalers"]["no_imp"].transform(X_batch)
                    elif scenario_key == "median":
                        X_batch_imputed = assets["imputers"]["median"].transform(X_batch)
                        X_batch_processed = assets["scalers"]["median"].transform(X_batch_imputed)
                    else:
                        X_batch_imputed = assets["imputers"]["knn"].transform(X_batch)
                        X_batch_processed = assets["scalers"]["knn"].transform(X_batch_imputed)
                        
                    model = assets["models"][scenario_key][model_name]
                    y_pred = model.predict(X_batch_processed)
                
                results_df = df.copy()
                results_df['Predicted_SP_[%]'] = y_pred
                results_df['Absolute_Error'] = np.abs(results_df["SP [%]"] - results_df['Predicted_SP_[%]'])
                results_df['Relative_Error_%'] = (results_df['Absolute_Error'] / results_df["SP [%]"]) * 100
                
                results_df = results_df.round(2)

                y_true = df["SP [%]"]
                R2_val = r2_score(y_true, y_pred)

                y_true_filtered = results_df[results_df["SP [%]"] >= 15]["SP [%]"]
                y_pred_filtered = results_df[results_df["SP [%]"] >= 15]['Predicted_SP_[%]']
                A30_val = compute_A30(y_true_filtered, y_pred_filtered) if len(y_true_filtered) > 0 else 0.0
                
                mae = np.mean(results_df['Absolute_Error'])
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                
                st.markdown("---")
                st.markdown("# 📊 Performance Summary")
                
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                with perf_col1:
                    st.metric("R² Score", f"{R2_val:.2f}", help="Coefficient of determination")
                with perf_col2:
                    st.metric("A30 Index", f"{A30_val:.2f}", help="Proportion within ±30% (SP ≥ 15%)")
                with perf_col3:
                    st.metric("MAE", f"{mae:.2f}%", help="Mean Absolute Error")
                with perf_col4:
                    st.metric("RMSE", f"{rmse:.2f}%", help="Root Mean Square Error")
                
                st.markdown("## 📈 Performance Visualization")
                fig = plot_real_vs_pred(y_true, y_pred, model_name)
                st.pyplot(fig, use_container_width=True)
                
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                st.download_button(
                    label="💾 Download Performance Plot (png)",
                    data=buf.getvalue(),
                    file_name=f"performance_plot_{model_name.lower()}_{scenario_key}.png",
                    mime="image/png"
                )
                
                st.markdown("## 📋 Detailed Results")
                
                error_col1, error_col2 = st.columns(2)
                with error_col1:
                    st.markdown("### Error Statistics")
                    error_stats = pd.DataFrame({
                        'Metric': ['Mean Error', 'Std Error', 'Min Error', 'Max Error', '90th Percentile'],
                        'Value (%)': [
                            f"{np.mean(results_df['Absolute_Error']):.2f}",
                            f"{np.std(results_df['Absolute_Error']):.2f}",
                            f"{np.min(results_df['Absolute_Error']):.2f}",
                            f"{np.max(results_df['Absolute_Error']):.2f}",
                            f"{np.percentile(results_df['Absolute_Error'], 90):.2f}"
                        ]
                    })
                    st.dataframe(error_stats, hide_index=True, use_container_width=True)
                
                with error_col2:
                    st.markdown("### Prediction Categories")
                    pred_categories = pd.cut(results_df['Predicted_SP_[%]'], 
                                           bins=[0, 15, 25, 35, float('inf')], 
                                           labels=['Low (0-15)', 'Medium (15-25)', 'High (25-35)', 'Very High (>35)'])
                    category_counts = pred_categories.value_counts()
                    category_df = pd.DataFrame({
                        'Category': category_counts.index,
                        'Count': category_counts.values,
                        'Percentage': (category_counts.values / len(results_df) * 100).round(2)
                    })
                    st.dataframe(category_df, hide_index=True, use_container_width=True)
                
                st.markdown("### Complete Results Table")
                
                def highlight_errors(val):
                    if isinstance(val, (int, float)):
                        if val < 5:
                            return 'background-color: #d4edda'
                        elif val < 15:
                            return 'background-color: #fff3cd'
                        else:
                            return 'background-color: #f8d7da'
                    return ''
                
                display_df = results_df.copy()
                display_df.index = np.arange(1, len(display_df) + 1)
                numeric_cols = display_df.select_dtypes(include=np.number).columns
                format_dict = {col: '{:.2f}' for col in numeric_cols}
                styled_results = display_df.style.applymap(
                    highlight_errors, subset=['Absolute_Error']
                ).format(format_dict, na_rep='-')
                
                st.dataframe(styled_results, use_container_width=True)
                
                excel_results = to_excel(results_df)
                st.download_button(
                    label="📊 Export Complete Results to Excel",
                    data=excel_results,
                    file_name=f"batch_prediction_results_{model_name.lower()}_{scenario_key}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

# --- Enhanced Batch Prediction (Unseen) Tab ---
with tab_batch_unseen:
    st.markdown("# 📈 Batch Prediction for New Data")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(155, 89, 182, 0.1), rgba(142, 68, 173, 0.1));
                padding: 2rem; border-radius: 15px; border-left: 5px solid #9B59B6; margin-bottom: 2rem;">
        <h3>🔮 Predict Unknown Swelling Potential</h3>
        <p>Upload datasets without known swelling potential values to generate predictions for new soil samples.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        default_data_without_output = pd.read_excel("data-without-output.xlsx")
        st.success("✅ Default unseen dataset loaded successfully")
        
        st.markdown("## 📈 Dataset Overview")
        overview_col1, overview_col2, overview_col3 = st.columns(3)
        with overview_col1:
            st.metric("Total Samples", len(default_data_without_output))
        with overview_col2:
            st.metric("Input Features", len(feature_names))
        with overview_col3:
            missing_pct = default_data_without_output[feature_names].isnull().sum().sum() / (len(default_data_without_output) * len(feature_names)) * 100
            st.metric("Missing Data", f"{missing_pct:.2f}%")
            
    except FileNotFoundError:
        st.error("⚠️ Default unseen data file 'data-without-output.xlsx' not found")
        default_data_without_output = None

    if default_data_without_output is not None:
        template_df_unseen = default_data_without_output.copy()
        excel_template_unseen = to_excel(template_df_unseen)
        
        st.markdown("## 📥 Sample Data")
        download_col1, download_col2 = st.columns([1, 2])
        with download_col1:
            st.download_button(
                label="📥 Download Sample Template",
                data=excel_template_unseen,
                file_name="unseen_data_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        with download_col2:
            st.info("Download this template to see the required format for your unseen data")

        st.markdown("## 📂 Upload New Data")
        uploaded_file = st.file_uploader(
            "Select your Excel file containing the 12 soil features (no SP column needed)",
            type=["xlsx"],
            key="unseen_data_upload",
            help="File should contain only the 12 input features - no swelling potential column required"
        )
        
        if uploaded_file is None:
            df = template_df_unseen.copy()
            st.info("📊 Displaying predictions for default unseen dataset. Upload your file to predict for new samples.")
        else:
            try:
                df = pd.read_excel(uploaded_file)
                st.success(f"✅ Successfully loaded {len(df)} new samples for prediction")
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                df = None

        if df is not None:
            missing_cols = [col for col in feature_names if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing required feature columns: {', '.join(missing_cols)}")
                st.info("Please ensure your file contains all 12 required input features")
            else:
                X_batch = df[feature_names].copy()
                
                missing_analysis = {}
                for col in feature_names:
                    missing_count = X_batch[col].isnull().sum()
                    if missing_count > 0:
                        missing_analysis[col] = missing_count

                if missing_analysis and scenario_key != "no_imp":
                    st.warning(f"🔧 Applying {scenario} for missing values in: {', '.join(missing_analysis.keys())}")
                elif missing_analysis and scenario_key == "no_imp":
                    st.info(f"🤖 Model handling missing values natively in: {', '.join(missing_analysis.keys())}")

                with st.spinner("Generating predictions for unseen data..."):
                    if scenario_key == "no_imp":
                        X_batch_processed = assets["scalers"]["no_imp"].transform(X_batch)
                    elif scenario_key == "median":
                        X_batch_imputed = assets["imputers"]["median"].transform(X_batch)
                        X_batch_processed = assets["scalers"]["median"].transform(X_batch_imputed)
                    else:
                        X_batch_imputed = assets["imputers"]["knn"].transform(X_batch)
                        X_batch_processed = assets["scalers"]["knn"].transform(X_batch_imputed)
                        
                    model = assets["models"][scenario_key][model_name]
                    y_pred = model.predict(X_batch_processed)
                
                results_df = df.copy()
                results_df['Predicted_SP_[%]'] = y_pred
                results_df = results_df.round(2)

                def categorize_prediction(val):
                    if val >= 35: return "Very High"
                    elif val >= 25: return "High"
                    elif val >= 15: return "Medium"
                    else: return "Low"
                
                results_df['Risk_Category'] = results_df['Predicted_SP_[%]'].apply(categorize_prediction)
                
                st.markdown("---")
                st.markdown("# 📊 Prediction Summary")
                
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                with summary_col1:
                    st.metric("Samples Processed", len(results_df))
                with summary_col2:
                    avg_prediction = np.mean(y_pred)
                    st.metric("Average SP", f"{avg_prediction:.2f}%")
                with summary_col3:
                    std_prediction = np.std(y_pred)
                    st.metric("Standard Deviation", f"{std_prediction:.2f}%")
                with summary_col4:
                    max_prediction = np.max(y_pred)
                    st.metric("Maximum SP", f"{max_prediction:.2f}%")

                st.markdown("## 📈 Risk Distribution")
                category_counts = results_df['Risk_Category'].value_counts()
                
                cat_cols = st.columns(4)
                colors = {'Low': '#34A853', 'Medium': '#F9AB00', 'High': '#EA4335', 'Very High': '#6A0DAD'}
                
                for i, (category, color) in enumerate(colors.items()):
                    count = category_counts.get(category, 0)
                    percentage = (count / len(results_df)) * 100
                    with cat_cols[i]:
                        st.markdown(f"""
                        <div style="background: {color}15; padding: 1rem; border-radius: 10px; 
                                   border-left: 4px solid {color}; text-align: center;">
                            <h4 style="color: {color}; margin: 0;">{category}</h4>
                            <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">{count}</p>
                            <p style="margin: 0; color: #666;">({percentage:.2f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("## 📋 Detailed Predictions")
                
                display_df = results_df.copy()
                display_df.index = np.arange(1, len(display_df) + 1)
                
                def highlight_categories(val):
                    color_map = {'Very High': '#6A0DAD', 'High': '#EA4335', 'Medium': '#F9AB00', 'Low': '#34A853'}
                    return f'background-color: {color_map.get(val, "#ffffff")}22'
                
                numeric_cols = display_df.select_dtypes(include=np.number).columns
                format_dict = {col: '{:.2f}' for col in numeric_cols}
                styled_df = display_df.style.applymap(
                    highlight_categories, subset=['Risk_Category']
                ).format(format_dict, na_rep='-')
                
                st.dataframe(styled_df, use_container_width=True)

                excel_results = to_excel(results_df)
                st.download_button(
                    label="📊 Export Predictions to Excel",
                    data=excel_results,
                    file_name=f"unseen_predictions_{model_name.lower()}_{scenario_key}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

# --- Enhanced About Tab for Academic Paper Quality ---
with tab_about:
    st.markdown("# ℹ️ About the web app")
    
    st.markdown("""
    ### Introduction
    This web app leverages advanced machine learning techniques to predict soil swelling potential, a critical factor in geotechnical engineering. It is designed for researchers, engineers, and students to perform rapid assessments with high accuracy.
    
    ### Methodology
    - **Models**: XGBoost, LightGBM, CatBoost with hyperparameter tuning via cross-validation.
    - **Preprocessing**: Handling missing values using native model capabilities, median, or KNN imputation; standardized scaling.
    - **Interpretability**: SHAP values for feature contribution analysis.
    - **Dataset**: Compiled from comprehensive soil testing data (Onyekpe, 2021), consisting of 395 samples with 12 geotechnical features.
    
    **Key Metrics**:
    - R² for model fit.
    - A30-index for accuracy in high-swelling soils (SP ≥ 15%).
    - MAE and RMSE for error quantification.
    
    ### Research Team
    - PhD Candidate Khaled Hamdaoui
    - PhD. Billal Sari Ahmed
    - PhD. Mohamed Elhebib Guellil
    - Prof. Mohamed Ghrici
    
    **Affiliation**: Geomaterials Laboratory, Civil Engineering Dept., Hassiba Benbouali University, Chlef, Algeria
    
    **Contact**: k.hamdaoui92@univ-chlef.dz
    
    ### Technical Stack
    - Frontend: Streamlit for interactive UI.
    - Backend: Scikit-learn for preprocessing, XGBoost/LightGBM/CatBoost for modeling, SHAP for explanations.
    - Data Handling: Pandas, NumPy.
    - Visualization: Matplotlib.
    
    ### Acknowledgments
    Based on dataset from Onyekpe (2021). Thanks to open-source contributors for libraries used.
    
    ### Citation
    If using this web app in academic work, please cite:
    
    Onyekpe, U., 2021. Data on one-dimensional vertical free swelling potential of soils and related soil properties. Data in Brief, 39, p.107608.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #7F8C8D;">
    <p>🏛️ Developed at Hassiba Benbouali University | 🔬 Geomaterials Laboratory</p>
    <p>📧 Contact: k.hamdaoui92@univ-chlef.dz | 🌟 Version 2.3 - Academic Enhanced Edition</p>
</div>
""", unsafe_allow_html=True)