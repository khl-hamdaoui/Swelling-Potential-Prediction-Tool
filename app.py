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

# Enhanced CSS for Chrome optimization and modern UI
st.markdown(
    """
    <style>
    /* Import Google Fonts for better typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for consistent theming */
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
    
    /* Global font and base styling */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 16px;
        line-height: 1.6;
        color: #2C3E50;
    }
    
    /* Main container styling */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
        margin: 0 auto;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }
    
    /* Content wrapper for better contrast */
    .main .block-container > div {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: var(--shadow-hover);
        margin-bottom: 2rem;
    }
    
    /* Enhanced header styling */
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
    
    /* Enhanced sidebar styling */
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
    
    /* Enhanced button styling */
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
    
    /* Enhanced download button styling */
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
    
    /* Enhanced metric styling */
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
    
    /* Enhanced alert styling */
    .stAlert {
        border-radius: 12px;
        border: none;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    
    /* Enhanced input styling */
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
    
    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--light-bg);
        padding: 0.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.75rem 1.2rem; /* Reduced horizontal padding */
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-color), #2980B9);
        color: white;
    }
    
    /* Enhanced dataframe styling */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: var(--shadow);
        margin: 1rem 0;
    }
    
    /* Enhanced expander styling */
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
    
    /* Enhanced file uploader styling */
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
    
    /* Enhanced checkbox and radio styling */
    .stCheckbox > label,
    .stRadio > label {
        font-weight: 500;
        color: var(--secondary-color);
    }
    
    /* Responsive design for mobile */
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
    
    /* Custom success badge styling */
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
    
    /* Custom warning badge styling */
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
    
    /* Custom danger badge styling */
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
    
    /* Custom info badge styling */
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
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Smooth transitions for all interactive elements */
    * {
        transition: all 0.3s ease;
    }
    
    /* Custom scrollbar for Chrome */
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
    """Loads all pre-trained models and preprocessors to avoid reloading on each interaction."""
    try:
        # Scalers and Imputers
        scaler_noimp = joblib.load("models/scaler.pkl")
        median_imputer = joblib.load("models/median_imputer.pkl")
        scaler_median = joblib.load("models/scaler_median.pkl")
        knn_imputer = joblib.load("models/knn_imputer.pkl")
        scaler_knn = joblib.load("models/scaler_knn.pkl")

        # Models
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

        # SHAP Explainers
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

# Load all assets with loading indicator
with st.spinner('Loading models and preprocessors...'):
    assets = load_models_and_preprocessors()

# -----------------------------
# Helper Functions
# -----------------------------
def compute_A30(y_true, y_pred):
    """Compute A30-index: proportion of predictions within ±30% of observed values."""
    evaluator = RegressionMetric(y_true.tolist(), y_pred.tolist())
    return evaluator.A30()

def plot_real_vs_pred(y_true, y_pred, model_name):
    """Generate and return a scatter plot of true vs. predicted values with color-coded ranges."""
    
    swelling_potential_categories = {
        "Low (0-15)": "#34A853",
        "Medium (15-25)": "#F9AB00",
        "High (25-35)": "#EA4335",
        "Very High (>35)": "#6A0DAD"
    }

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set fixed axis limits
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
    
    # Plot the major axes and tolerance lines
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

    # Add a black border to the legend
    ax.legend(handles=handles, title=f"Predicted Swelling Potential\n({model_name})", loc="upper left", fontsize=14, title_fontsize=14, frameon=True, edgecolor='black')
    
     # Grid settings
    ax.grid(True, which='major', linestyle='-', linewidth=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5)
    ax.minorticks_on()

    return fig

def to_excel(df):
    """Converts a DataFrame to an Excel file in memory."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Predictions']
        # Add formatting
        header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top'})
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
    processed_data = output.getvalue()
    return processed_data

# -----------------------------
# Sidebar: Enhanced Model Selection
# -----------------------------
st.sidebar.markdown("## 🤖 Model Configuration")

# Visual model selection using custom styling
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

# Enhanced imputation strategy selection
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

# Configuration summary
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current Configuration")
st.sidebar.markdown(f"""
<div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;'>
    <p><strong>Model:</strong> {model_name}</p>
    <p><strong>Imputation:</strong> {scenario}</p>
</div>
""", unsafe_allow_html=True)

# Map scenario to correct model and preprocessor keys
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
tab_home, tab_models, tab_single_pred, tab_batch_existing, tab_batch_unseen, tab_dataset, tab_about = st.tabs([
    "🏠 Home",
    "🧠 Models", 
    "🔮 Single Prediction",
    "📊 Batch (Existing)",
    "📈 Batch (Unseen)",
    "📚 Dataset Info",
    "ℹ️ About"
])

# --- Home Tab ---
with tab_home:
    # Enhanced welcome section with better visual hierarchy
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2 style="color: #2E4053; margin-bottom: 1rem;">Welcome to Advanced Soil Analysis</h2>
            <p style="font-size: 1.2rem; line-height: 1.8; color: #34495E;">
                Leverage cutting-edge machine learning to predict soil swelling potential with unprecedented accuracy and speed.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature cards with enhanced styling
    st.markdown("### 🎯 Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; text-align: center;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h4>⚡ Rapid Prediction</h4>
            <p>Get instant swelling potential estimates in seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; text-align: center;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h4>🧠 AI-Powered</h4>
            <p>State-of-the-art gradient boosting algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; text-align: center;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h4>📊 Comprehensive</h4>
            <p>Handle missing data and provide detailed insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced problem description
    st.markdown("## 🚧 The Challenge of Expansive Soils")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Expansive soils** pose a significant threat to civil infrastructure worldwide. These clay-rich soils 
        undergo dramatic volume changes with moisture variations, causing billions of dollars in damage annually 
        to foundations, pavements, and structures.
        
        **Traditional assessment methods** are:
        - ⏱️ Time-consuming (days to weeks)
        - 💰 Expensive (laboratory equipment and expertise)
        - 🔬 Limited in scope (single-point testing)
        
        **Our solution provides**:
        - ⚡ Instant predictions (seconds)
        - 💡 Transparent AI explanations
        - 📊 Batch processing capabilities
        - 🎯 High accuracy (validated models)
        """)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
                    padding: 2rem; border-radius: 20px; text-align: center;">
            <h3 style="color: #2E4053; margin-bottom: 1rem;">Quick Stats</h3>
            <div style="color: #34495E; font-weight: 600;">
                <p>🗂️ 395 Training Samples</p>
                <p>🌍 Multi-continental Data</p>
                <p>🔬 12 Input Features</p>
                <p>🎯 3 ML Models</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced navigation guide
    st.markdown("## 🧭 Application Guide")
    
    nav_items = [
        ("🧠 Models", "Explore the machine learning models and preprocessing techniques"),
        ("🔮 Single Prediction", "Predict swelling potential for individual soil samples with SHAP explanations"),
        ("📊 Batch (Existing)", "Evaluate model performance against known results"),
        ("📈 Batch (Unseen)", "Generate predictions for new, unlabeled data"),
        ("📚 Dataset Info", "Comprehensive dataset statistics and visualizations"),
        ("ℹ️ About", "Project details, authors, and technical specifications")
    ]
    
    for i, (title, desc) in enumerate(nav_items):
        with st.expander(f"**{title}**", expanded=False):
            st.markdown(f"📝 {desc}")
    
    # Configuration guide
    st.markdown("### ⚙️ Sidebar Configuration")
    st.info("""
    **Imputation Strategies:**
    - **Without Imputation**: For models that handle missing values natively
    - **Median Imputation**: Replace missing values with feature medians (robust to outliers)
    - **KNN Imputation**: Estimate missing values using 5 nearest neighbors (captures relationships)
    
    **Model Selection:**
    - **XGBoost**: Extreme gradient boosting with regularization
    - **LightGBM**: Fast gradient boosting with GOSS and EFB optimizations
    - **CatBoost**: Categorical boosting with ordered boosting and symmetric trees
    """)

# --- Enhanced Models & Methodology Tab ---
with tab_models:
    st.markdown("# 🧠 Machine Learning Models and Methodological Framework")
    
    # Model comparison cards
    st.markdown("## 🏆 Model Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem; border-radius: 15px; color: white; height: 300px;">
            <h3>🚀 XGBoost</h3>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <p><strong>Strengths:</strong></p>
            <ul>
                <li>High performance</li>
                <li>Built-in regularization</li>
                <li>Parallel processing</li>
                <li>Feature importance</li>
            </ul>
            <p><strong>Best for:</strong> Structured data with mixed feature types</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 2rem; border-radius: 15px; color: white; height: 300px;">
            <h3>⚡ LightGBM</h3>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <p><strong>Strengths:</strong></p>
            <ul>
                <li>Fast training speed</li>
                <li>Memory efficient</li>
                <li>GOSS sampling</li>
                <li>EFB bundling</li>
            </ul>
            <p><strong>Best for:</strong> Large datasets requiring speed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    padding: 2rem; border-radius: 15px; color: white; height: 300px;">
            <h3>🎯 CatBoost</h3>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <p><strong>Strengths:</strong></p>
            <ul>
                <li>Handles categoricals</li>
                <li>Overfitting resistant</li>
                <li>Ordered boosting</li>
                <li>Symmetric trees</li>
            </ul>
            <p><strong>Best for:</strong> Categorical features and robustness</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Methodology workflow
    st.markdown("## 🔄 Methodology Workflow")
    
    workflow_steps = [
        ("1️⃣", "Data Preprocessing", "Handle missing values using three strategies"),
        ("2️⃣", "Feature Scaling", "Standardize features to zero mean and unit variance"),
        ("3️⃣", "Model Training", "Train three models with each imputation strategy"),
        ("4️⃣", "Model Selection", "Evaluate performance using cross-validation"),
        ("5️⃣", "Prediction & Explanation", "Generate predictions with SHAP interpretability")
    ]
    
    for emoji, title, desc in workflow_steps:
        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"<h2 style='text-align: center; margin: 0;'>{emoji}</h2>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**{title}**")
                st.markdown(desc)
        st.markdown("---")
    
    # Technical details
    st.markdown("## 🔬 Technical Implementation")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        ### 📊 Data Processing Pipeline
        - **Missing Value Detection**: Automated identification of NaN values
        - **Imputation Methods**: Median, KNN (k=5), or native handling
        - **Feature Scaling**: StandardScaler transformation
        - **Validation**: Stratified cross-validation
        """)
    
    with tech_col2:
        st.markdown("""
        ### 🎯 Model Interpretability
        - **SHAP Values**: Feature contribution analysis
        - **Waterfall Plots**: Individual prediction explanations
        - **Global Importance**: Overall feature ranking
        - **Local Explanations**: Sample-specific insights
        """)

# --- Enhanced Single Prediction Tab ---
with tab_single_pred:
    st.markdown("# 🔮 Single Soil Sample Prediction")
    
    # Enhanced description with visual elements
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
                padding: 2rem; border-radius: 15px; border-left: 5px solid #667eea; margin-bottom: 2rem;">
        <h3>🎯 Interactive Prediction Interface</h3>
        <p>Input your soil parameters below to get instant swelling potential predictions with detailed explanations 
        of how each feature contributes to the final result.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced input section with better organization
    st.markdown("## 📝 Soil Property Inputs")
    
    with st.expander("🔧 **Configure Soil Feature Values**", expanded=True):
        # Create tabs for different property groups
        prop_tab1, prop_tab2, prop_tab3 = st.tabs(["🏗️ Physical Properties", "💧 Moisture & Density", "📊 Index Properties"])
        
        inputs = {}
        
        with prop_tab1:
            col1, col2 = st.columns(2)
            physical_props = ["G", "e", "γd [kN/m3]", "MDD [kN/m3]"]
            for i, key in enumerate(physical_props):
                info = inputs_info[key]
                with col1 if i % 2 == 0 else col2:
                    is_missing = st.checkbox(f"Missing: {info['description']}", key=f"missing_{key}")
                    if is_missing:
                        inputs[key] = np.nan
                        st.markdown(f"<span class='warning-badge'>{key} set to Missing</span>", unsafe_allow_html=True)
                    else:
                        inputs[key] = st.number_input(
                            label=f"{info['description']} ({key})",
                            min_value=float(info["range"][0]),
                            max_value=float(info["range"][1]),
                            value=round(float(np.mean(info["range"])), 2),
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
                    is_missing = st.checkbox(f"Missing: {info['description']}", key=f"missing_{key}")
                    if is_missing:
                        inputs[key] = np.nan
                        st.markdown(f"<span class='warning-badge'>{key} set to Missing</span>", unsafe_allow_html=True)
                    else:
                        inputs[key] = st.number_input(
                            label=f"{info['description']} ({key})",
                            min_value=float(info["range"][0]),
                            max_value=float(info["range"][1]),
                            value=round(float(np.mean(info["range"])), 2),
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
                    is_missing = st.checkbox(f"Missing: {info['description']}", key=f"missing_{key}")
                    if is_missing:
                        inputs[key] = np.nan
                        st.markdown(f"<span class='warning-badge'>{key} set to Missing</span>", unsafe_allow_html=True)
                    else:
                        inputs[key] = st.number_input(
                            label=f"{info['description']} ({key})",
                            min_value=float(info["range"][0]),
                            max_value=float(info["range"][1]),
                            value=round(float(np.mean(info["range"])), 2),
                            step=0.1,
                            format="%.2f",
                            key=f"input_{key}",
                            help=f"Range: {info['range'][0]} - {info['range'][1]} {info['unit']}"
                        )

    def get_swelling_category(value):
        """Enhanced category function with colors and descriptions"""
        if value >= 35:
            return "Very High", "#6A0DAD", "Extreme swelling potential - Special design considerations required"
        elif value >= 25:
            return "High", "#EA4335", "High swelling potential - Significant structural impact"
        elif value >= 15:
            return "Medium", "#F9AB00", "Moderate swelling potential - Engineering precautions needed"
        else:
            return "Low", "#34A853", "Low swelling potential - Minimal structural impact"

    # Enhanced prediction button
    if st.button("🔍 **Generate Prediction & Analysis**", type="primary", use_container_width=True):
        # Validate the number of missing inputs before proceeding
        missing_count = sum(1 for value in inputs.values() if pd.isna(value))
        
        if missing_count >= 3:
            st.error("❌ **Prediction Error:** The model cannot provide a reliable estimate with 3 or more missing features. Please provide more data.")
        else:
            with st.spinner("Processing soil data and generating predictions..."):
                input_df = pd.DataFrame([inputs])
                
                # Processing logic
                if scenario_key == "no_imp":
                    X_processed = assets["scalers"]["no_imp"].transform(input_df)
                elif scenario_key == "median":
                    X_imputed = assets["imputers"]["median"].transform(input_df)
                    X_processed = assets["scalers"]["median"].transform(X_imputed)
                else:  # KNN
                    X_imputed = assets["imputers"]["knn"].transform(input_df)
                    X_processed = assets["scalers"]["knn"].transform(X_imputed)
                    
                model = assets["models"][scenario_key][model_name]
                prediction = model.predict(X_processed)[0]
                
                category, color, description = get_swelling_category(prediction)
        
        # Enhanced results display
        st.markdown("---")
        st.markdown("# 📊 Prediction Results")
        
        # Main results with enhanced styling
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
            # Model info and settings
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
        
        # SHAP Analysis with enhanced presentation
        st.markdown("---")
        st.markdown("# 💡 Feature Impact Analysis (SHAP)")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(52, 152, 219, 0.1), rgba(41, 128, 185, 0.1));
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;">
            <p><strong>🔍 Understanding the Prediction:</strong> This waterfall plot shows how each soil property 
            contributed to the final prediction. Red bars push the prediction higher, blue bars push it lower, 
            starting from the average prediction (base value) to arrive at your specific result.</p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            explainer = assets["explainers"][scenario_key][model_name]
            shap_values = explainer.shap_values(X_processed)
            expected_value = explainer.expected_value

            # Enhanced SHAP plot styling
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(14, 10))
            fig.patch.set_facecolor('white')
            
            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=expected_value,
                data=X_processed[0],
                feature_names=feature_names
            )

            shap.waterfall_plot(explanation, show=False, max_display=12)
            
            # Enhance plot appearance
            ax.set_title(f'Feature Contributions to Swelling Potential Prediction\n'
                        f'Model: {model_name} | Scenario: {scenario}', 
                        fontsize=18, fontweight='bold', pad=20)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            # Add download option for SHAP plot
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button(
                label="💾 Download SHAP Analysis",
                data=buf.getvalue(),
                file_name=f"shap_analysis_{model_name.lower()}.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"Error generating SHAP analysis: {str(e)}")
            st.info("SHAP analysis may not be available for all model configurations. The prediction result above is still valid.")

# Continue with other tabs using similar enhanced styling patterns...

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
    
    # Load and display default data
    try:
        default_data_with_output = pd.read_excel("data-with-output.xlsx")
        st.success("✅ Default validation dataset loaded successfully")
        
        # Data overview
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
        
        # Enhanced download section
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

        # Enhanced file upload
        st.markdown("## 📂 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Select your Excel file containing the 12 soil features plus SP [%] column",
            type=["xlsx"],
            key="existing_output_upload",
            help="File must include all 12 input features and a 'SP [%]' column for comparison"
        )
        
        # Data selection logic
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
            # Validation
            required_cols_with_output = feature_names + ["SP [%]"]
            missing_cols = [col for col in required_cols_with_output if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Please ensure your file contains all required columns as shown in the sample data")
            else:
                # Data processing and prediction
                X_batch = df[feature_names].copy()
                
                # Missing data analysis
                missing_info = []
                for col in feature_names:
                    missing_count = X_batch[col].isnull().sum()
                    if missing_count > 0:
                        missing_info.append(f"{col}: {missing_count}")
                
                if missing_info and scenario_key != "no_imp":
                    st.warning(f"🔧 Applying {scenario} to handle missing values in: {', '.join(missing_info)}")
                elif missing_info and scenario_key == "no_imp":
                    st.info(f"🤖 Model handling missing values natively: {', '.join(missing_info)}")

                # Model processing
                with st.spinner("Generating predictions and performance analysis..."):
                    if scenario_key == "no_imp":
                        X_batch_processed = assets["scalers"]["no_imp"].transform(X_batch)
                    elif scenario_key == "median":
                        X_batch_imputed = assets["imputers"]["median"].transform(X_batch)
                        X_batch_processed = assets["scalers"]["median"].transform(X_batch_imputed)
                    else:  # KNN
                        X_batch_imputed = assets["imputers"]["knn"].transform(X_batch)
                        X_batch_processed = assets["scalers"]["knn"].transform(X_batch_imputed)
                        
                    model = assets["models"][scenario_key][model_name]
                    y_pred = model.predict(X_batch_processed)
                
                # Results
                results_df = df.copy()
                results_df['Predicted_SP_[%]'] = y_pred
                results_df['Absolute_Error'] = np.abs(results_df["SP [%]"] - results_df['Predicted_SP_[%]'])
                results_df['Relative_Error_%'] = (results_df['Absolute_Error'] / results_df["SP [%]"]) * 100
                
                # Round all numerical columns to 2 decimal places for consistent display
                results_df = results_df.round(2)

                # Performance metrics
                y_true = df["SP [%]"]
                R2_val = r2_score(y_true, y_pred)

                y_true_filtered = results_df[results_df["SP [%]"] >= 15]["SP [%]"]
                y_pred_filtered = results_df[results_df["SP [%]"] >= 15]['Predicted_SP_[%]']
                A30_val = compute_A30(y_true_filtered, y_pred_filtered) if len(y_true_filtered) > 0 else 0.0
                
                mae = np.mean(results_df['Absolute_Error'])
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                
                # Performance summary
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
                
                # Enhanced performance plot
                st.markdown("## 📈 Performance Visualization")
                fig = plot_real_vs_pred(y_true, y_pred, model_name)
                st.pyplot(fig, use_container_width=True)
                
                # Plot download
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                st.download_button(
                    label="💾 Download Performance Plot (png)",
                    data=buf.getvalue(),
                    file_name=f"performance_plot_{model_name.lower()}_{scenario_key}.png",
                    mime="image/svg+xml"
                )
                
                # Results table with enhanced display
                st.markdown("## 📋 Detailed Results")
                
                # Add error analysis
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
                
                # Full results table
                st.markdown("### Complete Results Table")
                
                # Color-code results based on error
                def highlight_errors(val):
                    # Check if the value is numeric before comparison
                    if isinstance(val, (int, float)):
                        if val < 5:
                            return 'background-color: #d4edda'  # Green for good predictions
                        elif val < 15:
                            return 'background-color: #fff3cd'  # Yellow for moderate errors
                        else:
                            return 'background-color: #f8d7da'  # Red for high errors
                    return '' # Return empty string for non-numeric cells
                
                # Create a copy for display and re-index to start from 1
                display_df = results_df.copy()
                display_df.index = np.arange(1, len(display_df) + 1)

                # Define a format dictionary for all numeric columns to ensure two decimal places
                numeric_cols = display_df.select_dtypes(include=np.number).columns
                format_dict = {col: '{:.2f}' for col in numeric_cols}

                # Display results with styling, 1-based index, and two-digit formatting
                styled_results = display_df.style.applymap(
                    highlight_errors, subset=['Absolute_Error']
                ).format(format_dict, na_rep='-')
                
                st.dataframe(styled_results, use_container_width=True)
                
                # Export options
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
        <p>Upload datasets without known swelling potential values to generate predictions for new soil samples. 
        Perfect for prospective analysis and real-world applications.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load default unseen data
    try:
        default_data_without_output = pd.read_excel("data-without-output.xlsx")
        st.success("✅ Default unseen dataset loaded successfully")
        
        # Data overview for unseen data
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
        
        # Enhanced download and upload sections
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
        
        # Data processing
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
            # Validation
            missing_cols = [col for col in feature_names if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing required feature columns: {', '.join(missing_cols)}")
                st.info("Please ensure your file contains all 12 required input features")
            else:
                X_batch = df[feature_names].copy()
                
                # Missing data info
                missing_analysis = {}
                for col in feature_names:
                    missing_count = X_batch[col].isnull().sum()
                    if missing_count > 0:
                        missing_analysis[col] = missing_count

                if missing_analysis and scenario_key != "no_imp":
                    st.warning(f"🔧 Applying {scenario} for missing values in: {', '.join(missing_analysis.keys())}")
                elif missing_analysis and scenario_key == "no_imp":
                    st.info(f"🤖 Model handling missing values natively in: {', '.join(missing_analysis.keys())}")

                # Generate predictions
                with st.spinner("Generating predictions for unseen data..."):
                    if scenario_key == "no_imp":
                        X_batch_processed = assets["scalers"]["no_imp"].transform(X_batch)
                    elif scenario_key == "median":
                        X_batch_imputed = assets["imputers"]["median"].transform(X_batch)
                        X_batch_processed = assets["scalers"]["median"].transform(X_batch_imputed)
                    else:  # KNN
                        X_batch_imputed = assets["imputers"]["knn"].transform(X_batch)
                        X_batch_processed = assets["scalers"]["knn"].transform(X_batch_imputed)
                        
                    model = assets["models"][scenario_key][model_name]
                    y_pred = model.predict(X_batch_processed)
                
                # Prepare results
                results_df = df.copy()
                results_df['Predicted_SP_[%]'] = y_pred
            
                results_df = np.round(results_df, 2)

                # Add categories
                def categorize_prediction(val):
                    if val >= 35: return "Very High"
                    elif val >= 25: return "High"
                    elif val >= 15: return "Medium"
                    else: return "Low"
                
                results_df['Risk_Category'] = results_df['Predicted_SP_[%]'].apply(categorize_prediction)
                
               

                # Summary statistics
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

                # Category distribution
                st.markdown("## 📈 Risk Distribution")
                category_counts = results_df['Risk_Category'].value_counts()
                
                # Create columns for category display
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

                # Results table
                st.markdown("## 📋 Detailed Predictions")
                
                # Create a copy for display, sorted by index (default), and re-index to start from 1
                display_df = results_df.copy()
                display_df.index = np.arange(1, len(display_df) + 1)
                
                # Style the dataframe
                def highlight_categories(val):
                    color_map = {'Very High': '#6A0DAD', 'High': '#EA4335', 'Medium': '#F9AB00', 'Low': '#34A853'}
                    return f'background-color: {color_map.get(val, "#ffffff")}22'
                
                # Define a format dictionary for all numeric columns to ensure two decimal places
                numeric_cols = display_df.select_dtypes(include=np.number).columns
                format_dict = {col: '{:.2f}' for col in numeric_cols}

                # Display results with styling, 1-based index, and two-digit formatting
                styled_df = display_df.style.applymap(
                    highlight_categories, subset=['Risk_Category']
                ).format(format_dict, na_rep='-')
                
                st.dataframe(styled_df, use_container_width=True)

                # Export functionality
                excel_results = to_excel(results_df)
                st.download_button(
                    label="📊 Export Predictions to Excel",
                    data=excel_results,
                    file_name=f"unseen_predictions_{model_name.lower()}_{scenario_key}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

# --- Enhanced Dataset Information Tab ---
with tab_dataset:
    st.markdown("# 📚 Comprehensive Dataset Information")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                padding: 3rem; border-radius: 20px; color: white; text-align: center; margin-bottom: 2rem;">
        <h2 style="margin: 0; color: white;">Advanced Soil Swelling Prediction</h2>
        <p style="font-size: 1.2rem; margin: 1rem 0 0 0; opacity: 0.9;">
            Leveraging Machine Learning for Geotechnical Risk Assessment
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Authors section
    st.markdown("## 👥 Research Team")
    
    author_col1, author_col2 = st.columns(2)
    
    with author_col1:
        st.markdown("""
        <div style="background: #F8F9FA; padding: 2rem; border-radius: 15px; border-left: 5px solid #3498DB;">
            <h4>Primary Investigators</h4>
            <ul style="list-style: none; padding: 0;">
                <li><strong>🎓 Khaled Hamdaoui</strong><br>PhD Candidate in Geotechnical Engineering</li>
                <li><strong>👨‍🏫 Prof. Billal Sari Ahmed</strong><br>Professor of Civil Engineering</li>
                <li><strong>👨‍🏫 Assoc. Prof. Mohamed Elhebib Guellil</strong><br>Associate Professor</li>
                <li><strong>👨‍🏫 Prof. Mohamed Ghrici</strong><br>Professor of Geotechnical Engineering</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with author_col2:
        st.markdown("""
        <div style="background: #F8F9FA; padding: 2rem; border-radius: 15px; border-left: 5px solid #E74C3C;">
            <h4>Institutional Affiliation</h4>
            <p><strong>🏛️ Geomaterials Laboratory</strong><br>
            Civil Engineering Department<br>
            Hassiba Benbouali University<br>
            Chlef, Algeria</p>
            
            📧 Contact:
            k.hamdaoui92@univ-chlef.dz
        </div>
        """, unsafe_allow_html=True)

    # Technical implementation
    st.markdown("## 🛠️ Technical Stack")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        **Frontend & Interface**
        - Streamlit Framework
        - Custom CSS Styling  
        - Responsive Design
        - Interactive Components
        """)
    
    with tech_col2:
        st.markdown("""
        **Machine Learning**
        - XGBoost, LightGBM, CatBoost
        - Scikit-learn Pipeline
        - SHAP Interpretability
        - Model Persistence (Joblib)
        """)
    
    with tech_col3:
        st.markdown("""
        **Data & Visualization**
        - Pandas Data Processing
        - Matplotlib Plotting
        - Excel I/O Operations
        - Statistical Analysis
        """)

    # Project objectives
    st.markdown("## 🎯 Project Objectives")
    
    objectives = [
        ("⚡", "Rapid Assessment", "Provide instant swelling potential predictions to accelerate geotechnical analysis"),
        ("🔍", "Model Transparency", "Implement SHAP explainability for trustworthy AI-driven decisions"),
        ("🛡️", "Data Robustness", "Handle missing data through multiple imputation strategies"),
        ("📊", "Performance Validation", "Compare state-of-the-art ML models across different scenarios"),
        ("🌍", "Practical Application", "Enable real-world deployment for geotechnical risk assessment")
    ]
    
    for emoji, title, description in objectives:
        st.markdown(f"""
        <div style="background: #F8F9FA; padding: 1.5rem; border-radius: 12px; 
                   margin-bottom: 1rem; border-left: 4px solid #3498DB;">
            <h4 style="margin: 0; color: #2E4053;">{emoji} {title}</h4>
            <p style="margin: 0.5rem 0 0 0; color: #34495E;">{description}</p>
        </div>
        """, unsafe_allow_html=True)

    # Acknowledgments
    st.markdown("---")
    st.markdown("## 🙏 Acknowledgments")
    
    st.markdown("""
    The development of this tool was made possible through the collaborative efforts of:
    
    - **Geomaterials Laboratory** at Hassiba Benbouali University for providing research infrastructure
    - **Original dataset contributors** (Eyo & Onyekpe, 2021) for making valuable geotechnical data openly available  
    - **Open-source community** for developing the machine learning libraries and frameworks utilized
    - **Academic reviewers and colleagues** for valuable feedback during development
    
    This project exemplifies the power of open science and collaborative research in advancing geotechnical engineering solutions.
    """)

    # Technical specifications
    st.markdown("## ⚙️ System Requirements & Performance")
    
    spec_col1, spec_col2 = st.columns(2)
    
    with spec_col1:
        st.markdown("""
        **Recommended Browser**
        - Google Chrome (latest)
        - Mozilla Firefox (latest)
        - Safari (latest)
        - Microsoft Edge (latest)
        """)
    
    with spec_col2:
        st.markdown("""
        **Performance Characteristics**
        - Single prediction: < 1 second
        - Batch processing: ~100 samples/second
        - Model loading: < 5 seconds
        - Memory usage: < 500MB
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #7F8C8D;">
    <p>🏛️ Developed at Hassiba Benbouali University | 🔬 Geomaterials Laboratory</p>
    <p>📧 Contact: k.hamdaoui@univ-chlef.dz | 🌟 Version 2.0 - Enhanced Chrome Experience</p>
</div>
""", unsafe_allow_html=True)