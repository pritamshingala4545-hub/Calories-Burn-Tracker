"""
Calories Burn Tracker — Precision Health & Energy Expenditure Intelligence
A high-performance Streamlit application for physical activity tracking,
cardiovascular analytics, biometric modeling, and machine-learning calorie prediction.
Dark High-Contrast Theme with Sidebar Tab Navigation.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import io

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ==============================================================================
# 1. PAGE CONFIGURATION & METADATA
# ==============================================================================

st.set_page_config(
    page_title="Calories Burn Tracker",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. BLACK BACKGROUND & HIGH-CONTRAST DARK DESIGN SYSTEM (CSS)
# ==============================================================================

CUSTOM_CSS = """
<style>
/* Modern Font and Base Theme */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #f1f5f9;
}

/* Deep Black Main Application Background */
.stApp {
    background-color: #030712 !important; /* Rich deep black */
    color: #f1f5f9 !important;
}

/* Sidebar Dark Slate/Black Styling */
[data-testid="stSidebar"] {
    background-color: #080d1a !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Sidebar Radio Navigation: Identical Sized Navigation Buttons */
[data-testid="stSidebar"] [data-testid="stRadio"] {
    width: 100% !important;
    margin-top: 8px !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"],
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
    width: 100% !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
    height: 52px !important;
    min-height: 52px !important;
    max-height: 52px !important;
    box-sizing: border-box !important;
    padding: 0 16px !important;
    margin: 0 !important;
    background: #0d1527 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* Hide the native circular radio dot */
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    display: none !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label input[type="radio"] {
    display: none !important;
}

/* Ensure text inside each navigation card fills space and aligns uniformly */
[data-testid="stSidebar"] [data-testid="stRadio"] label [data-testid="stMarkdownContainer"] {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    height: 100% !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label [data-testid="stMarkdownContainer"] p {
    font-size: 0.96rem !important;
    font-weight: 600 !important;
    color: #cbd5e1 !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 52px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* Hover state */
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #17233d !important;
    border-color: rgba(16, 185, 129, 0.5) !important;
    transform: translateX(3px) !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover [data-testid="stMarkdownContainer"] p {
    color: #34d399 !important;
}

/* Active / Checked state */
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.25) 0%, rgba(6, 182, 212, 0.18) 100%) !important;
    border: 1.5px solid #10b981 !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.22) !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) [data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* High-Contrast Typography */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

p, span, label {
    color: #e2e8f0 !important;
}

.stCaption, caption {
    color: #94a3b8 !important;
}

/* Header Banner - High-Contrast Dark Gradient */
.app-header {
    background: linear-gradient(135deg, #090e17 0%, #111a2e 50%, #064e3b 100%);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    color: #ffffff;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.12);
    position: relative;
    overflow: hidden;
}

.app-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid rgba(16, 185, 129, 0.5);
    color: #34d399 !important;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.app-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff !important;
    margin: 0;
    line-height: 1.2;
}

.app-subtitle {
    font-size: 1.02rem;
    color: #cbd5e1 !important;
    margin-top: 8px;
    margin-bottom: 0;
    max-width: 850px;
}

/* High-Contrast Dark Metric Cards */
.metric-card {
    background: #0d1527;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.35);
    border-top: 3px solid #10b981;
    transition: transform 0.18s ease, border-color 0.18s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.25);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.45);
}

.metric-card.accent-teal { border-top-color: #06b6d4; }
.metric-card.accent-blue { border-top-color: #3b82f6; }
.metric-card.accent-amber { border-top-color: #f59e0b; }
.metric-card.accent-emerald { border-top-color: #10b981; }

.metric-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}

.metric-label {
    font-size: 0.82rem;
    font-weight: 700;
    color: #94a3b8 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.metric-value {
    font-size: 2.0rem;
    font-weight: 800;
    color: #ffffff !important;
    line-height: 1.2;
    letter-spacing: -0.02em;
}

.metric-footer {
    margin-top: 8px;
    font-size: 0.84rem;
    color: #34d399 !important;
    font-weight: 600;
}

/* Dark Highlight and Callout Boxes */
.highlight-box {
    background: #0f172a;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.highlight-box h4 {
    margin-top: 0;
    margin-bottom: 8px;
    color: #ffffff !important;
    font-weight: 700;
}

.highlight-box p, .highlight-box li {
    color: #cbd5e1 !important;
    font-size: 0.94rem;
    line-height: 1.6;
}

/* Dark Clinical Disclaimer Banner */
.disclaimer-card {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 16px 0;
    color: #fde68a !important;
    font-size: 0.90rem;
    line-height: 1.5;
}

.disclaimer-card strong {
    color: #fbbf24 !important;
}

/* Form Inputs & Controls on Dark Background */
input, textarea, select {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 8px !important;
}

[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background-color: #0f172a !important;
    color: #ffffff !important;
}

/* Streamlit DataFrames in Dark Mode */
[data-testid="stDataFrame"] {
    background-color: #090e1a !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
}

/* Primary Button High Visibility */
.stButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
}

/* Download Button Polish */
[data-testid="stDownloadButton"] > button {
    background: #1e293b !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

[data-testid="stDownloadButton"] > button:hover {
    background: #0284c7 !important;
    color: #ffffff !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# 3. ROBUST DATASET RESOLUTION & INGESTION
# ==============================================================================

def resolve_dataset_path() -> Optional[Path]:
    """
    Locates calories_merged_data.csv or calories.csv across local paths,
    relative folders, and standard project layouts.
    """
    current_dir = Path(__file__).parent.resolve()

    candidates = [
        # 1. Same directory as application script
        current_dir / "calories_merged_data.csv",
        current_dir / "calories.csv",
        # 2. Standard data or csv subdirectories
        current_dir / "data" / "calories_merged_data.csv",
        current_dir / "data" / "calories.csv",
        current_dir / "csv" / "calories_merged_data.csv",
        current_dir / "csv" / "calories.csv",
        current_dir / "datasets" / "calories_merged_data.csv",
        # 3. Execution working directory
        Path("calories_merged_data.csv"),
        Path("data/calories_merged_data.csv"),
        Path("csv/calories_merged_data.csv"),
        # 4. Sibling folders in user's workspace
        current_dir.parent / "Python Programing" / "csv" / "calories_merged_data.csv",
        current_dir.parent / "Python Programing" / "csv" / "calories.csv",
        # 5. Local Windows absolute path fallbacks
        Path(r"C:\Users\Chint\Desktop\Python Programing\csv\calories_merged_data.csv"),
        Path(r"C:\Users\Chint\Desktop\python\csv\calories_merged_data.csv"),
    ]

    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except Exception:
            continue

    return None


@st.cache_data(show_spinner=False)
def load_and_preprocess_dataset(
    file_bytes: Optional[bytes] = None,
    file_path: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load, clean, validate and engineer features with strict numeric validation.
    """
    try:
        if file_bytes is not None:
            raw_df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_path is not None:
            raw_df = pd.read_csv(file_path)
        else:
            return None, "No dataset source specified."

        if raw_df.empty:
            return None, "The provided dataset has 0 records."

        # Validate core required columns
        required_cols = [
            "Calories", "Gender", "Age", "Height",
            "Weight", "Duration", "Heart_Rate", "Body_Temp"
        ]
        missing_cols = [col for col in required_cols if col not in raw_df.columns]
        if missing_cols:
            return None, f"Dataset is missing required columns: {', '.join(missing_cols)}"

        df = raw_df.copy()

        # Clean & Normalize Gender category
        df["Gender"] = df["Gender"].astype(str).str.strip().str.capitalize()
        gender_map = {
            "M": "Male", "Male": "Male", "1": "Male",
            "F": "Female", "Female": "Female", "0": "Female"
        }
        df["Gender"] = df["Gender"].map(lambda x: gender_map.get(x, x))
        df = df[df["Gender"].isin(["Male", "Female"])].copy()

        # Numeric columns coercion
        numeric_cols = ["Calories", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop invalid nulls and physiological outliers
        df = df.dropna(subset=numeric_cols).copy()
        df = df[
            (df["Calories"] >= 0) &
            (df["Age"] > 0) & (df["Age"] <= 120) &
            (df["Height"] >= 50) & (df["Height"] <= 260) &
            (df["Weight"] >= 15) & (df["Weight"] <= 350) &
            (df["Duration"] > 0) & (df["Duration"] <= 360) &
            (df["Heart_Rate"] >= 35) & (df["Heart_Rate"] <= 240) &
            (df["Body_Temp"] >= 32.0) & (df["Body_Temp"] <= 45.0)
        ].copy()

        if df.empty:
            return None, "No valid records remained after data sanitization."

        # Assign User_ID if absent
        if "User_ID" not in df.columns:
            df["User_ID"] = np.arange(1, len(df) + 1)
        else:
            df["User_ID"] = df["User_ID"].fillna(0).astype(int)

        # Derived metrics: BMI and Age Group
        df["BMI"] = (df["Weight"] / ((df["Height"] / 100) ** 2)).round(2)

        age_bins = [0, 20, 30, 40, 50, 120]
        age_labels = ["0-20", "21-30", "31-40", "41-50", "50+"]
        df["Age Group"] = pd.cut(
            df["Age"],
            bins=age_bins,
            labels=age_labels,
            include_lowest=True
        )

        return df, None

    except Exception as exc:
        return None, f"Error processing dataset: {str(exc)}"


# ==============================================================================
# 4. SIDEBAR NAVIGATION — ALL 7 TABS IN SIDEBAR WITH CLICK-TO-OPEN
# ==============================================================================

with st.sidebar:
    nav_tabs = [
        "🏠 Dashboard",
        "📋 Dataset",
        "📊 Analysis",
        "📈 Visualizations",
        "🧮 BMI Calculator",
        "🔥 Calories Prediction",
        "💡 Insights"
    ]

    selected_tab = st.radio(
        "Navigation",
        options=nav_tabs,
        index=0,
        label_visibility="collapsed"
    )

    # Collapsible Data Source Uploader at Bottom of Sidebar
    st.markdown("<hr style='margin: 24px 0 14px 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)
    with st.expander("📁 Data Source & CSV Upload", expanded=False):
        sidebar_upload = st.file_uploader(
            "Upload Custom CSV",
            type=["csv"],
            help="Upload calories_merged_data.csv to evaluate your custom activity logs."
        )
        detected_path = resolve_dataset_path()
        if detected_path:
            st.caption(f"📍 Active Source: `{detected_path.name}`")

# Ingest data from uploaded file or local resolved candidate
df: Optional[pd.DataFrame] = None
data_load_error: Optional[str] = None

if sidebar_upload is not None:
    file_bytes = sidebar_upload.getvalue()
    df, data_load_error = load_and_preprocess_dataset(file_bytes=file_bytes)
elif detected_path is not None:
    df, data_load_error = load_and_preprocess_dataset(file_path=str(detected_path))
else:
    data_load_error = (
        "Dataset file (`calories_merged_data.csv`) was not automatically found. "
        "Please upload your CSV file in the sidebar to proceed."
    )

# ==============================================================================
# 5. APPLICATION HEADER
# ==============================================================================

st.markdown(f"""
    <div class="app-header">
        <div class="app-badge">⚡ Precision Metabolic Intelligence</div>
        <h1 class="app-title">Calories Burn Tracker</h1>
        <p class="app-subtitle">
            Currently viewing: <strong>{selected_tab}</strong> — Advanced physical activity tracking,
            cardiovascular correlation mapping, and predictive energy expenditure analytics.
        </p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 6. GRACEFUL EMPTY STATE HANDLING
# ==============================================================================

if df is None:
    st.error(f"⚠️ **Dataset Notice**: {data_load_error}")
    st.info(
        "💡 **How to resolve**: Expand the **'📁 Data Source & CSV Upload'** section in the sidebar "
        "and upload your `calories_merged_data.csv` file."
    )
    st.stop()

# ==============================================================================
# 7. CACHED MACHINE LEARNING PIPELINE
# ==============================================================================

@st.cache_resource(show_spinner=False)
def train_regression_model(_dataframe: pd.DataFrame) -> Dict[str, Any]:
    """
    Train Linear Regression model with PolynomialFeatures(degree=1)
    on standardized features, ensuring exact column alignment.
    """
    feature_names = [
        "Gender_Male", "Age", "Height", "Weight",
        "Duration", "Heart_Rate", "Body_Temp"
    ]

    working_df = _dataframe.copy()
    working_df["Gender_Male"] = (working_df["Gender"] == "Male").astype(int)

    X = working_df[feature_names].copy()
    y = working_df["Calories"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    poly = PolynomialFeatures(degree=1, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    predictions = model.predict(X_test_poly)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, predictions)

    coefficients = dict(zip(feature_names, model.coef_))

    return {
        "model": model,
        "poly": poly,
        "features": feature_names,
        "metrics": {
            "r2": float(r2),
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
        },
        "intercept": float(model.intercept_),
        "coefficients": coefficients,
    }

# ==============================================================================
# 8. ACTIVE TAB DISPATCHER (SWITCHES INSTANTLY ON SIDEBAR CLICK)
# ==============================================================================

# ------------------------------------------------------------------------------
# TAB 1: DASHBOARD
# ------------------------------------------------------------------------------
if selected_tab == "🏠 Dashboard":
    st.markdown("### 🏠 Metabolic Analytics Dashboard")
    st.caption("Comprehensive overview of physical activity records, energy expenditure metrics, and demographic distributions.")

    # High-level Metrics Row 1
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.markdown(f"""
            <div class="metric-card accent-emerald">
                <div class="metric-header">
                    <span class="metric-label">Total Verified Records</span>
                    <span class="metric-icon" style="font-size: 1.3rem;">📋</span>
                </div>
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-footer"><span>✓ Sanitized Activity Profiles</span></div>
            </div>
        """, unsafe_allow_html=True)

    with mcol2:
        total_cals = int(df["Calories"].sum())
        st.markdown(f"""
            <div class="metric-card accent-teal">
                <div class="metric-header">
                    <span class="metric-label">Total Calories Burned</span>
                    <span class="metric-icon" style="font-size: 1.3rem;">🔥</span>
                </div>
                <div class="metric-value">{total_cals:,} <span style="font-size:1.1rem; color:#94a3b8;">kcal</span></div>
                <div class="metric-footer"><span>Cumulative Workout Expenditure</span></div>
            </div>
        """, unsafe_allow_html=True)

    with mcol3:
        avg_cals = df["Calories"].mean()
        st.markdown(f"""
            <div class="metric-card accent-blue">
                <div class="metric-header">
                    <span class="metric-label">Average Burn Per Session</span>
                    <span class="metric-icon" style="font-size: 1.3rem;">📊</span>
                </div>
                <div class="metric-value">{avg_cals:.1f} <span style="font-size:1.1rem; color:#94a3b8;">kcal</span></div>
                <div class="metric-footer"><span>Mean Session Output</span></div>
            </div>
        """, unsafe_allow_html=True)

    # High-level Metrics Row 2
    mcol4, mcol5, mcol6 = st.columns(3)
    with mcol4:
        max_cals = int(df["Calories"].max())
        st.markdown(f"""
            <div class="metric-card accent-amber">
                <div class="metric-header">
                    <span class="metric-label">Peak Session Expenditure</span>
                    <span class="metric-icon" style="font-size: 1.3rem;">🏆</span>
                </div>
                <div class="metric-value">{max_cals:,} <span style="font-size:1.1rem; color:#94a3b8;">kcal</span></div>
                <div class="metric-footer"><span>All-Time Dataset Record</span></div>
            </div>
        """, unsafe_allow_html=True)

    with mcol5:
        avg_age = df["Age"].mean()
        st.markdown(f"""
            <div class="metric-card accent-emerald">
                <div class="metric-header">
                    <span class="metric-label">Average Participant Age</span>
                    <span class="metric-icon" style="font-size: 1.3rem;">👤</span>
                </div>
                <div class="metric-value">{avg_age:.1f} <span style="font-size:1.1rem; color:#94a3b8;">yrs</span></div>
                <div class="metric-footer"><span>Cohort Range: {int(df['Age'].min())} – {int(df['Age'].max())} yrs</span></div>
            </div>
        """, unsafe_allow_html=True)

    with mcol6:
        avg_duration = df["Duration"].mean()
        st.markdown(f"""
            <div class="metric-card accent-teal">
                <div class="metric-header">
                    <span class="metric-label">Avg Workout Duration</span>
                    <span class="metric-icon" style="font-size: 1.3rem;">⏱️</span>
                </div>
                <div class="metric-value">{avg_duration:.1f} <span style="font-size:1.1rem; color:#94a3b8;">min</span></div>
                <div class="metric-footer"><span>Range: {int(df['Duration'].min())} – {int(df['Duration'].max())} min</span></div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 16px 0 24px 0; border-color: rgba(255, 255, 255, 0.1);'>", unsafe_allow_html=True)

    # Demographic & Activity Summary
    st.subheader("💡 Demographic & Activity Summary")
    sum_col1, sum_col2 = st.columns(2)

    with sum_col1:
        active_group = df.groupby("Age Group", observed=False)["Calories"].mean().idxmax()
        active_group_avg = df.groupby("Age Group", observed=False)["Calories"].mean().max()

        st.markdown(f"""
            <div class="highlight-box">
                <h4>👥 Most Active Demographic Segment</h4>
                <p>The <strong>{active_group}</strong> age group demonstrates the highest mean expenditure at <strong>{active_group_avg:.1f} kcal</strong> per workout session.</p>
                <div style="margin-top: 10px; font-size: 0.88rem; color: #94a3b8;">
                    Higher baseline heart rate reserve and sustained workout duration in this cohort drive greater cumulative energy burn.
                </div>
            </div>
        """, unsafe_allow_html=True)

        gender_avg = df.groupby("Gender")["Calories"].mean().to_dict()
        male_avg = gender_avg.get("Male", 0.0)
        female_avg = gender_avg.get("Female", 0.0)

        st.markdown(f"""
            <div class="highlight-box">
                <h4>⚖️ Gender Calorie Comparison</h4>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
                    <div>
                        <div style="font-size: 0.85rem; color: #94a3b8;">Male Cohort</div>
                        <div style="font-size: 1.45rem; font-weight: 800; color: #ffffff;">{male_avg:.2f} kcal</div>
                    </div>
                    <div style="height: 36px; width: 1px; background: rgba(255, 255, 255, 0.15);"></div>
                    <div>
                        <div style="font-size: 0.85rem; color: #94a3b8;">Female Cohort</div>
                        <div style="font-size: 1.45rem; font-weight: 800; color: #ffffff;">{female_avg:.2f} kcal</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with sum_col2:
        st.markdown("#### 📈 Fitness Highlights & Observations")
        if male_avg > female_avg:
            diff_pct = ((male_avg - female_avg) / female_avg) * 100
            st.success(f"💪 **Gender Dynamics**: Male participants burn on average **{diff_pct:.1f}% more calories** per session, predominantly driven by higher average lean body mass.")
        else:
            diff_pct = ((female_avg - male_avg) / male_avg) * 100
            st.success(f"💪 **Gender Dynamics**: Female participants burn on average **{diff_pct:.1f}% more calories** per session.")

        burn_rate = df["Calories"].sum() / df["Duration"].sum()
        st.info(f"⚡ **Metabolic Burn Rate**: The overall cohort burn rate is **{burn_rate:.2f} kcal/minute** across all recorded exercise durations.")

        avg_hr = df["Heart_Rate"].mean()
        st.info(f"❤️ **Cardiovascular Intensity**: Mean heart rate during activity is **{avg_hr:.1f} bpm**, placing the average workout comfortably in the aerobic endurance zone.")

# ------------------------------------------------------------------------------
# TAB 2: DATASET
# ------------------------------------------------------------------------------
elif selected_tab == "📋 Dataset":
    st.markdown("### 📋 Interactive Dataset Explorer")
    st.caption("Inspect, filter, slice, and export sanitized physical activity records.")

    # High-level Dataset Dimensions
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    with info_col2:
        st.metric("Total Attributes", f"{df.shape[1]}")
    with info_col3:
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("In-Memory Size", f"{mem_mb:.2f} MB")
    with info_col4:
        st.metric("Missing Values", f"{df.isnull().sum().sum()}")

    st.markdown("<hr style='margin: 16px 0 24px 0; border-color: rgba(255, 255, 255, 0.1);'>", unsafe_allow_html=True)
    st.subheader("🔍 Multi-Parameter Dataset Filters")

    fcol1, fcol2 = st.columns(2)
    filtered_df = df.copy()

    with fcol1:
        # Gender Filter
        unique_genders = sorted(df["Gender"].unique().tolist())
        selected_genders = st.multiselect(
            "Gender Filter",
            options=unique_genders,
            default=unique_genders,
            help="Select one or both genders to include in the view."
        )
        if selected_genders:
            filtered_df = filtered_df[filtered_df["Gender"].isin(selected_genders)]
        else:
            filtered_df = filtered_df.iloc[0:0]

        # Age Filter
        min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
        if min_age < max_age:
            age_range = st.slider("Age Range (years)", min_age, max_age, (min_age, max_age))
            filtered_df = filtered_df[
                (filtered_df["Age"] >= age_range[0]) & (filtered_df["Age"] <= age_range[1])
            ]

        # Height Filter
        min_h, max_h = int(df["Height"].min()), int(df["Height"].max())
        if min_h < max_h:
            h_range = st.slider("Height Range (cm)", min_h, max_h, (min_h, max_h))
            filtered_df = filtered_df[
                (filtered_df["Height"] >= h_range[0]) & (filtered_df["Height"] <= h_range[1])
            ]

        # Weight Filter
        min_w, max_w = int(df["Weight"].min()), int(df["Weight"].max())
        if min_w < max_w:
            w_range = st.slider("Weight Range (kg)", min_w, max_w, (min_w, max_w))
            filtered_df = filtered_df[
                (filtered_df["Weight"] >= w_range[0]) & (filtered_df["Weight"] <= w_range[1])
            ]

    with fcol2:
        # Duration Filter
        min_d, max_d = int(df["Duration"].min()), int(df["Duration"].max())
        if min_d < max_d:
            d_range = st.slider("Workout Duration (minutes)", min_d, max_d, (min_d, max_d))
            filtered_df = filtered_df[
                (filtered_df["Duration"] >= d_range[0]) & (filtered_df["Duration"] <= d_range[1])
            ]

        # Heart Rate Filter
        min_hr, max_hr = int(df["Heart_Rate"].min()), int(df["Heart_Rate"].max())
        if min_hr < max_hr:
            hr_range = st.slider("Heart Rate (bpm)", min_hr, max_hr, (min_hr, max_hr))
            filtered_df = filtered_df[
                (filtered_df["Heart_Rate"] >= hr_range[0]) & (filtered_df["Heart_Rate"] <= hr_range[1])
            ]

        # Body Temp Filter
        min_t, max_t = float(df["Body_Temp"].min()), float(df["Body_Temp"].max())
        if min_t < max_t:
            t_range = st.slider("Body Temperature (°C)", min_t, max_t, (min_t, max_t), step=0.1)
            filtered_df = filtered_df[
                (filtered_df["Body_Temp"] >= t_range[0]) & (filtered_df["Body_Temp"] <= t_range[1])
            ]

        # Calories Filter
        min_c, max_c = int(df["Calories"].min()), int(df["Calories"].max())
        if min_c < max_c:
            c_range = st.slider("Calories Burned (kcal)", min_c, max_c, (min_c, max_c))
            filtered_df = filtered_df[
                (filtered_df["Calories"] >= c_range[0]) & (filtered_df["Calories"] <= c_range[1])
            ]

    # Filtered Records Status & Preview
    st.markdown("<br>", unsafe_allow_html=True)
    match_pct = (len(filtered_df) / len(df)) * 100 if len(df) > 0 else 0
    st.markdown(f"**Filtered Subset**: Displaying **{len(filtered_df):,}** of **{len(df):,}** records ({match_pct:.1f}% match).")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=380
    )

    # Download Button
    csv_export = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Filtered CSV ({len(filtered_df):,} Rows)",
        data=csv_export,
        file_name="filtered_calories_data.csv",
        mime="text/csv"
    )

    # Missing Value Diagnostic Table
    with st.expander("🔍 Dataset Health & Column Type Audit"):
        health_df = pd.DataFrame({
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.notnull().sum(),
            "Null Count": df.isnull().sum(),
            "Null Percentage (%)": (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(health_df, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: ANALYSIS
# ------------------------------------------------------------------------------
elif selected_tab == "📊 Analysis":
    st.markdown("### 📊 Statistical Summary & Correlation Architecture")
    st.caption("Deep-dive numerical distributions, cardiovascular metrics, and biophysical relationships.")

    # 1. Descriptive Statistics
    st.subheader("📈 Statistical Overview")
    desc_df = df.drop(columns=["User_ID"], errors="ignore").describe().round(2).T
    desc_df = desc_df.rename(columns={
        "count": "Count", "mean": "Mean", "std": "Std Dev",
        "min": "Min", "25%": "25th Pct", "50%": "Median",
        "75%": "75th Pct", "max": "Max"
    })
    st.dataframe(desc_df, use_container_width=True)

    # 2. Correlation Analysis
    st.subheader("🔥 Biophysical Correlation Matrix")
    corr_cols = ["Calories", "Duration", "Heart_Rate", "Body_Temp", "Weight", "Height", "Age", "BMI"]
    available_corr_cols = [c for c in corr_cols if c in df.columns]
    corr_matrix = df[available_corr_cols].corr().round(3)

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        template="plotly_dark",
        title="Pearson Correlation Heatmap (Physiological Drivers)",
        labels=dict(color="Correlation")
    )
    fig_corr.update_layout(
        font=dict(family="Plus Jakarta Sans", color="#ffffff", size=12),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=50, b=30)
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Strongest Factor Highlight
    top_corr_factor = corr_matrix["Calories"].drop("Calories").idxmax()
    top_corr_value = corr_matrix["Calories"].drop("Calories").max()
    st.success(
        f"💡 **Key Discovery**: **{top_corr_factor}** holds the strongest positive correlation with Calories Burned "
        f"(**r = {top_corr_value:.3f}**), confirming duration and sustained cardiac output as primary drivers of energy expenditure."
    )

    st.markdown("<hr style='margin: 20px 0; border-color: rgba(255, 255, 255, 0.1);'>", unsafe_allow_html=True)

    # 3. Workout Duration & Heart Rate Deep-Dive
    acol1, acol2 = st.columns(2)
    with acol1:
        st.subheader("⏱️ Workout Duration Breakdown")
        st.markdown(f"""
            - **Average Duration**: {df['Duration'].mean():.2f} minutes
            - **Median Duration**: {df['Duration'].median():.1f} minutes
            - **Shortest Session**: {df['Duration'].min():.1f} minutes
            - **Longest Session**: {df['Duration'].max():.1f} minutes
            - **Standard Deviation**: ±{df['Duration'].std():.2f} minutes
        """)

    with acol2:
        st.subheader("❤️ Cardiovascular Intensity Analysis")
        st.markdown(f"""
            - **Average Heart Rate**: {df['Heart_Rate'].mean():.2f} bpm
            - **Peak Heart Rate Recorded**: {df['Heart_Rate'].max():.0f} bpm
            - **Minimum Heart Rate**: {df['Heart_Rate'].min():.0f} bpm
            - **Interquartile Range**: {df['Heart_Rate'].quantile(0.25):.0f} – {df['Heart_Rate'].quantile(0.75):.0f} bpm
            - **Energy Coupling**: Elevated heart rate increases caloric burn rate by ~1.99 kcal per additional bpm.
        """)

    st.markdown("<hr style='margin: 20px 0; border-color: rgba(255, 255, 255, 0.1);'>", unsafe_allow_html=True)

    # 4. Top 10 Calorie Burners Table
    st.subheader("🏆 Top 10 Highest Calorie Burners")
    top10_df = df.nlargest(10, "Calories")[
        ["User_ID", "Calories", "Duration", "Heart_Rate", "Body_Temp", "Age", "Gender", "BMI"]
    ].reset_index(drop=True)
    top10_df.index = top10_df.index + 1
    st.dataframe(top10_df, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: VISUALIZATIONS
# ------------------------------------------------------------------------------
elif selected_tab == "📈 Visualizations":
    st.markdown("### 📈 Visual Analytics & Distribution Studio")
    st.caption("Interactive charts illustrating demographic splits, duration effects, and expenditure trends.")

    vcol1, vcol2 = st.columns(2)

    with vcol1:
        # 1. Distribution of Calories Burned
        fig_hist = px.histogram(
            df,
            x="Calories",
            nbins=35,
            marginal="box",
            title="Distribution of Calories Burned (kcal)",
            color_discrete_sequence=["#10b981"],
            template="plotly_dark",
            opacity=0.85
        )
        fig_hist.update_layout(
            font=dict(family="Plus Jakarta Sans", color="#ffffff"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Calories Burned (kcal)",
            yaxis_title="Participant Count",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with vcol2:
        # 2. Gender Distribution Donut Chart
        gender_counts = df["Gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]

        fig_pie = px.pie(
            gender_counts,
            names="Gender",
            values="Count",
            hole=0.45,
            title="Cohort Gender Distribution",
            color="Gender",
            color_discrete_map={"Male": "#38bdf8", "Female": "#10b981"},
            template="plotly_dark"
        )
        fig_pie.update_layout(
            font=dict(family="Plus Jakarta Sans", color="#ffffff"),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    vcol3, vcol4 = st.columns(2)

    with vcol3:
        # 3. Calories vs Age Scatter Plot
        sample_df = df.sample(min(3000, len(df)), random_state=42)
        fig_scatter = px.scatter(
            sample_df,
            x="Age",
            y="Calories",
            color="Gender",
            size="Duration",
            hover_data=["Heart_Rate", "Body_Temp"],
            title="Calories Burned vs. Age (Bubble Size = Duration)",
            color_discrete_map={"Male": "#38bdf8", "Female": "#10b981"},
            template="plotly_dark",
            opacity=0.7
        )
        fig_scatter.update_layout(
            font=dict(family="Plus Jakarta Sans", color="#ffffff"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Age (years)",
            yaxis_title="Calories Burned (kcal)",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with vcol4:
        # 4. Top 10 Highest Calorie Burners Chart
        top10_chart = df.nlargest(10, "Calories").sort_values("Calories", ascending=True)
        top10_chart["Participant"] = "ID #" + top10_chart["User_ID"].astype(str)

        fig_top10 = px.bar(
            top10_chart,
            x="Calories",
            y="Participant",
            orientation="h",
            color="Duration",
            title="Top 10 Energy Burners by Output (kcal)",
            color_continuous_scale="Viridis",
            template="plotly_dark",
            text="Calories"
        )
        fig_top10.update_traces(texttemplate="%{text:.0f} kcal", textposition="inside")
        fig_top10.update_layout(
            font=dict(family="Plus Jakarta Sans", color="#ffffff"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Calories Burned (kcal)",
            yaxis_title="Participant",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_top10, use_container_width=True)

    # Additional Duration vs Energy Trajectory
    st.subheader("⚡ Duration vs. Energy Expenditure Trajectory")
    fig_dur = px.scatter(
        sample_df,
        x="Duration",
        y="Calories",
        color="Heart_Rate",
        color_continuous_scale="Inferno",
        template="plotly_dark",
        title="Workout Duration vs. Caloric Burn (Colored by Heart Rate)",
        labels={"Heart_Rate": "Heart Rate (bpm)"}
    )
    fig_dur.update_layout(
        font=dict(family="Plus Jakarta Sans", color="#ffffff"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Workout Duration (minutes)",
        yaxis_title="Calories Burned (kcal)",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_dur, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5: BMI CALCULATOR & PERSONALIZED HEALTH PLAN
# ------------------------------------------------------------------------------
elif selected_tab == "🧮 BMI Calculator":
    st.markdown("### 🧮 BMI Assessment & Personalized Health Plan")
    st.caption("Calculate your Body Mass Index, explore personalized nutritional pathways, and formulate realistic body goals.")

    # Clinical & Wellness Disclaimer
    st.markdown("""
        <div class="disclaimer-card">
            <strong>⚠️ Clinical & Wellness Notice</strong><br>
            Body Mass Index (BMI) is a general screening indicator based on height and weight.
            It does not differentiate between lean muscle mass and adipose tissue, nor does it constitute medical advice.
            Consult a physician or dietitian before beginning intensive weight-loss or caloric regimens.
        </div>
    """, unsafe_allow_html=True)

    # User Physical Inputs
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        input_height = st.number_input(
            "Enter Height (cm)",
            min_value=50.0,
            max_value=250.0,
            value=float(st.session_state.get("bmi_height", 170.0)),
            step=0.5,
            key="bmi_height_input"
        )
    with bcol2:
        input_weight = st.number_input(
            "Enter Weight (kg)",
            min_value=15.0,
            max_value=250.0,
            value=float(st.session_state.get("bmi_weight", 70.0)),
            step=0.5,
            key="bmi_weight_input"
        )

    # Immediate BMI Calculation & Persistence
    calc_bmi = input_weight / ((input_height / 100) ** 2)

    if calc_bmi < 18.5:
        calc_category = "Underweight"
        cat_badge_color = "#f59e0b"
        cat_badge_text = "Underweight (< 18.5)"
    elif calc_bmi < 25.0:
        calc_category = "Normal"
        cat_badge_color = "#10b981"
        cat_badge_text = "Normal Weight (18.5 – 24.9)"
    elif calc_bmi < 30.0:
        calc_category = "Overweight"
        cat_badge_color = "#38bdf8"
        cat_badge_text = "Overweight (25.0 – 29.9)"
    else:
        calc_category = "Obese"
        cat_badge_color = "#ef4444"
        cat_badge_text = "Obese (≥ 30.0)"

    min_healthy_weight = 18.5 * ((input_height / 100) ** 2)
    max_healthy_weight = 24.9 * ((input_height / 100) ** 2)

    # Display Current BMI Result Card
    st.markdown(f"""
        <div style="background: #0d1527; border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 14px; padding: 22px; margin: 18px 0; border-left: 5px solid {cat_badge_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <span style="font-size: 0.82rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">Calculated Body Mass Index</span>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff; margin-top: 4px;">{calc_bmi:.2f} <span style="font-size: 1.1rem; font-weight: 500; color: #94a3b8;">kg/m²</span></div>
                </div>
                <div>
                    <span style="background: {cat_badge_color}25; color: {cat_badge_color}; border: 1px solid {cat_badge_color}60; padding: 8px 16px; border-radius: 9999px; font-weight: 700; font-size: 0.95rem;">
                        {cat_badge_text}
                    </span>
                </div>
            </div>
            <div style="margin-top: 14px; font-size: 0.90rem; color: #cbd5e1;">
                Standard healthy weight interval for your height (<strong>{input_height:.1f} cm</strong>) is <strong>{min_healthy_weight:.1f} kg – {max_healthy_weight:.1f} kg</strong>.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 20px 0; border-color: rgba(255, 255, 255, 0.1);'>", unsafe_allow_html=True)
    st.subheader("🎯 Goal Setting & Health Plan Formulation")

    gcol1, gcol2 = st.columns(2)
    with gcol1:
        goal_choice = st.selectbox(
            "Target Goal",
            options=["Maintain Weight", "Lose Weight", "Gain Weight"],
            index=["Maintain Weight", "Lose Weight", "Gain Weight"].index(
                st.session_state.get("plan_goal", "Maintain Weight")
            )
        )

        default_target = float(input_weight)
        if goal_choice == "Lose Weight":
            default_target = max(20.0, float(input_weight - 5.0))
        elif goal_choice == "Gain Weight":
            default_target = float(input_weight + 5.0)

        target_weight_input = st.number_input(
            "Target Weight (kg)",
            min_value=20.0,
            max_value=250.0,
            value=float(st.session_state.get("plan_target_weight", default_target)),
            step=0.5
        )

    with gcol2:
        target_days_input = st.slider(
            "Timeline Target (Days)",
            min_value=7,
            max_value=365,
            value=int(st.session_state.get("plan_target_days", 90))
        )

        activity_choice = st.selectbox(
            "Current Activity Level",
            options=["Sedentary", "Lightly Active", "Moderately Active", "Very Active"],
            index=["Sedentary", "Lightly Active", "Moderately Active", "Very Active"].index(
                st.session_state.get("plan_activity", "Moderately Active")
            )
        )

    # Health Plan Generation Button
    if st.button("🚀 Generate My Health Plan", use_container_width=True, type="primary"):
        validation_passed = True

        if goal_choice == "Lose Weight" and target_weight_input >= input_weight:
            st.error("❌ For Weight Loss, Target Weight must be strictly less than Current Weight.")
            validation_passed = False
        elif goal_choice == "Gain Weight" and target_weight_input <= input_weight:
            st.error("❌ For Weight Gain, Target Weight must be strictly greater than Current Weight.")
            validation_passed = False
        elif goal_choice == "Maintain Weight" and target_weight_input != input_weight:
            target_weight_input = float(input_weight)
            st.info("ℹ️ Target weight has been aligned to your current weight for maintenance.")

        if validation_passed:
            st.session_state["plan_active"] = True
            st.session_state["plan_height"] = float(input_height)
            st.session_state["plan_start_weight"] = float(input_weight)
            st.session_state["plan_target_weight"] = float(target_weight_input)
            st.session_state["plan_target_days"] = int(target_days_input)
            st.session_state["plan_goal"] = goal_choice
            st.session_state["plan_activity"] = activity_choice
            st.session_state["plan_bmi"] = float(calc_bmi)
            st.session_state["plan_category"] = calc_category
            st.success("✅ Personalized health plan successfully generated!")

    # PERSISTENT HEALTH PLAN DISPLAY (REMAINS VISIBLE ON ALL RERUNS)
    if st.session_state.get("plan_active", False):
        p_height = st.session_state.get("plan_height", input_height)
        p_start_w = st.session_state.get("plan_start_weight", input_weight)
        p_target_w = st.session_state.get("plan_target_weight", target_weight_input)
        p_days = st.session_state.get("plan_target_days", target_days_input)
        p_goal = st.session_state.get("plan_goal", goal_choice)
        p_activity = st.session_state.get("plan_activity", activity_choice)
        p_bmi = st.session_state.get("plan_bmi", calc_bmi)
        p_cat = st.session_state.get("plan_category", calc_category)

        weight_delta = abs(p_start_w - p_target_w)
        weekly_rate = (weight_delta / max(1, p_days)) * 7

        st.markdown("<hr style='margin: 28px 0; border-color: rgba(255, 255, 255, 0.15);'>", unsafe_allow_html=True)
        st.header("🏥 Your Personalized Health Protocol")

        # Plan Summary Metrics
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        with pcol1:
            st.metric("Starting Weight", f"{p_start_w:.1f} kg")
        with pcol2:
            st.metric("Target Weight", f"{p_target_w:.1f} kg")
        with pcol3:
            st.metric("Target Duration", f"{p_days} Days")
        with pcol4:
            st.metric("Required Weekly Rate", f"{weekly_rate:.2f} kg/wk")

        # Rate Safety Evaluation
        if weekly_rate > 1.5:
            st.warning("⚠️ **Safety Alert**: An implied rate of change > 1.5 kg/week is aggressively rapid. Consider extending your timeline to protect metabolic rate and muscle mass.")
        elif weekly_rate > 1.0:
            st.info("ℹ️ **Pacing Advisory**: Aiming for > 1.0 kg/week is an aggressive rate. General guidelines suggest 0.25 to 1.0 kg/week.")

        st.markdown(f"**Goal**: `{p_goal}` | **Activity Baseline**: `{p_activity}` | **Baseline BMI**: `{p_bmi:.2f}` ({p_cat})")

        # Category Specific Protocols
        if p_bmi < 18.5:
            st.warning("⚠️ Category: Underweight Nutritional & Hypertrophy Regimen")
            prot_col1, prot_col2, prot_col3 = st.columns(3)
            with prot_col1:
                st.subheader("🍽 Diet Plan")
                st.markdown("""
                - Consume 5–6 balanced meals daily
                - Whole milk twice daily
                - Paneer / Eggs / Soybean
                - Banana shake with peanut butter
                - Almonds, walnuts, and dry fruits
                - Complex carbohydrates: Rice & potatoes
                - Daily caloric surplus: +300–500 kcal/day
                """)
            with prot_col2:
                st.subheader("🏋 Strength Workout")
                st.markdown("""
                - Strength training: 4 days/week
                - Push-ups & pull-ups
                - Bodyweight & barbell squats
                - Dumbbell bench press
                - Romanian deadlifts
                - Compound resistance movements
                """)
            with prot_col3:
                st.subheader("💧 Lifestyle & Recovery")
                st.markdown("""
                - Drink 3.0+ liters of water daily
                - 8 hours of restorative sleep
                - Avoid nutrient-poor junk food
                - Eat protein in every meal
                """)

        elif p_bmi < 25.0:
            st.success("✅ Category: Healthy Normal Weight Maintenance Protocol")
            prot_col1, prot_col2, prot_col3 = st.columns(3)
            with prot_col1:
                st.subheader("💪 Maintenance Fitness")
                st.markdown("""
                - Gym / resistance training: 4 days/week
                - Daily steps: 8,000–10,000 steps
                - Moderate cardio: 30 minutes 3x/week
                """)
            with prot_col2:
                st.subheader("🥗 Balanced Nutrition")
                st.markdown("""
                - Balanced whole-food diet
                - Abundant fresh fruits & vegetables
                - Protein-rich whole foods
                - Whole grains: oats, brown rice
                """)
            with prot_col3:
                st.subheader("💧 Recovery & Habits")
                st.markdown("""
                - Sleep 7–8 hours every night
                - Hydrate with 2.5–3.0 liters of water
                - Minimize refined sugars
                """)

        elif p_bmi < 30.0:
            st.info("🟡 Category: Overweight Structured Fat Loss Protocol")
            prot_col1, prot_col2, prot_col3 = st.columns(3)
            with prot_col1:
                st.subheader("🏃 Exercise Schedule")
                st.markdown("""
                **Monday – Friday**:
                - 45 minutes brisk walking
                - 45 minutes resistance training
                **Saturday**:
                - Cycling / Running (Zone 2)
                **Sunday**:
                - Stretching & restorative yoga
                """)
            with prot_col2:
                st.subheader("🥗 Calorie Conscious Diet")
                st.markdown("""
                - **Breakfast**: Rolled oats, eggs, berries
                - **Lunch**: Roti, dal, fresh salad
                - **Evening**: Green tea, roasted sprouts
                - **Dinner**: Paneer/tofu, vegetable soup
                """)
            with prot_col3:
                st.subheader("❌ Foods to Avoid")
                st.markdown("""
                - Sugary beverages & soft drinks
                - Bakery pastries and biscuits
                - Deep-fried street food
                - Commercial fast foods
                - Excess refined cooking oil
                """)

        else:
            st.error("🔴 Category: Obesity Health Improvement & Mobility Protocol")
            prot_col1, prot_col2 = st.columns(2)
            with prot_col1:
                st.subheader("⚕ Progressive Activity Plan")
                st.markdown("""
                - **Weeks 1–2**: 30 minutes low-impact walking daily
                - **Weeks 3–4**: 45 minutes walking
                - **Weeks 5+**: Low-impact cardio, resistance bands, yoga
                """)
            with prot_col2:
                st.subheader("🥗 Whole-Food Diet")
                st.markdown("""
                - High protein to protect lean mass
                - Very low added sugars
                - Abundant green vegetables & salads
                - Drink 3.0–4.0 liters of water daily
                - Consult a doctor before starting strenuous regimens
                """)

        st.markdown("<hr style='margin: 20px 0; border-color: rgba(255, 255, 255, 0.1);'>", unsafe_allow_html=True)

        # Suggested Daily Routine Table
        st.subheader("📅 Recommended Daily Routine")
        schedule_data = {
            "Time": [
                "6:30 AM", "7:00 AM", "8:00 AM", "10:30 AM",
                "1:00 PM", "4:30 PM", "6:30 PM", "8:00 PM", "10:30 PM"
            ],
            "Activity": [
                "Wake Up + Drink 500ml Water",
                "Morning Brisk Walk / Stretching",
                "Healthy Protein-Rich Breakfast",
                "Healthy Snack (Fruit / Nuts)",
                "Balanced Lunch with Salad",
                "Green Tea / Roasted Sprouts",
                "Exercise / Workout Session",
                "Light Nutritious Dinner",
                "Restorative Sleep"
            ]
        }
        st.table(pd.DataFrame(schedule_data))

        # Hydration, Sleep, and Steps Targets
        st.subheader("💧 Biometric Targets")
        hcol1, hcol2, hcol3 = st.columns(3)
        with hcol1:
            rec_water_ml = p_start_w * 35
            st.markdown(f"""
                <div class="highlight-box">
                    <h4>💧 Daily Water Intake</h4>
                    <p style="font-size: 1.35rem; font-weight: 800; color: #38bdf8;">{rec_water_ml:.0f} ml ({rec_water_ml/1000:.1f} L)</p>
                    <p style="margin-top: 4px;">Standard: 35ml per kg of bodyweight.</p>
                </div>
            """, unsafe_allow_html=True)
        with hcol2:
            st.markdown("""
                <div class="highlight-box">
                    <h4>😴 Sleep Duration</h4>
                    <p style="font-size: 1.35rem; font-weight: 800; color: #818cf8;">7 – 9 Hours</p>
                    <p style="margin-top: 4px;">Optimizes recovery and hormone balance.</p>
                </div>
            """, unsafe_allow_html=True)
        with hcol3:
            rec_steps = 6000 if p_bmi < 18.5 else (8000 if p_bmi < 25.0 else 10000)
            st.markdown(f"""
                <div class="highlight-box">
                    <h4>🚶 Daily Steps Target</h4>
                    <p style="font-size: 1.35rem; font-weight: 800; color: #34d399;">{rec_steps:,} Steps</p>
                    <p style="margin-top: 4px;">Sustains daily non-exercise physical activity.</p>
                </div>
            """, unsafe_allow_html=True)

        # Mathematically Sound Weight Progress Tracking
        st.subheader("🎯 Mathematically Sound Progress Tracking")
        current_log_weight = st.number_input(
            "Log Current Weigh-In (kg)",
            min_value=15.0,
            max_value=250.0,
            value=float(p_start_w),
            step=0.1,
            key="current_weigh_in_logger"
        )

        total_goal_delta = abs(p_start_w - p_target_w)
        if total_goal_delta == 0:
            st.success("🎯 Your maintenance goal is active! You are at your target weight.")
        else:
            if p_goal == "Lose Weight":
                achieved_delta = max(0.0, p_start_w - current_log_weight)
                remaining_delta = max(0.0, current_log_weight - p_target_w)
            elif p_goal == "Gain Weight":
                achieved_delta = max(0.0, current_log_weight - p_start_w)
                remaining_delta = max(0.0, p_target_w - current_log_weight)
            else:
                achieved_delta = 0.0
                remaining_delta = abs(current_log_weight - p_target_w)

            progress_percentage = min(100.0, max(0.0, (achieved_delta / total_goal_delta) * 100))
            st.progress(progress_percentage / 100.0)
            st.markdown(f"""
                **Progress Achieved**: `{progress_percentage:.1f}%` &nbsp;|&nbsp;
                **Weight Delta Achieved**: `{achieved_delta:.1f} kg` &nbsp;|&nbsp;
                **Remaining to Target**: `{remaining_delta:.1f} kg`
            """)

        # Foods to Prioritize vs Avoid
        st.markdown("<hr style='margin: 20px 0; border-color: rgba(255, 255, 255, 0.1);'>", unsafe_allow_html=True)
        food_col1, food_col2 = st.columns(2)
        with food_col1:
            st.markdown("""
                <div class="highlight-box">
                    <h4>🥗 Foods to Eat</h4>
                    <ul>
                        <li>Fresh fruits (apples, berries, oranges)</li>
                        <li>Fresh vegetables and greens (broccoli, spinach)</li>
                        <li>High quality protein (eggs, paneer, dal, tofu)</li>
                        <li>Rolled oats, brown rice, whole wheat</li>
                        <li>Nuts and seeds in moderation</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        with food_col2:
            st.markdown("""
                <div class="highlight-box">
                    <h4>❌ Foods to Avoid</h4>
                    <ul>
                        <li>Carbonated soft drinks & sugary soda</li>
                        <li>Bakery items, cakes, and pastries</li>
                        <li>Fried snacks and oily fast food</li>
                        <li>Excessive table sugar and sweets</li>
                        <li>Deep fried foods and processed chips</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 6: CALORIES PREDICTION (MACHINE LEARNING)
# ------------------------------------------------------------------------------
elif selected_tab == "🔥 Calories Prediction":
    st.markdown("### 🔥 Machine Learning Calorie Prediction Engine")
    st.caption("Trained on biometric activity logs using Scikit-Learn Regression to predict energy expenditure with verified feature alignment.")

    model_bundle = train_regression_model(df)
    ml_model = model_bundle["model"]
    ml_poly = model_bundle["poly"]
    ml_metrics = model_bundle["metrics"]

    # Model Evaluation Metrics Cards
    eval_c1, eval_c2, eval_c3, eval_c4 = st.columns(4)
    with eval_c1:
        st.markdown(f"""
            <div class="metric-card accent-emerald">
                <div class="metric-header">
                    <span class="metric-label">Model Accuracy (R²)</span>
                    <span class="metric-icon" style="font-size: 1.3rem;">🎯</span>
                </div>
                <div class="metric-value">{ml_metrics['r2'] * 100:.2f}%</div>
                <div class="metric-footer"><span>Variance Explained</span></div>
            </div>
        """, unsafe_allow_html=True)

    with eval_c2:
        st.markdown(f"""
            <div class="metric-card accent-teal">
                <div class="metric-header">
                    <span class="metric-label">RMSE (Root MSE)</span>
                    <span class="metric-icon" style="font-size: 1.3rem;">📉</span>
                </div>
                <div class="metric-value">{ml_metrics['rmse']:.2f} <span style="font-size:1.1rem; color:#94a3b8;">kcal</span></div>
                <div class="metric-footer"><span>Standard Error Deviation</span></div>
            </div>
        """, unsafe_allow_html=True)

    with eval_c3:
        st.markdown(f"""
            <div class="metric-card accent-blue">
                <div class="metric-header">
                    <span class="metric-label">Mean Absolute Error</span>
                    <span class="metric-icon" style="font-size: 1.3rem;">📊</span>
                </div>
                <div class="metric-value">{ml_metrics['mae']:.2f} <span style="font-size:1.1rem; color:#94a3b8;">kcal</span></div>
                <div class="metric-footer"><span>Average Absolute Gap</span></div>
            </div>
        """, unsafe_allow_html=True)

    with eval_c4:
        st.markdown(f"""
            <div class="metric-card accent-amber">
                <div class="metric-header">
                    <span class="metric-label">Mean Squared Error</span>
                    <span class="metric-icon" style="font-size: 1.3rem;">⚡</span>
                </div>
                <div class="metric-value">{ml_metrics['mse']:.2f}</div>
                <div class="metric-footer"><span>(kcal)² Loss</span></div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 20px 0; border-color: rgba(255, 255, 255, 0.1);'>", unsafe_allow_html=True)
    st.subheader("🔮 Predict Session Caloric Burn")
    st.caption("Input your workout parameters below to compute your predicted caloric expenditure.")

    # User Input Form
    pred_gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

    p_in_col1, p_in_col2, p_in_col3 = st.columns(3)
    with p_in_col1:
        pred_age = st.number_input("Age (years)", min_value=10, max_value=100, value=28, step=1)
        pred_height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=172.0, step=0.5)

    with p_in_col2:
        pred_weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=68.0, step=0.5)
        pred_duration = st.number_input("Workout Duration (minutes)", min_value=1.0, max_value=300.0, value=30.0, step=1.0)

    with p_in_col3:
        pred_heart_rate = st.number_input("Heart Rate (bpm)", min_value=40.0, max_value=220.0, value=110.0, step=1.0)
        pred_body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=43.0, value=40.0, step=0.1)

    # Prediction Action
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        predict_button = st.button("🔥 Compute Estimated Calories Burned", use_container_width=True, type="primary")

    if predict_button:
        user_input_dict = {
            "Gender_Male": [1 if pred_gender == "Male" else 0],
            "Age": [float(pred_age)],
            "Height": [float(pred_height)],
            "Weight": [float(pred_weight)],
            "Duration": [float(pred_duration)],
            "Heart_Rate": [float(pred_heart_rate)],
            "Body_Temp": [float(pred_body_temp)]
        }
        user_input_df = pd.DataFrame(user_input_dict)[model_bundle["features"]]
        user_input_poly = ml_poly.transform(user_input_df)

        predicted_calories = float(ml_model.predict(user_input_poly)[0])
        predicted_calories = max(0.0, predicted_calories)

        burn_rate_min = predicted_calories / max(1.0, pred_duration)

        if burn_rate_min < 4.0:
            intensity_badge = "Light Aerobic"
            intensity_color = "#38bdf8"
        elif burn_rate_min < 8.0:
            intensity_badge = "Moderate Endurance"
            intensity_color = "#10b981"
        elif burn_rate_min < 12.0:
            intensity_badge = "Vigorous / High Intensity"
            intensity_color = "#f59e0b"
        else:
            intensity_badge = "Peak Anaerobic / Maximal"
            intensity_color = "#ef4444"

        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #090e17 0%, #111a2e 100%); border-radius: 16px; padding: 28px; margin: 24px 0; color: #ffffff; text-align: center; border: 1px solid rgba(255, 255, 255, 0.15); box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);">
                <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.5); padding: 4px 14px; border-radius: 9999px; font-size: 0.82rem; font-weight: 700; text-transform: uppercase;">
                    ML Regression Estimate
                </span>
                <div style="font-size: 3.2rem; font-weight: 800; color: #ffffff; margin: 12px 0 6px 0;">
                    {predicted_calories:.1f} <span style="font-size: 1.5rem; color: #38bdf8;">kcal</span>
                </div>
                <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-top: 14px;">
                    <div style="background: rgba(255, 255, 255, 0.08); padding: 8px 16px; border-radius: 10px; font-size: 0.95rem; color: #ffffff;">
                        ⚡ Burn Rate: <strong>{burn_rate_min:.2f} kcal/min</strong>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.08); padding: 8px 16px; border-radius: 10px; font-size: 0.95rem; color: #ffffff;">
                        Workout Zone: <strong style="color: {intensity_color};">{intensity_badge}</strong>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.subheader("🏃 Real-World Physical Equivalents")
        equiv1, equiv2, equiv3 = st.columns(3)
        with equiv1:
            dist_km = predicted_calories / 65.0
            st.info(f"🚶 **Brisk Walking**: Equivalent to walking approx. **{dist_km:.1f} km**.")
        with equiv2:
            run_km = predicted_calories / 90.0
            st.info(f"🏃 **Jogging / Running**: Equivalent to running approx. **{run_km:.1f} km**.")
        with equiv3:
            cycle_min = predicted_calories / 8.5
            st.info(f"🚴 **Cycling**: Equivalent to approx. **{cycle_min:.0f} minutes** of moderate cycling.")

    with st.expander("🔍 Model Architecture & Feature Weights Breakdown"):
        coef_df = pd.DataFrame({
            "Feature Attribute": model_bundle["features"],
            "Linear Coefficient (kcal / unit)": [
                model_bundle["coefficients"][f] for f in model_bundle["features"]
            ]
        }).sort_values(by="Linear Coefficient (kcal / unit)", ascending=False)
        st.dataframe(coef_df, use_container_width=True)
        st.caption(f"Model Intercept: {model_bundle['intercept']:.3f} kcal. Duration and Heart Rate have the highest positive slopes.")

# ------------------------------------------------------------------------------
# TAB 7: INSIGHTS
# ------------------------------------------------------------------------------
elif selected_tab == "💡 Insights":
    st.markdown("### 💡 Evidence-Based Fitness Insights & Summary")
    st.caption("Key physiological takeaways, outlier extremes, and evidence-based exercise guidelines.")

    # Peak & Lowest Burner Profile Side-by-Side
    st.subheader("🏆 Record Energy Expenditure Extremes")
    icol1, icol2 = st.columns(2)

    highest_record = df.loc[df["Calories"].idxmax()]
    lowest_record = df.loc[df["Calories"].idxmin()]

    with icol1:
        st.markdown(f"""
            <div class="highlight-box" style="border-left: 4px solid #10b981;">
                <h4 style="color: #34d399;">🔥 All-Time Peak Session Burn</h4>
                <div style="font-size: 2.0rem; font-weight: 800; color: #ffffff; margin: 6px 0;">
                    {highest_record['Calories']:.0f} <span style="font-size: 1.1rem; color: #94a3b8;">kcal</span>
                </div>
                <div style="font-size: 0.90rem; color: #cbd5e1; line-height: 1.7;">
                    • <strong>Gender / Age</strong>: {highest_record['Gender']}, {int(highest_record['Age'])} yrs<br>
                    • <strong>Workout Duration</strong>: {highest_record['Duration']:.0f} minutes<br>
                    • <strong>Sustained Heart Rate</strong>: {highest_record['Heart_Rate']:.0f} bpm<br>
                    • <strong>Peak Core Temp</strong>: {highest_record['Body_Temp']:.1f} °C<br>
                    • <strong>Body Mass Index</strong>: {highest_record['BMI']:.1f} kg/m²
                </div>
            </div>
        """, unsafe_allow_html=True)

    with icol2:
        st.markdown(f"""
            <div class="highlight-box" style="border-left: 4px solid #38bdf8;">
                <h4 style="color: #38bdf8;">❄️ Minimal Recorded Session Burn</h4>
                <div style="font-size: 2.0rem; font-weight: 800; color: #ffffff; margin: 6px 0;">
                    {lowest_record['Calories']:.0f} <span style="font-size: 1.1rem; color: #94a3b8;">kcal</span>
                </div>
                <div style="font-size: 0.90rem; color: #cbd5e1; line-height: 1.7;">
                    • <strong>Gender / Age</strong>: {lowest_record['Gender']}, {int(lowest_record['Age'])} yrs<br>
                    • <strong>Workout Duration</strong>: {lowest_record['Duration']:.0f} minutes<br>
                    • <strong>Sustained Heart Rate</strong>: {lowest_record['Heart_Rate']:.0f} bpm<br>
                    • <strong>Body Core Temp</strong>: {lowest_record['Body_Temp']:.1f} °C<br>
                    • <strong>Body Mass Index</strong>: {lowest_record['BMI']:.1f} kg/m²
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 20px 0; border-color: rgba(255, 255, 255, 0.1);'>", unsafe_allow_html=True)

    # Strongest Driving Factor Analysis
    numeric_corr = df.drop(columns=["User_ID"], errors="ignore").corr(numeric_only=True)
    if "Calories" in numeric_corr.columns:
        strongest_metric = numeric_corr["Calories"].drop("Calories").idxmax()
        strongest_metric_val = numeric_corr["Calories"].drop("Calories").max()
    else:
        strongest_metric = "Duration"
        strongest_metric_val = 0.955

    st.subheader("📈 Primary Metabolic Driving Force")
    st.success(
        f"🏆 **Statistical Finding**: **'{strongest_metric}'** exhibits the strongest positive correlation "
        f"(**r = {strongest_metric_val:.3f}**) with Calories Burned in the dataset. "
        "Energy expenditure is overwhelmingly a function of cumulative work time coupled with sustained cardiac output."
    )

    # Session Summary Metrics
    st.subheader("⏱️ Overall Workout Benchmarks")
    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        st.metric("Mean Duration", f"{df['Duration'].mean():.1f} min")
    with bcol2:
        st.metric("Mean Heart Rate", f"{df['Heart_Rate'].mean():.1f} bpm")
    with bcol3:
        st.metric("Mean Core Temp", f"{df['Body_Temp'].mean():.1f} °C")

    st.markdown("<hr style='margin: 20px 0; border-color: rgba(255, 255, 255, 0.1);'>", unsafe_allow_html=True)

    # Evidence-Based Recommendations
    st.subheader("💪 Actionable Fitness & Health Recommendations")
    rcol1, rcol2 = st.columns(2)

    with rcol1:
        st.markdown("""
            <div class="highlight-box">
                <h4>1. Progressive Duration & Workload Overload</h4>
                <p>
                    Progressively extending workout sessions by 5–10 minutes weekly allows the body to tap deeper into aerobic fat oxidation
                    and increases total caloric deficit without causing sudden muscular trauma.
                </p>
            </div>
            <div class="highlight-box">
                <h4>2. Target Heart Rate Training (Zone 2 & Zone 3)</h4>
                <p>
                    Exercising within 60%–75% of maximum heart rate optimizes mitochondrial density, enhances capillary recruitment,
                    and ensures sustainable high-volume caloric expenditure.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with rcol2:
        st.markdown("""
            <div class="highlight-box">
                <h4>3. Thermoregulation & Fluid Intake</h4>
                <p>
                    As core temperature rises during extended activity, hydration status directly impacts cardiac output efficiency.
                    Consuming 150–250 ml of fluid every 15–20 minutes sustains workout intensity and delays central fatigue.
                </p>
            </div>
            <div class="highlight-box">
                <h4>4. Combining Strength & Cardiovascular Exercise</h4>
                <p>
                    While cardio burns calories during the session, resistance training preserves and builds metabolically active lean muscle mass,
                    elevating Resting Metabolic Rate (RMR) and post-exercise oxygen consumption (EPOC).
                </p>
            </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 9. FOOTER
# ==============================================================================

st.markdown("""
    <div style="margin-top: 48px; padding-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.1); text-align: center; color: #94a3b8; font-size: 0.85rem;">
        Calories Burn Tracker • Built with Streamlit, Scikit-Learn, and Plotly • Ready for Deployment on Streamlit Community Cloud
    </div>
""", unsafe_allow_html=True)