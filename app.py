import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0d0d0f;
    color: #e8e3dc;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111114;
    border-right: 1px solid #2a2a30;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Syne', sans-serif;
    color: #f0c060;
}

/* Main header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #f0c060;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #888;
    font-weight: 300;
    margin-bottom: 2rem;
}

/* Cards */
.metric-card {
    background: #16161a;
    border: 1px solid #2a2a30;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

.metric-card h4 {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #555;
    margin: 0 0 0.4rem 0;
}

.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
}

.risk-low    { color: #4ade80; }
.risk-medium { color: #f0c060; }
.risk-high   { color: #f87171; }

/* Result banner */
.result-banner {
    border-radius: 14px;
    padding: 2rem 2.4rem;
    margin: 1.5rem 0;
    border: 1px solid;
}
.result-banner.safe {
    background: #0d1f14;
    border-color: #4ade80;
}
.result-banner.risk {
    background: #1f0d0d;
    border-color: #f87171;
}
.result-banner h2 {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
}
.result-banner p {
    margin: 0;
    color: #aaa;
    font-size: 0.95rem;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #2a2a30;
    margin: 1.5rem 0;
}

/* Input labels */
label { color: #ccc !important; }

/* Predict button */
div.stButton > button {
    background: #f0c060;
    color: #0d0d0f;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.5px;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.85; }

/* Section headers inside main */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #f0c060;
    margin-bottom: 0.8rem;
    margin-top: 1.6rem;
}

/* Feature importance bar */
.fi-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.fi-label {
    font-size: 0.8rem;
    color: #bbb;
    width: 200px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.fi-bar-bg {
    flex: 1;
    background: #222;
    border-radius: 4px;
    height: 8px;
}
.fi-bar-fill {
    height: 8px;
    border-radius: 4px;
    background: #f0c060;
}
.fi-pct {
    font-size: 0.75rem;
    color: #666;
    width: 38px;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)


# ── Load Pickle ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("hr_attrition_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    artifact = load_model()
    model         = artifact["model"]
    scaler        = artifact["scaler"]
    feature_names = artifact["feature_names"]
    model_loaded  = True
except FileNotFoundError:
    model_loaded = False


# ── Sidebar — Employee Inputs ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Employee Profile")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("**Personal**")
    age               = st.slider("Age", 18, 60, 35)
    gender            = st.selectbox("Gender", ["Male", "Female"])
    marital_status    = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    distance_from_home = st.slider("Distance from Home (km)", 1, 29, 10)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("**Job Details**")
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    job_role   = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ])
    job_level           = st.slider("Job Level", 1, 5, 2)
    job_satisfaction    = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
    job_involvement     = st.slider("Job Involvement (1–4)", 1, 4, 3)
    work_life_balance   = st.slider("Work-Life Balance (1–4)", 1, 4, 3)
    over_time           = st.selectbox("Works Overtime?", ["No", "Yes"])
    business_travel     = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    environment_satisfaction = st.slider("Environment Satisfaction (1–4)", 1, 4, 3)
    relationship_satisfaction = st.slider("Relationship Satisfaction (1–4)", 1, 4, 3)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("**Compensation & Experience**")
    monthly_income       = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=500)
    percent_salary_hike  = st.slider("Salary Hike % (last year)", 11, 25, 15)
    stock_option_level   = st.slider("Stock Option Level (0–3)", 0, 3, 1)
    education            = st.slider("Education Level (1–5)", 1, 5, 3)
    education_field      = st.selectbox("Education Field", [
        "Life Sciences", "Medical", "Marketing",
        "Technical Degree", "Human Resources", "Other"
    ])
    total_working_years  = st.slider("Total Working Years", 0, 40, 10)
    years_at_company     = st.slider("Years at Company", 0, 40, 5)
    years_in_current_role = st.slider("Years in Current Role", 0, 18, 3)
    years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 2)
    years_with_curr_manager    = st.slider("Years with Current Manager", 0, 17, 3)
    num_companies_worked       = st.slider("No. of Companies Worked", 0, 9, 2)
    training_times_last_year   = st.slider("Trainings Last Year", 0, 6, 3)
    performance_rating         = st.slider("Performance Rating (1–4)", 1, 4, 3)

    predict_btn = st.button("🔍 Predict Attrition")


# ── Main Panel ─────────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>HR Attrition Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Machine-learning powered employee retention intelligence</div>", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ **`hr_attrition_model.pkl` not found.** Place the pickle file in the same directory as `app.py` and restart.")
    st.stop()


# ── Build Input DataFrame ──────────────────────────────────────────────────────
def build_input():
    row = {
        "Age": age,
        "BusinessTravel": business_travel,
        "DailyRate": 800,           # default mid-value (not collected in UI)
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": 1 if gender == "Male" else 0,
        "HourlyRate": 65,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": 14000,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": 1 if over_time == "Yes" else 0,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
    }

    # One-hot for MaritalStatus (not in features but let's skip if absent)
    df = pd.DataFrame([row])

    # One-hot encode Department
    for dept in ["Research & Development", "Sales", "Human Resources"]:
        df[f"Department_{dept}"] = 1 if department == dept else 0

    # One-hot encode EducationField
    for ef in ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]:
        df[f"EducationField_{ef}"] = 1 if education_field == ef else 0

    # One-hot encode JobRole
    for jr in ["Sales Executive", "Research Scientist", "Laboratory Technician",
               "Manufacturing Director", "Healthcare Representative", "Manager",
               "Sales Representative", "Research Director", "Human Resources"]:
        df[f"JobRole_{jr}"] = 1 if job_role == jr else 0

    # Drop raw object columns
    df.drop(columns=["BusinessTravel", "MaritalStatus"], inplace=True, errors="ignore")

    # Align with model's expected features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    return df


# ── Default State ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='metric-card'>
        <h4>Model</h4>
        <div class='value' style='font-size:1.1rem; color:#f0c060;'>Random Forest</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <h4>Features</h4>
        <div class='value' style='color:#f0c060;'>{len(feature_names)}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <h4>Estimators</h4>
        <div class='value' style='color:#f0c060;'>{model.n_estimators}</div>
    </div>""", unsafe_allow_html=True)


# ── Prediction ─────────────────────────────────────────────────────────────────
if predict_btn:
    df_input  = build_input()
    X_scaled  = scaler.transform(df_input)
    pred      = model.predict(X_scaled)[0]
    proba     = model.predict_proba(X_scaled)[0]
    risk_pct  = round(proba[1] * 100, 1)

    # Risk level
    if risk_pct < 30:
        risk_label = "Low Risk"
        risk_class = "risk-low"
        banner_cls = "safe"
        emoji      = "✅"
        verdict    = "Likely to Stay"
        detail     = "This employee shows strong retention signals. Continue regular check-ins and career development support."
    elif risk_pct < 60:
        risk_label = "Medium Risk"
        risk_class = "risk-medium"
        banner_cls = "risk"
        emoji      = "⚠️"
        verdict    = "Monitor Closely"
        detail     = "There are moderate attrition signals. Consider a 1-on-1 conversation about satisfaction and career goals."
    else:
        risk_label = "High Risk"
        risk_class = "risk-high"
        banner_cls = "risk"
        emoji      = "🚨"
        verdict    = "At Risk of Leaving"
        detail     = "Strong attrition indicators detected. Immediate retention intervention is recommended."

    # Banner
    st.markdown(f"""
    <div class='result-banner {banner_cls}'>
        <h2>{emoji} {verdict}</h2>
        <p>{detail}</p>
    </div>""", unsafe_allow_html=True)

    # Metrics row
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Attrition Probability</h4>
            <div class='value {risk_class}'>{risk_pct}%</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Retention Probability</h4>
            <div class='value risk-low'>{round(proba[0]*100, 1)}%</div>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Risk Level</h4>
            <div class='value {risk_class}'>{risk_label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Top Predictive Factors</div>", unsafe_allow_html=True)

    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(10)
    max_imp = fi_df["Importance"].max()

    fi_html = ""
    for _, row_fi in fi_df.iterrows():
        pct  = round(row_fi["Importance"] / max_imp * 100)
        disp = round(row_fi["Importance"] * 100, 1)
        fi_html += f"""
        <div class='fi-row'>
            <div class='fi-label'>{row_fi['Feature']}</div>
            <div class='fi-bar-bg'><div class='fi-bar-fill' style='width:{pct}%'></div></div>
            <div class='fi-pct'>{disp}%</div>
        </div>"""
    st.markdown(fi_html, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Input summary table ───────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Input Summary</div>", unsafe_allow_html=True)
    summary = {
        "Age": age, "Gender": gender, "Department": department,
        "Job Role": job_role, "Monthly Income": f"${monthly_income:,}",
        "OverTime": over_time, "Job Satisfaction": job_satisfaction,
        "Work-Life Balance": work_life_balance, "Years at Company": years_at_company,
        "Total Working Years": total_working_years
    }
    st.dataframe(
        pd.DataFrame(summary.items(), columns=["Attribute", "Value"]),
        use_container_width=True,
        hide_index=True
    )

else:
    st.markdown("""
    <div style='margin-top:3rem; text-align:center; color:#444;'>
        <div style='font-size:3rem;'>🧠</div>
        <div style='font-family:Syne,sans-serif; font-size:1.1rem; margin-top:0.5rem;'>
            Fill in the employee profile on the left<br>and click <span style='color:#f0c060;'>Predict Attrition</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr class='divider'>
<div style='text-align:center; color:#333; font-size:0.75rem; padding-bottom:1rem;'>
    HR Attrition Predictor · Powered by Random Forest · For HR use only
</div>
""", unsafe_allow_html=True)