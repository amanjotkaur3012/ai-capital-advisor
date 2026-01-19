# =========================================================
# FINCAP-AI | Intelligent Capital Allocation Advisor
# MSc Finance & Analytics Live Project
# Author: Aman
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import pulp
import sqlite3
import google.generativeai as genai

# ----------------------------------------------------
# 1. Page Configuration & Theme
# ----------------------------------------------------
st.set_page_config(
    page_title="FINCAP-AI | AI Capital Allocation Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp { background: #0f172a; }

h1, h2, h3, h4 { color: #f8fafc !important; }
p, span, label { color: #cbd5e1 !important; }

section[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1e293b;
}

div[data-testid="stMetric"] {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Database
# ----------------------------------------------------
DB_FILE = "fincap.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS scenarios (
        id INTEGER PRIMARY KEY,
        name TEXT,
        date TEXT,
        budget REAL,
        npv REAL,
        roi REAL,
        projects INTEGER,
        wacc REAL
    )
    """)
    conn.commit()
    conn.close()

init_db()

# ----------------------------------------------------
# 3. Core Financial Logic
# ----------------------------------------------------
def train_model(df):
    features = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(df[features], df["Actual_ROI_Pct"])
    return model

def calculate_npv(row, wacc):
    cashflow = row["Investment_Capital"] * (1 + row["Pred_ROI"] / 100)
    years = max(row["Duration_Months"], 1) / 12
    return (cashflow / ((1 + wacc) ** years)) - row["Investment_Capital"]

def optimize_portfolio(df, budget):
    prob = pulp.LpProblem("CapitalPlan", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", df.index, cat="Binary")

    prob += pulp.lpSum(df.loc[i, "NPV"] * x[i] for i in df.index)
    prob += pulp.lpSum(df.loc[i, "Investment_Capital"] * x[i] for i in df.index) <= budget

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    df["Selected"] = [int(x[i].value()) for i in df.index]
    return df

# ----------------------------------------------------
# 4. Demo Data
# ----------------------------------------------------
@st.cache_data
def demo_data():
    hist = pd.DataFrame({
        "Investment_Capital": np.random.randint(5e5, 5e6, 50),
        "Duration_Months": np.random.randint(6, 36, 50),
        "Risk_Score": np.random.uniform(1, 10, 50),
        "Strategic_Alignment": np.random.uniform(1, 10, 50),
        "Market_Trend_Index": np.random.uniform(0.5, 1.5, 50),
        "Actual_ROI_Pct": np.random.uniform(5, 25, 50)
    })

    prop = pd.DataFrame({
        "Project_ID": [f"P{i:03d}" for i in range(1, 21)],
        "Department": np.random.choice(["IT", "R&D", "Ops", "Marketing"], 20),
        "Investment_Capital": np.random.randint(5e5, 5e6, 20),
        "Duration_Months": np.random.randint(6, 36, 20),
        "Risk_Score": np.random.uniform(1, 10, 20),
        "Strategic_Alignment": np.random.uniform(1, 10, 20),
        "Market_Trend_Index": np.random.uniform(0.5, 1.5, 20)
    })
    return hist, prop

# ----------------------------------------------------
# 5. Sidebar
# ----------------------------------------------------
with st.sidebar:
    st.markdown("## FINCAP-AI")
    st.caption("AI Capital Allocation Engine")
    st.divider()

    if "page" not in st.session_state:
        st.session_state.page = "Data Intake & Setup"

    pages = [
        "Data Intake & Setup",
        "Capital Allocation Overview",
        "Model Intelligence Lab",
        "Risk–Return Frontier",
        "Investment Selection Register",
        "Portfolio Topology Map",
        "Scenario Stress Lab",
        "Investment Committee Notes"
    ]

    st.session_state.page = st.radio("Navigation", pages)

    st.divider()
    budget = st.number_input("Total Budget (INR)", 1e6, 5e7, 1.5e7, step=5e5)
    wacc = st.slider("WACC (%)", 5.0, 15.0, 10.0) / 100

# ----------------------------------------------------
# 6. Data Loading
# ----------------------------------------------------
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if st.session_state.page == "Data Intake & Setup":
    st.title("Data Intake & System Setup")

    if st.button("Load Demo Data"):
        hist, prop = demo_data()
        model = train_model(hist)
        prop["Pred_ROI"] = model.predict(prop[[
            "Investment_Capital", "Duration_Months", "Risk_Score",
            "Strategic_Alignment", "Market_Trend_Index"
        ]])
        prop["NPV"] = prop.apply(lambda r: calculate_npv(r, wacc), axis=1)
        prop = optimize_portfolio(prop, budget)

        st.session_state.df = prop
        st.session_state.data_loaded = True
        st.success("Demo data loaded successfully")

# ----------------------------------------------------
# 7. Shared Guard
# ----------------------------------------------------
if st.session_state.page != "Data Intake & Setup" and not st.session_state.data_loaded:
    st.warning("Please load data first.")
    st.stop()

df = st.session_state.df.copy()
portfolio = df[df["Selected"] == 1]
rejected = df[df["Selected"] == 0]

# ----------------------------------------------------
# 8. Capital Allocation Overview
# ----------------------------------------------------
if st.session_state.page == "Capital Allocation Overview":
    st.title("Capital Allocation Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Projects Approved", len(portfolio))
    c2.metric("Capital Deployed", f"₹{portfolio['Investment_Capital'].sum()/1e6:.2f}M")
    c3.metric("Portfolio NPV", f"₹{portfolio['NPV'].sum()/1e6:.2f}M")

    fig = px.bar(portfolio, x="Department", y="Investment_Capital", color="Pred_ROI")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# 9. Model Intelligence Lab
# ----------------------------------------------------
elif st.session_state.page == "Model Intelligence Lab":
    st.title("Model Intelligence Lab")
    st.info("This section explains how the AI prioritizes investments using historical signals.")

    fig = px.scatter(df, x="Risk_Score", y="Pred_ROI", color="Selected")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# 10. Risk–Return Frontier
# ----------------------------------------------------
elif st.session_state.page == "Risk–Return Frontier":
    st.title("Risk–Return Frontier")

    fig = px.scatter(df, x="Risk_Score", y="Pred_ROI", color="NPV")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# 11. Investment Selection Register
# ----------------------------------------------------
elif st.session_state.page == "Investment Selection Register":
    st.title("Investment Selection Register")
    st.dataframe(portfolio, use_container_width=True)

# ----------------------------------------------------
# 12. Portfolio Topology Map
# ----------------------------------------------------
elif st.session_state.page == "Portfolio Topology Map":
    st.title("Portfolio Topology Map")

    fig = px.scatter_3d(
        portfolio,
        x="Strategic_Alignment",
        y="Pred_ROI",
        z="Risk_Score",
        color="NPV",
        size="Investment_Capital"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# 13. Scenario Stress Lab
# ----------------------------------------------------
elif st.session_state.page == "Scenario Stress Lab":
    st.title("Scenario Stress Lab")
    st.info("Adjust budget and WACC from sidebar to test resilience of portfolio.")

# ----------------------------------------------------
# 14. Investment Committee Notes
# ----------------------------------------------------
elif st.session_state.page == "Investment Committee Notes":
    st.title("Investment Committee Notes")
    for _, r in portfolio.iterrows():
        st.success(f"{r['Project_ID']} approved due to strong ROI and acceptable risk.")
