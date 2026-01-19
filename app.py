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

# =========================================================
# 1. PAGE CONFIG & UI THEME
# =========================================================
st.set_page_config(
    page_title="FINCAP-AI | Capital Allocation Advisor",
    layout="wide",
    page_icon="📊"
)

st.markdown("""
<style>
.stApp { background-color: #0f172a; }
h1,h2,h3,h4 { color:#f8fafc; }
p,span,label { color:#cbd5e1; }
div[data-testid="stMetric"] {
    background:#1e293b; border:1px solid #334155; padding:15px; border-radius:6px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. DATABASE (SCENARIO STORAGE)
# =========================================================
DB_FILE = "fincap_ai.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS scenarios (
            id INTEGER PRIMARY KEY,
            name TEXT,
            budget REAL,
            npv REAL,
            roi REAL,
            projects INTEGER,
            wacc REAL,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_scenario(name, budget, npv, roi, count, wacc):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO scenarios VALUES (NULL,?,?,?,?,?, ?, datetime('now'))
    """, (name, budget, npv, roi, count, wacc))
    conn.commit()
    conn.close()

def get_scenarios():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM scenarios ORDER BY id DESC", conn)
    conn.close()
    return df

# =========================================================
# 3. AI MODEL (ROI PREDICTION)
# =========================================================
@st.cache_resource
def train_model(df):
    features = ["Investment", "Risk", "Strategy", "Market"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(df[features], df["ROI"])
    return model

# =========================================================
# 4. FINANCE FUNCTIONS (REAL WORLD)
# =========================================================
def calculate_npv(investment, roi, wacc, years=3):
    cashflow = investment * (roi / 100)
    dcf = sum([cashflow / ((1 + wacc) ** t) for t in range(1, years+1)])
    return dcf - investment

def optimize_portfolio(df, budget):
    prob = pulp.LpProblem("CapitalAllocation", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("Select", df.index, cat="Binary")

    prob += pulp.lpSum(df.loc[i, "NPV"] * x[i] for i in df.index)
    prob += pulp.lpSum(df.loc[i, "Investment"] * x[i] for i in df.index) <= budget

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    df["Selected"] = [int(x[i].value()) for i in df.index]
    return df

# =========================================================
# 5. DATA (DEMO, CAN BE REPLACED WITH UPLOAD)
# =========================================================
@st.cache_data
def load_data():
    hist = pd.DataFrame({
        "Investment": np.random.randint(500000, 5000000, 100),
        "Risk": np.random.uniform(1, 10, 100),
        "Strategy": np.random.uniform(1, 10, 100),
        "Market": np.random.uniform(0.8, 1.3, 100),
        "ROI": np.random.uniform(5, 25, 100)
    })

    prop = pd.DataFrame({
        "Project": [f"P{i:02d}" for i in range(1, 15)],
        "Investment": np.random.randint(500000, 4000000, 14),
        "Risk": np.random.uniform(1, 9, 14),
        "Strategy": np.random.uniform(3, 10, 14),
        "Market": np.random.uniform(0.9, 1.2, 14)
    })
    return hist, prop

hist, prop = load_data()
model = train_model(hist)
prop["Pred_ROI"] = model.predict(prop[["Investment","Risk","Strategy","Market"]])

# =========================================================
# 6. SIDEBAR (FINANCE CONTROLS)
# =========================================================
with st.sidebar:
    st.title("FINCAP-AI")
    st.caption("AI Capital Allocation Advisor")

    budget = st.number_input("Capital Budget (₹)", 5_000_000, 50_000_000, 15_000_000, step=500_000)
    wacc = st.slider("WACC (%)", 6.0, 18.0, 10.0) / 100
    max_risk = st.slider("Risk Appetite", 1.0, 10.0, 6.5)

    scenario_name = st.text_input("Scenario Name", "Base Case")

# =========================================================
# 7. FINANCE EVALUATION
# =========================================================
prop = prop[prop["Risk"] <= max_risk]
prop["NPV"] = prop.apply(lambda r: calculate_npv(r["Investment"], r["Pred_ROI"], wacc), axis=1)
prop = optimize_portfolio(prop, budget)
portfolio = prop[prop["Selected"] == 1]

# =========================================================
# 8. DASHBOARD
# =========================================================
st.title("Executive Capital Allocation Dashboard")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Projects Approved", len(portfolio))
k2.metric("Capital Deployed", f"₹{portfolio['Investment'].sum():,.0f}")
k3.metric("Avg ROI", f"{portfolio['Pred_ROI'].mean():.2f}%")
k4.metric("Portfolio NPV", f"₹{portfolio['NPV'].sum():,.0f}")

st.markdown("---")

st.subheader("Optimized Investment Plan")
st.dataframe(
    prop.style.format({
        "Investment": "₹{:,.0f}",
        "Pred_ROI": "{:.2f}%",
        "NPV": "₹{:,.0f}"
    }).background_gradient(subset=["NPV"], cmap="Greens"),
    use_container_width=True
)

fig = px.scatter(prop, x="Risk", y="Pred_ROI", size="Investment", color="Selected",
                 title="Risk vs Return Landscape", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 9. SCENARIO SAVE
# =========================================================
if st.button("💾 Save Scenario"):
    save_scenario(
        scenario_name,
        budget,
        portfolio["NPV"].sum(),
        portfolio["Pred_ROI"].mean(),
        len(portfolio),
        wacc
    )
    st.success("Scenario saved successfully!")

# =========================================================
# 10. DOWNLOAD
# =========================================================
csv = portfolio.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Investment Plan",
    csv,
    "fincap_ai_plan.csv",
    "text/csv"
)
