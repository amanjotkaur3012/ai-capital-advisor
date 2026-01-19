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
from sklearn.ensemble import GradientBoostingRegressor
import pulp
import sqlite3
from datetime import datetime

# ----------------------------------------------------
# 1. CORE ARCHITECTURE: FINCAP-AI ENGINE
# ----------------------------------------------------
class FinCapEngine:
    @staticmethod
    def compute_dcf_metrics(capex, cash_flow, life_years, wacc):
        """Calculates NPV and Profitability Index (PI)."""
        # Present Value of Cash Flows (Annuity Formula)
        pv_factor = (1 - (1 + wacc) ** -life_years) / wacc
        pv_of_inflows = cash_flow * pv_factor
        npv = pv_of_inflows - capex
        pi = pv_of_inflows / capex if capex > 0 else 0
        return round(npv, 2), round(pi, 2)

    @staticmethod
    def simulate_risk_exposure(base_npv, risk_beta):
        """Monte Carlo simulation for Value-at-Risk (VaR)."""
        sims = np.random.normal(base_npv, base_npv * (risk_beta * 0.1), 1000)
        return np.percentile(sims, 5) # 5% VaR (Worst Case)

# ----------------------------------------------------
# 2. BRANDED UI & THEME
# ----------------------------------------------------
st.set_page_config(page_title="FINCAP-AI | Intelligent Advisor", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    
    .stApp { background-color: #05070a; color: #e0e0e0; }
    
    /* Executive KPI Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #374151;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Finance-Grade Sidebar */
    section[data-testid="stSidebar"] { background-color: #0b0f19; border-right: 1px solid #1f2937; }
    
    .main-title {
        font-family: 'Roboto Mono', monospace;
        color: #60a5fa;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: 2px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: GOVERNANCE & CONSTRAINTS ---
with st.sidebar:
    st.markdown("<div class='main-title'>FINCAP-AI</div>", unsafe_allow_html=True)
    st.caption("Intelligent Capital Allocation Advisor")
    st.divider()
    
    auth_token = st.text_input("Institutional Token", type="password", value="ADMIN_SECURE")
    st.divider()
    
    st.subheader("Capital Controls")
    total_budget = st.number_input("Total Allocation Pool (₹)", value=100000000, step=5000000)
    wacc_hurdle = st.slider("Hurdle Rate (WACC %)", 5.0, 20.0, 11.5) / 100
    risk_buffer = st.slider("Contingency Reserve (%)", 0, 20, 5) / 100
    
    st.divider()
    st.info(f"Available Capital: ₹{(total_budget * (1-risk_buffer))/1e6:.1f}M")

# ----------------------------------------------------
# 3. ANALYTIC WORKSPACE
# ----------------------------------------------------
tabs = st.tabs(["🏛️ Allocation Strategy", "📊 Risk Frontier", "📄 Portfolio Audit"])

# Mock Project Data with Advanced Finance Attributes
df = pd.DataFrame({
    "Project_ID": [f"PRJ-{i:03}" for i in range(1, 9)],
    "Sector": ["Technology", "Energy", "Healthcare", "Infrastructure", "Technology", "Consumer", "Healthcare", "Energy"],
    "Capex": [15M, 35M, 12M, 40M, 18M, 10M, 25M, 30M],
    "Annual_Inflow": [4.2M, 8.5M, 3.1M, 9.2M, 4.8M, 2.8M, 6.5M, 7.2M],
    "Life_Span": [5, 10, 4, 12, 6, 4, 7, 8],
    "Beta_Risk": [1.4, 0.9, 1.1, 0.8, 1.5, 1.0, 1.2, 0.95]
})

# Engineering Metrics
df['NPV'], df['PI'] = zip(*df.apply(lambda x: FinCapEngine.compute_dcf_metrics(x['Capex'], x['Annual_Inflow'], x['Life_Span'], wacc_hurdle), axis=1))
df['VaR_WorstCase'] = df.apply(lambda x: FinCapEngine.simulate_risk_exposure(x['NPV'], x['Beta_Risk']), axis=1)

# Linear Programming: Binary Integer Optimization
prob = pulp.LpProblem("Capital_Rationing", pulp.LpMaximize)
select = pulp.LpVariable.dicts("Project", df.index, cat='Binary')
# Objective: Maximize Total Portfolio NPV
prob += pulp.lpSum([df.loc[i, 'NPV'] * select[i] for i in df.index])
# Constraint: Net Investment <= Adjusted Budget
prob += pulp.lpSum([df.loc[i, 'Capex'] * select[i] for i in df.index]) <= total_budget * (1 - risk_buffer)
prob.solve(pulp.PULP_CBC_CMD(msg=0))

df['Decision'] = [int(select[i].varValue) for i in df.index]

with tabs[0]:
    st.header("Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    funded = df[df['Decision'] == 1]
    
    col1.metric("Projects Selected", f"{len(funded)} / {len(df)}")
    col2.metric("Capital Utilization", f"₹{funded['Capex'].sum()/1e6:.1f}M", f"{funded['Capex'].sum()/total_budget*100:.1f}%")
    col3.metric("Projected Wealth Creation", f"₹{funded['NPV'].sum()/1e6:.1f}M")
    col4.metric("Avg. Profitability Index", f"{funded['PI'].mean():.2f}")

    st.subheader("Recommended Capital Schedule")
    st.dataframe(df.style.background_gradient(subset=['PI', 'Decision'], cmap='Blues'), use_container_width=True)
    
    

with tabs[1]:
    st.header("Financial Risk Topology")
    
    # Efficient Frontier Visualization
    fig = px.scatter(df, x="Beta_Risk", y="PI", size="Capex", color="Decision", 
                     symbol="Sector", text="Project_ID",
                     title="Capital Efficiency (PI) vs. Systematic Risk (Beta)")
    
    fig.add_hline(y=1.0, line_dash="dash", line_color="#ef4444", annotation_text="Value Destruction Zone")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    
    

with tabs[2]:
    st.header("Audit-Ready Investment Memos")
    for _, row in funded.iterrows():
        with st.expander(f"📑 APPROVAL MEMO: {row['Project_ID']} | Sector: {row['Sector']}"):
            st.markdown(f"""
            **Fiscal Justification:**
            The project demonstrates a **Profitability Index of {row['PI']}**, indicating that for every 1 Rupee invested, the firm generates {row['PI']} Rupees in present value. 
            
            **Risk Sensitivity:**
            With a Beta of {row['Beta_Risk']}, the project has a 5% Value-at-Risk (VaR) of ₹{abs(row['VaR_WorstCase'])/1e6:.2f}M under extreme market stress.
            
            **Strategic Recommendation:** Proceed with full funding. Immediate NPV accretion expected upon deployment.
            """)
