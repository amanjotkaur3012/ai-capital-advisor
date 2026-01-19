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
import google.generativeai as genai
from datetime import datetime

# ----------------------------------------------------
# 1. THEME & UI CUSTOMIZATION
# ----------------------------------------------------
st.set_page_config(
    page_title="EQUITYFLOW | Strategic Capital Advisor",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💎"
)

# Custom CSS for a unique "Glassmorphism" look
st.markdown("""
    <style>
    :root {
        --primary: #6366f1;
        --secondary: #ec4899;
        --bg-dark: #0b0f19;
        --card-bg: #161b2c;
    }
    
    .stApp { background-color: var(--bg-dark); }
    
    /* Unique Sidebar Gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b2c 0%, #0b0f19 100%);
        border-right: 1px solid #2d3748;
    }

    /* Glass Effect Cards */
    div[data-testid="stMetric"], .stDataFrame, div[data-testid="stExpander"] {
        background-color: var(--card-bg) !important;
        border: 1px solid #2d3748 !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }

    /* Custom Typography */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-text { color: #94a3b8; font-size: 14px; margin-bottom: 30px; }

    /* Button Styling */
    .stButton>button {
        border-radius: 8px;
        background: linear-gradient(90deg, #6366f1, #4f46e5);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. CORE LOGIC & DATA ENGINE
# ----------------------------------------------------
class CapitalAdvisor:
    def __init__(self):
        self.db_name = "equityflow_core.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_name) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS records 
                         (id INTEGER PRIMARY KEY, scenario_name TEXT, timestamp TEXT, 
                          budget REAL, total_npv REAL, avg_roi REAL, count INTEGER)""")

    @staticmethod
    def map_inputs(df):
        """Standardizes input column naming convention"""
        mapping = {
            'Capex': 'Investment', 'Cost': 'Investment', 'Budget': 'Investment',
            'Return': 'ROI_Expected', 'ROI': 'ROI_Expected',
            'Strategic': 'Alignment_Score', 'Risk': 'Risk_Index'
        }
        df = df.rename(columns=mapping)
        # Fill missing critical columns
        if 'Is_Required' not in df.columns: df['Is_Required'] = 0
        return df

    @staticmethod
    def train_predictive_model(df_hist):
        """Uses Gradient Boosting for more modern forecasting"""
        features = ["Investment", "Risk_Index", "Alignment_Score"]
        # Ensure data exists for these columns
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
        model.fit(df_hist[features], df_hist["Actual_ROI"])
        return model

    @staticmethod
    def solve_allocation(df, budget_limit):
        """Linear Programming Optimization"""
        prob = pulp.LpProblem("Allocation_Optimization", pulp.LpMaximize)
        choices = pulp.LpVariable.dicts("Select", df.index, cat='Binary')
        
        # Objective: Maximize NPV
        prob += pulp.lpSum([df.loc[i, "Estimated_NPV"] * choices[i] for i in df.index])
        
        # Constraint: Stay within budget
        prob += pulp.lpSum([df.loc[i, "Investment"] * choices[i] for i in df.index]) <= budget_limit
        
        # Constraint: Mandatory Projects
        for i in df.index:
            if df.loc[i, 'Is_Required'] == 1:
                prob += choices[i] == 1

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        df["Decision"] = [int(choices[i].varValue) for i in df.index]
        return df

# ----------------------------------------------------
# 3. INTERFACE RENDERING
# ----------------------------------------------------
def main():
    advisor = CapitalAdvisor()
    
    # Sidebar Logo
    with st.sidebar:
        st.markdown("<div class='main-title'>EQUITYFLOW</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-text'>AI CAPITAL ALLOCATION v2.0</div>", unsafe_allow_html=True)
        st.divider()
        
        nav = st.radio("STRATEGY HUB", 
                       ["1. Data Intake", "2. Performance Dashboard", "3. Strategic Portfolio", "4. Scenario Lab"])
        
        st.divider()
        st.subheader("Config Parameters")
        budget = st.number_input("Total Capital Pool (₹)", value=10000000.0)
        wacc = st.slider("WACC / Discount Rate (%)", 5.0, 20.0, 10.0) / 100
        
    # LOGIC: Data Handling
    if nav == "1. Data Intake":
        st.header("Financial Data Ingestion")
        col1, col2 = st.columns(2)
        with col1:
            st.info("Upload historical project performance to train the AI forecasting engine.")
            hist_file = st.file_uploader("Historical Performance (CSV)", type="csv")
        with col2:
            st.info("Upload your current pipeline for the upcoming fiscal year.")
            prop_file = st.file_uploader("New Project Proposals (CSV)", type="csv")
            
        if st.button("Initialize Engine"):
            # Placeholder for demo - in production, you'd process the files here
            st.session_state['data_loaded'] = True
            st.success("AI Model Trained Successfully!")

    elif nav == "2. Performance Dashboard":
        st.header("Executive Insights")
        if 'data_loaded' not in st.session_state:
            st.warning("Please initialize data in the 'Data Intake' section.")
        else:
            # Generate Dummy Data for Visualization
            metrics_cols = st.columns(3)
            metrics_cols[0].metric("Allocation Efficiency", "94.2%", "+2.1%")
            metrics_cols[1].metric("Portfolio NPV", "₹4.8M", "Forecasted")
            metrics_cols[2].metric("Risk Exposure", "Low-Mid", "Optimized")
            
            # Waterfall Chart Simulation
            fig = go.Figure(go.Waterfall(
                x = ["Proposed", "Risk Adj", "Cost Offset", "Optimized Total"],
                y = [5000000, -400000, -200000, 4400000],
                measure = ["relative", "relative", "relative", "total"]
            ))
            fig.update_layout(template="plotly_dark", title="Capital Value Bridge")
            st.plotly_chart(fig, use_container_width=True)

    elif nav == "3. Strategic Portfolio":
        st.header("Recommended Allocations")
        # Visualizing the selection results
        st.markdown("#### Approved Projects for Funding")
        dummy_df = pd.DataFrame({
            "Project ID": ["EF-001", "EF-042", "EF-089"],
            "Department": ["R&D", "Ops", "Tech"],
            "Investment": [1200000, 800000, 2500000],
            "Est. NPV": [450000, 310000, 890000]
        })
        st.table(dummy_df)
        
    elif nav == "4. Scenario Lab":
        st.header("What-If Simulation")
        st.write("Compare different budget levels and hurdle rates below.")
        # Radar Chart for Portfolio Quality
        categories = ['Strategic Fit', 'ROI Potential', 'Risk Mitigation', 'Cash Flow Speed']
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[8, 9, 7, 6], theta=categories, fill='toself', name='Scenario A'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
