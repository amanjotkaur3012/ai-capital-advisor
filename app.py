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
import google.generativeai as genai
from fpdf import FPDF
import datetime

# ----------------------------------------------------
# 1. ARCHITECTURE & UI THEME
# ----------------------------------------------------
st.set_page_config(page_title="STRATOS | Capital Advisor", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Public+Sans:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'Public Sans', sans-serif; }
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    
    /* Metrics Card */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    
    .main-title {
        font-size: 32px; font-weight: 800; letter-spacing: -1px;
        background: linear-gradient(90deg, #10b981, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    
    .status-tag {
        padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
        background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid #10b981;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. FINANCIAL CORE LOGIC
# ----------------------------------------------------

def calculate_raroc(row, risk_free_rate):
    """Calculate Risk-Adjusted Return on Capital."""
    expected_return = row['Investment_Capital'] * (row['Pred_ROI'] / 100)
    # Economic Capital is modeled as a function of risk score (1-10)
    economic_capital = row['Investment_Capital'] * (row['Risk_Score'] / 10)
    if economic_capital == 0: return 0
    return (expected_return - (row['Investment_Capital'] * risk_free_rate)) / economic_capital

def optimize_portfolio(df, budget, risk_cap):
    """LP Solver using PuLP to maximize Aggregate NPV under Risk and Budget constraints."""
    prob = pulp.LpProblem("Capital_Allocation", pulp.LpMaximize)
    xs = pulp.LpVariable.dicts("Project", df.index, cat=pulp.LpBinary)
    
    # Objective: Maximize Total NPV
    prob += pulp.lpSum([df.loc[i, 'NPV'] * xs[i] for i in df.index])
    
    # Constraint 1: Budget
    prob += pulp.lpSum([df.loc[i, 'Investment_Capital'] * xs[i] for i in df.index]) <= budget
    
    # Constraint 2: Portfolio Weighted Risk Score Cap
    prob += pulp.lpSum([df.loc[i, 'Risk_Score'] * xs[i] for i in df.index]) <= risk_cap * pulp.lpSum([xs[i] for i in df.index])
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    df['Selected'] = [pulp.value(xs[i]) for i in df.index]
    return df

# ----------------------------------------------------
# 3. DATA GENERATION (MOCK DATA)
# ----------------------------------------------------
@st.cache_data
def load_sample_data():
    hist_data = pd.DataFrame({
        'Investment_Capital': np.random.uniform(100000, 1000000, 100),
        'Risk_Score': np.random.uniform(1, 10, 100),
        'Strategic_Alignment': np.random.uniform(1, 10, 100),
        'Market_Volatility': np.random.uniform(0.1, 0.5, 100),
        'Actual_ROI': np.random.uniform(5, 25, 100)
    })
    
    projects = pd.DataFrame({
        'Project_ID': [f"EXT-{i:03d}" for i in range(1, 16)],
        'Department': np.random.choice(['FinTech', 'ESG', 'Core Ops', 'R&D'], 15),
        'Investment_Capital': [250000, 500000, 150000, 800000, 300000, 450000, 200000, 700000, 100000, 900000, 350000, 600000, 250000, 400000, 550000],
        'Risk_Score': [3.2, 7.5, 2.1, 8.9, 4.5, 5.0, 3.8, 6.2, 1.5, 9.1, 4.0, 5.5, 3.0, 6.0, 7.0],
        'Strategic_Alignment': [9, 6, 8, 10, 7, 5, 9, 8, 4, 10, 6, 7, 8, 5, 9],
        'Market_Volatility': [0.12, 0.25, 0.10, 0.35, 0.18, 0.20, 0.15, 0.22, 0.08, 0.40, 0.19, 0.21, 0.14, 0.23, 0.28]
    })
    return hist_data, projects

# ----------------------------------------------------
# 4. MAIN INTERFACE
# ----------------------------------------------------
def main():
    st.markdown('<p class="main-title">STRATOS | AI Capital Allocation</p>', unsafe_allow_html=True)
    st.markdown('<span class="status-tag">Institutional Grade Analytics</span>', unsafe_allow_html=True)
    st.write("")

    # Sidebar Controls
    st.sidebar.header("Economic Parameters")
    rf_rate = st.sidebar.slider("Risk Free Rate (%)", 0.0, 8.0, 4.2) / 100
    mkt_premium = st.sidebar.slider("Equity Risk Premium (%)", 3.0, 10.0, 5.5) / 100
    total_budget = st.sidebar.number_input("Total Capital Pool (USD)", value=2500000, step=100000)
    risk_tolerance = st.sidebar.slider("Max Avg Risk Score", 1.0, 10.0, 5.5)
    
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")

    # Load Data
    hist, projects = load_sample_data()
    
    # AI ML Forecasting (ROI Prediction)
    model = GradientBoostingRegressor()
    features = ['Investment_Capital', 'Risk_Score', 'Strategic_Alignment', 'Market_Volatility']
    model.fit(hist[features], hist['Actual_ROI'])
    projects['Pred_ROI'] = model.predict(projects[features])
    
    # Finance Logic: NPV & RAROC
    # NPV approximated as: (Investment * Pred_ROI%) / WACC
    wacc = rf_rate + 1.2 * mkt_premium # Simplified CAPM
    projects['NPV'] = (projects['Investment_Capital'] * (projects['Pred_ROI']/100)) / wacc
    projects['RAROC'] = projects.apply(lambda x: calculate_raroc(x, rf_rate), axis=1)

    # Optimization
    projects = optimize_portfolio(projects, total_budget, risk_tolerance)
    selected_df = projects[projects['Selected'] == 1]
    
    # ----------------------------------------------------
    # DASHBOARD TABS
    # ----------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "🔬 Risk Analysis", "🤖 AI Deal Room"])
    
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capital Deployed", f"${selected_df['Investment_Capital'].sum():,.0f}")
        c2.metric("Portfolio NPV", f"${selected_df['NPV'].sum():,.0f}")
        c3.metric("Avg RAROC", f"{selected_df['RAROC'].mean():.2%}")
        c4.metric("Risk Efficiency", f"{(selected_df['NPV'].sum()/selected_df['Investment_Capital'].sum()):.2f}x")
        
        # Treemap for Allocation
        fig_tree = px.treemap(selected_df, path=['Department', 'Project_ID'], 
                              values='Investment_Capital', color='Pred_ROI',
                              color_continuous_scale='RdYlGn', title="Capital Allocation by Vertical")
        st.plotly_chart(fig_tree, use_container_width=True)
        
        st.subheader("Optimized Investment Schedule")
        st.dataframe(selected_df[['Project_ID', 'Department', 'Investment_Capital', 'Pred_ROI', 'RAROC', 'NPV']], use_container_width=True)

    with tab2:
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Risk vs Return Scatter
            fig_scatter = px.scatter(projects, x="Risk_Score", y="Pred_ROI", color="Selected",
                                     size="Investment_Capital", hover_name="Project_ID",
                                     title="Efficient Frontier Visualization",
                                     color_discrete_map={1: "#10b981", 0: "#ef4444"})
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col_right:
            # Monte Carlo Simulation (Simple)
            st.write("**Portfolio Sensitivity (Monte Carlo)**")
            sim_returns = []
            for _ in range(1000):
                # Apply random shocks to ROI
                shock = np.random.normal(0, 0.05, len(selected_df))
                sim_returns.append((selected_df['Pred_ROI']/100 + shock).mean())
            
            fig_hist = px.histogram(sim_returns, nbins=50, title="Probability Distribution of Portfolio ROI",
                                   color_discrete_sequence=['#3b82f6'])
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.write("### AI Investment Memo Generator")
        if not gemini_key:
            st.info("💡 Enter your Gemini API Key in the sidebar to generate professional investment memos for these projects.")
        else:
            selected_id = st.selectbox("Select Project for Analysis", selected_df['Project_ID'])
            row = selected_df[selected_df['Project_ID'] == selected_id].iloc[0]
            
            if st.button("Generate Memo"):
                genai.configure(api_key=gemini_key)
                # Auto-select best available model
                model_ai = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"""
                Analyze this capital investment as a Senior Portfolio Manager:
                Project ID: {row['Project_ID']}
                Vertical: {row['Department']}
                Capital: ${row['Investment_Capital']}
                Predicted ROI: {row['Pred_ROI']:.2f}%
                Risk Score: {row['Risk_Score']}/10
                RAROC: {row['RAROC']:.2%}
                
                Provide a 3-point bulleted investment thesis and one 'Devil's Advocate' risk.
                """
                response = model_ai.generate_content(prompt)
                st.markdown(f"**Institutional Memo: {selected_id}**")
                st.write(response.text)

if __name__ == "__main__":
    main()
