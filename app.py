# =========================================================
# FINCAP-AI | Intelligent Capital Allocation Advisor
# MSc Finance & Analytics Live Project
# Author: Aman
# =========================================================

# =========================================================
# FINCAP-AI | Intelligent Capital Allocation Advisor
# MSc Finance & Analytics Live Project - Strategic Edition
# Author: Aman
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import pulp
import google.generativeai as genai
from scipy.stats import norm

# ----------------------------------------------------
# 0. BACKEND CONFIGURATION & API
# ----------------------------------------------------
GEMINI_API_KEY = "AIzaSyBEFV8Q9A9DvRUMzB9tD3KlUPvsv_60j60"
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="STRATOS QUANT | MSc Capital Advisor", layout="wide")

# Bloomberg Terminal Aesthetic CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0b0f19; }
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    .main-title { 
        font-size: 34px; font-weight: 800; letter-spacing: -1.5px; 
        background: linear-gradient(90deg, #10b981, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    div[data-testid="stMetric"] {
        background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 22px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 1. QUANTITATIVE FINANCE ENGINES
# ----------------------------------------------------

def black_scholes_roa(S, K, T, r, sigma):
    """Calculates Real Option Value using Black-Scholes Formula."""
    if T <= 0 or sigma <= 0 or S <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def run_optimization(df, budget, esg_hurdle):
    """Mixed Integer Programming to maximize Portfolio Strategic Value."""
    prob = pulp.LpProblem("Portfolio_Opt", pulp.LpMaximize)
    xs = pulp.LpVariable.dicts("Select", df.index, cat=pulp.LpBinary)
    prob += pulp.lpSum([df.loc[i, 'Strategic_Value'] * xs[i] for i in df.index])
    prob += pulp.lpSum([df.loc[i, 'Investment_Capital'] * xs[i] for i in df.index]) <= budget
    prob += pulp.lpSum([df.loc[i, 'ESG_Total'] * xs[i] for i in df.index]) >= esg_hurdle * pulp.lpSum([xs[i] for i in df.index])
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    df['Selected'] = [pulp.value(xs[i]) for i in df.index]
    return df

# ----------------------------------------------------
# 2. MAIN APPLICATION LOGIC
# ----------------------------------------------------

def main():
    st.markdown('<p class="main-title">STRATOS | Capital Allocation Advisor</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8b949e; margin-bottom: 30px;">MSc Finance & Analytics | Advanced Strategic Decision System</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Data Management")
        up_file = st.file_uploader("Upload Project Dataset (CSV)", type="csv")
        use_demo = st.checkbox("Load MSc Institutional Demo", value=not up_file)
        
        st.header("Financial Benchmarks")
        rf_rate = st.slider("Risk-Free Rate (Rf)", 0.0, 8.0, 4.2) / 100
        budget = st.number_input("Capital Constraint ($)", value=5500000, step=500000)
        esg_min = st.slider("Global ESG Hurdle", 1, 10, 6)

    # Dataset Generation with ESG Pillar Breakdown
    if up_file:
        df = pd.read_csv(up_file)
    else:
        np.random.seed(42)
        df = pd.DataFrame({
            'Project_ID': [f"FIN-{i:03d}" for i in range(1, 26)],
            'Department': np.random.choice(['ESG Fintech', 'R&D', 'Infra', 'Digital Assets'], 25),
            'Investment_Capital': np.random.choice([250000, 500000, 1000000, 1500000], 25),
            'Risk_Score': np.random.uniform(2.0, 9.5, 25),
            'E_Pillar': np.random.uniform(1, 10, 25),
            'S_Pillar': np.random.uniform(1, 10, 25),
            'G_Pillar': np.random.uniform(1, 10, 25),
            'Volatility': np.random.uniform(0.12, 0.45, 25),
            'Strategic_Alignment': np.random.randint(4, 11, 25)
        })
        df['ESG_Total'] = (df['E_Pillar'] + df['S_Pillar'] + df['G_Pillar']) / 3
        df['Actual_ROI'] = (df['Strategic_Alignment'] * 1.8) - (df['Risk_Score'] * 0.5) + 11 + np.random.normal(0, 1.5, 25)

    # ML Forecaster
    feats = ['Investment_Capital', 'Risk_Score', 'ESG_Total', 'Volatility', 'Strategic_Alignment']
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(df[feats], df['Actual_ROI'])
    df['Pred_ROI'] = rf_model.predict(df[feats])
    feat_imp = pd.DataFrame({'Feature': feats, 'Importance': rf_model.feature_importances_})

    # Quantitative Metrics (Sharpe Score Addition)
    wacc = rf_rate + 0.06 
    df['ROA_Value'] = df.apply(lambda x: black_scholes_roa(x['Investment_Capital']*1.35, x['Investment_Capital'], 2, rf_rate, x['Volatility']), axis=1)
    df['NPV'] = (df['Investment_Capital'] * (df['Pred_ROI']/100)) / wacc
    df['Strategic_Value'] = df['NPV'] + df['ROA_Value']
    df['PI'] = df['Strategic_Value'] / df['Investment_Capital']
    # Sharpe Score = (Expected Return - Rf) / Volatility
    df['Sharpe_Score'] = ((df['Pred_ROI']/100) - rf_rate) / df['Volatility']

    # Solver Implementation
    df = run_optimization(df, budget, esg_min)
    selected = df[df['Selected'] == 1]

    # --- DASHBOARD TABS ---
    t1, t2, t3, t4, t5 = st.tabs(["🚀 Summary", "🌍 ESG Pillars", "📅 Lifecycle", "🛡️ Risk & Sharpe", "🤖 AI Thesis"])
    
    with t1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capital Allocated", f"${selected['Investment_Capital'].sum():,.0f}")
        c2.metric("Portfolio Strategic Value", f"${selected['Strategic_Value'].sum():,.0f}")
        c3.metric("Avg Portfolio Sharpe", f"{selected['Sharpe_Score'].mean():.2f}")
        c4.metric("Agg. PI", f"{selected['PI'].mean():.2f}x")
        
        st.write("### 📋 Portfolio Investment Schedule")
        st.dataframe(selected[['Project_ID', 'Department', 'Investment_Capital', 'Pred_ROI', 'Sharpe_Score', 'ESG_Total']].style.background_gradient(cmap='Greens'))
        
        if st.button("🤖 Explain My Portfolio Health"):
            with st.spinner("Analyzing..."):
                try:
                    m_list = genai.list_models()
                    available = [m.name for m in m_list if 'generateContent' in m.supported_generation_methods]
                    m_name = "gemini-1.5-flash" if f"models/gemini-1.5-flash" in available else available[0].replace("models/", "")
                    model = genai.GenerativeModel(m_name)
                    summary_prompt = f"CEO Brief: Budget used ${selected['Investment_Capital'].sum()}, Sharpe {selected['Sharpe_Score'].mean():.2f}, PI {selected['PI'].mean():.2f}. Comment on efficiency."
                    response = model.generate_content(summary_prompt)
                    st.info(response.text)
                except Exception as e: st.error("AI Cooling down.")

    with t2:
        st.subheader("ESG Pillar Breakdown Analysis")
        # Visualizing Environmental, Social, and Governance individually
        pillar_df = selected[['Project_ID', 'E_Pillar', 'S_Pillar', 'G_Pillar']].melt(id_vars='Project_ID', var_name='Pillar', value_name='Score')
        fig_esg = px.bar(pillar_df, x='Project_ID', y='Score', color='Pillar', barmode='group', color_discrete_sequence=['#10b981', '#3b82f6', '#fbbf24'])
        st.plotly_chart(fig_esg, use_container_width=True)
        st.info("💡 Strategic Insights: Analyzes the trade-off between sustainability dimensions. High Environmental scores often correlate with Green Tax benefits.")

    with t3:
        st.subheader("Project Lifecycle Phasing (3-Year Horizon)")
        # Simulating a Capex distribution over Year 1, 2, and 3
        selected['Year_1'] = selected['Investment_Capital'] * 0.5
        selected['Year_2'] = selected['Investment_Capital'] * 0.3
        selected['Year_3'] = selected['Investment_Capital'] * 0.2
        phase_df = selected[['Project_ID', 'Year_1', 'Year_2', 'Year_3']].melt(id_vars='Project_ID', var_name='Year', value_name='Capex')
        fig_phase = px.bar(phase_df, x='Year', y='Capex', color='Project_ID', title="Portfolio Cash Outflow Phasing", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_phase, use_container_width=True)

    with t4:
        st.subheader("Risk-Efficiency Analysis (Sharpe Score)")
        fig_sharpe = px.scatter(selected, x="Risk_Score", y="Sharpe_Score", size="Investment_Capital", color="Department",
                                hover_name="Project_ID", labels={"Sharpe_Score": "Risk-Adjusted Return (Sharpe)"}, title="Sharpe Efficiency vs. Project Risk")
        st.plotly_chart(fig_sharpe, use_container_width=True)
        
        # Portfolio Risk Metrics Snapshot
        st.markdown("---")
        mu_p = (selected['Pred_ROI'] / 100).mean()
        sigma_p = (selected['Pred_ROI'] / 100).std()
        VaR_95 = norm.ppf(0.05, mu_p, sigma_p)
        st.metric("Portfolio Value-at-Risk (95%)", f"{VaR_95:.2%}")

    with t5:
        st.subheader("Deep-Dive AI Investment Thesis")
        target = st.selectbox("Select Project for AI Expert Review", selected['Project_ID'])
        r = selected[selected['Project_ID'] == target].iloc[0]
        
        if st.button("🖋️ Generate Expert Memo"):
            with st.spinner("Accessing LLM Logic..."):
                try:
                    m_list = genai.list_models()
                    available = [m.name for m in m_list if 'generateContent' in m.supported_generation_methods]
                    m_name = "gemini-1.5-flash" if f"models/gemini-1.5-flash" in available else available[0].replace("models/", "")
                    model = genai.GenerativeModel(m_name)
                    prompt = f"Role: Senior Quant. Analyze {target}. Sharpe {r['Sharpe_Score']:.2f}, E: {r['E_Pillar']}, S: {r['S_Pillar']}, G: {r['G_Pillar']}. Is this high-quality capital usage?"
                    response = model.generate_content(prompt)
                    st.success(f"Analyst: {m_name.upper()}")
                    st.markdown(response.text)
                except Exception as e: st.error(f"Error: {e}")

    # Export Logic
    st.markdown("---")
    st.download_button("📥 Download Strategic Portfolio (CSV)", selected.to_csv(index=False), "Stratos_Portfolio.csv", "text/csv")

if __name__ == "__main__":
    main()
