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
from scipy.stats import norm
from fpdf import FPDF
import base64

# ----------------------------------------------------
# 0. BACKEND CONFIGURATION
# ----------------------------------------------------
GEMINI_API_KEY = "AIzaSyDAz2r4IRQT5dv3zNq-uuRO7D2O86nNueE"
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ----------------------------------------------------
# 1. UI THEME: "MODERN ASSET MANAGEMENT"
# ----------------------------------------------------
st.set_page_config(page_title="STRATOS | AI Capital Advisor", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    
    /* KPI Cards */
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .main-header {
        font-size: 30px; font-weight: 800; color: #ffffff;
        background: linear-gradient(90deg, #10b981, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .sub-header { color: #8b949e; font-size: 14px; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. FINANCIAL MODELS (BLACK-SCHOLES & RAROC)
# ----------------------------------------------------

def black_scholes_roa(S, K, T, r, sigma):
    """Calculates 'Real Option to Expand' value for strategic projects."""
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def generate_pdf_report(selected_df, stats):
    """Generates a professional branded investment report."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "STRATOS | Strategic Capital Allocation Report", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(200, 10, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Portfolio Executive Summary", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 5, f"Total Capital Deployed: ${stats['budget']:,.0f}\nAggregate Strategic NPV: ${stats['npv']:,.0f}\nPortfolio ESG Rating: {stats['esg']:.2f}/10\nRisk-Adjusted Return (RAROC): {stats['raroc']:.2%}")
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Approved Investment Schedule", ln=True)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(40, 7, "ID", 1); pdf.cell(50, 7, "Dept", 1); pdf.cell(50, 7, "Capital", 1); pdf.cell(40, 7, "RAROC", 1); pdf.ln()
    pdf.set_font("Arial", '', 10)
    for i, r in selected_df.iterrows():
        pdf.cell(40, 7, str(r['Project_ID']), 1); pdf.cell(50, 7, str(r['Department']), 1); pdf.cell(50, 7, f"${r['Investment_Capital']:,.0f}", 1); pdf.cell(40, 7, f"{r['RAROC']:.1%}", 1); pdf.ln()
    
    return pdf.output(dest='S').encode('latin-1')

# ----------------------------------------------------
# 3. CORE ANALYTICS ENGINE
# ----------------------------------------------------

def run_simulation(df, budget, esg_hurdle, rf, vol_shock):
    # 1. Financial Metrics calculation
    wacc = rf + 0.06 # Risk-Free + Equity Risk Premium
    df['Adjusted_ROI'] = df['Pred_ROI'] * (1 - (df['Market_Volatility'] * vol_shock))
    df['RAROC'] = (df['Investment_Capital'] * (df['Adjusted_ROI']/100)) / (df['Investment_Capital'] * (df['Risk_Score']/10))
    
    # 2. Real Option Valuation (Strategic Flexibility)
    df['ROA_Value'] = df.apply(lambda x: black_scholes_roa(x['Investment_Capital'] * 1.3, x['Investment_Capital'], 2.0, rf, x['Market_Volatility']), axis=1)
    df['Strategic_NPV'] = ((df['Investment_Capital'] * (df['Adjusted_ROI']/100)) / wacc) + df['ROA_Value']
    
    # 3. Optimization (Integer Programming)
    prob = pulp.LpProblem("Capital_Optimization", pulp.LpMaximize)
    choices = pulp.LpVariable.dicts("Select", df.index, cat=pulp.LpBinary)
    prob += pulp.lpSum([df.loc[i, 'Strategic_NPV'] * choices[i] for i in df.index])
    prob += pulp.lpSum([df.loc[i, 'Investment_Capital'] * choices[i] for i in df.index]) <= budget
    prob += pulp.lpSum([df.loc[i, 'ESG_Score'] * choices[i] for i in df.index]) >= esg_hurdle * pulp.lpSum([choices[i] for i in df.index])
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    df['Selected'] = [pulp.value(choices[i]) for i in df.index]
    return df

# ----------------------------------------------------
# 4. MAIN APP INTERFACE
# ----------------------------------------------------

def main():
    st.markdown('<div class="main-header">STRATOS | AI-Driven Capital Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">MSc Finance & Analytics | Advanced Asset Allocation Framework</div>', unsafe_allow_html=True)
    
    # Sidebar Setup
    with st.sidebar:
        st.header("Macro Environment")
        rf_rate = st.slider("Risk-Free Rate (10Y Treasury)", 0.0, 8.0, 4.2) / 100
        vol_stress = st.slider("Market Volatility Shock", 0.0, 2.0, 1.0)
        
        st.header("Strategic Hurdles")
        budget = st.number_input("Capital Pool ($)", 1000000, 10000000, 4000000)
        esg_min = st.slider("Min. Portfolio ESG Rating", 1, 10, 6)
        st.write("---")
        st.caption("Backend: Black-Scholes ROA, MIP Solver, Gemini 1.5-Flash")

    # Generate Mock Projects (20 projects)
    np.random.seed(42)
    data = {
        'Project_ID': [f"PRJ-{i:02d}" for i in range(1, 21)],
        'Department': np.random.choice(['ESG/Green', 'Core Fintech', 'Infrastructure', 'R&D'], 20),
        'Investment_Capital': np.random.choice([200000, 500000, 800000, 1200000, 300000], 20),
        'Risk_Score': np.random.uniform(2, 9, 20),
        'ESG_Score': np.random.uniform(3, 10, 20),
        'Market_Volatility': np.random.uniform(0.12, 0.35, 20),
        'Pred_ROI': np.random.uniform(7, 28, 20)
    }
    df = pd.DataFrame(data)
    
    # Run Engine
    df = run_simulation(df, budget, esg_min, rf_rate, vol_stress)
    selected = df[df['Selected'] == 1]
    
    # KPI Display
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capital Utilized", f"${selected['Investment_Capital'].sum():,.0f}")
    c2.metric("Portfolio NPV", f"${selected['Strategic_NPV'].sum():,.0f}")
    c3.metric("Avg ESG Impact", f"{selected['ESG_Score'].mean():.1f}")
    c4.metric("Risk Eff. (RAROC)", f"{selected['RAROC'].mean():.1%}")

    # Tabs for Visualization
    t1, t2, t3 = st.tabs(["Strategic Frontier", "Risk Waterfall", "AI Investment Memo"])
    
    with t1:
        st.subheader("Profitability vs. ESG Compliance")
        
        fig = px.scatter(df, x="ESG_Score", y="Strategic_NPV", size="Investment_Capital", 
                         color="Selected", color_discrete_map={1: '#10b981', 0: '#374151'},
                         title="Efficient Frontier: Value vs ESG",
                         labels={"Strategic_NPV": "Strategic NPV (incl. Real Options)"})
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.subheader("Financial Sensitivity (Stress Test)")
        base_val = selected['Strategic_NPV'].sum()
        risk_drag = -(selected['Investment_Capital'].sum() * selected['Market_Volatility'].mean() * vol_stress)
        
        fig_wat = go.Figure(go.Waterfall(
            orientation = "v", x = ["Baseline", "Macro Volatility", "Stress-Tested NPV"],
            y = [base_val, risk_drag, 0], measure = ["relative", "relative", "total"],
            decreasing = {"marker":{"color":"#ef4444"}},
            increasing = {"marker":{"color":"#10b981"}},
            totals = {"marker":{"color":"#3b82f6"}}
        ))
        st.plotly_chart(fig_wat, use_container_width=True)

   with t3:
        st.subheader("AI Institutional Intelligence")
        target_p = st.selectbox("Select Project for AI Deep Dive", selected['Project_ID'])
        row = selected[selected['Project_ID'] == target_p].iloc[0]
        
        if st.button("Generate Memo"):
            with st.spinner("Consulting AI..."):
                try:
                    # DYNAMIC MODEL SELECTION
                    # This fetches the models actually available to your specific API key
                    available_models = [m.name for m in genai.list_models() 
                                      if 'generateContent' in m.supported_generation_methods]
                    
                    # Selection Priority: Flash 1.5 -> Pro 1.5 -> Any available
                    if 'models/gemini-1.5-flash' in available_models:
                        model_id = 'gemini-1.5-flash'
                    elif 'models/gemini-1.5-pro' in available_models:
                        model_id = 'gemini-1.5-pro'
                    else:
                        model_id = available_models[0].split('/')[-1]

                    model = genai.GenerativeModel(model_id)
                    
                    prompt = f"""
                    Act as a Senior Investment Analyst. 
                    Project: {target_p} in {row['Department']}. 
                    Finance Metrics: Strategic NPV ${row['Strategic_NPV']:.2f}, ESG Score {row['ESG_Score']}/10, RAROC {row['RAROC']:.2%}.
                    Market Context: 10Y Yield at {rf_rate*100}%, Volatility Shock Level {vol_stress}.
                    Write a professional 2-sentence investment thesis and 1 critical risk factor.
                    """
                    
                    response = model.generate_content(prompt)
                    st.info(f"Analysis via {model_id.upper()}:")
                    st.write(response.text)
                    
                except Exception as e:
                    st.error(f"AI Connection Error: {str(e)}")
                    st.warning("Hint: Check if 'Generative Language API' is enabled in your Google Cloud Console.")
        
        # Download Section
        st.write("---")
        stats = {
            'budget': selected['Investment_Capital'].sum(),
            'npv': selected['Strategic_NPV'].sum(),
            'esg': selected['ESG_Score'].mean(),
            'raroc': selected['RAROC'].mean()
        }
        pdf_bytes = generate_pdf_report(selected, stats)
        st.download_button("📥 Download Official Executive Report (PDF)", data=pdf_bytes, file_name="Investment_Report.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
