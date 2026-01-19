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

# ----------------------------------------------------
# 0. BACKEND CONFIGURATION
# ----------------------------------------------------
GEMINI_API_KEY = "AIzaSyDAz2r4IRQT5dv3zNq-uuRO7D2O86nNueE"
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ----------------------------------------------------
# 1. UI THEME
# ----------------------------------------------------
st.set_page_config(page_title="STRATOS | AI Capital Advisor", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0b0f19; }
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
    }
    .main-header {
        font-size: 30px; font-weight: 800; color: #ffffff;
        background: linear-gradient(90deg, #10b981, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. FINANCIAL MODELS
# ----------------------------------------------------

def black_scholes_roa(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def generate_pdf_report(selected_df, stats):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "STRATOS | Strategic Capital Allocation Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Portfolio Executive Summary", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 5, f"Total Capital Deployed: ${stats['budget']:,.0f}\nStrategic NPV: ${stats['npv']:,.0f}\nESG Rating: {stats['esg']:.2f}")
    return pdf.output(dest='S').encode('latin-1')

def run_simulation(df, budget, esg_hurdle, rf, vol_stress):
    wacc = rf + 0.06
    df['Adjusted_ROI'] = df['Pred_ROI'] * (1 - (df['Market_Volatility'] * vol_stress))
    df['RAROC'] = (df['Investment_Capital'] * (df['Adjusted_ROI']/100)) / (df['Investment_Capital'] * (df['Risk_Score']/10))
    df['ROA_Value'] = df.apply(lambda x: black_scholes_roa(x['Investment_Capital'] * 1.3, x['Investment_Capital'], 2.0, rf, x['Market_Volatility']), axis=1)
    df['Strategic_NPV'] = ((df['Investment_Capital'] * (df['Adjusted_ROI']/100)) / wacc) + df['ROA_Value']
    
    prob = pulp.LpProblem("Capital_Optimization", pulp.LpMaximize)
    choices = pulp.LpVariable.dicts("Select", df.index, cat=pulp.LpBinary)
    prob += pulp.lpSum([df.loc[i, 'Strategic_NPV'] * choices[i] for i in df.index])
    prob += pulp.lpSum([df.loc[i, 'Investment_Capital'] * choices[i] for i in df.index]) <= budget
    prob += pulp.lpSum([df.loc[i, 'ESG_Score'] * choices[i] for i in df.index]) >= esg_hurdle * pulp.lpSum([choices[i] for i in df.index])
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    df['Selected'] = [pulp.value(choices[i]) for i in df.index]
    return df

# ----------------------------------------------------
# 3. MAIN APP
# ----------------------------------------------------

def main():
    st.markdown('<div class="main-header">STRATOS | AI-Driven Capital Advisor</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Macro Environment")
        rf_rate = st.slider("Risk-Free Rate", 0.0, 8.0, 4.2) / 100
        vol_stress = st.slider("Volatility Shock", 0.0, 2.0, 1.0)
        st.header("Strategic Hurdles")
        budget = st.number_input("Capital Pool ($)", 1000000, 10000000, 4000000)
        esg_min = st.slider("Min. ESG Rating", 1, 10, 6)

    # Data Gen
    np.random.seed(42)
    data = {
        'Project_ID': [f"PRJ-{i:02d}" for i in range(1, 21)],
        'Department': np.random.choice(['ESG', 'Fintech', 'Infra', 'R&D'], 20),
        'Investment_Capital': np.random.choice([200000, 500000, 800000, 1200000], 20),
        'Risk_Score': np.random.uniform(2, 9, 20),
        'ESG_Score': np.random.uniform(3, 10, 20),
        'Market_Volatility': np.random.uniform(0.12, 0.35, 20),
        'Pred_ROI': np.random.uniform(7, 28, 20)
    }
    df = pd.DataFrame(data)
    
    df = run_simulation(df, budget, esg_min, rf_rate, vol_stress)
    selected = df[df['Selected'] == 1]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capital Utilized", f"${selected['Investment_Capital'].sum():,.0f}")
    c2.metric("Portfolio NPV", f"${selected['Strategic_NPV'].sum():,.0f}")
    c3.metric("Avg ESG Impact", f"{selected['ESG_Score'].mean():.1f}")
    c4.metric("RAROC", f"{selected['RAROC'].mean():.1%}")

    t1, t2, t3 = st.tabs(["Strategic Frontier", "Risk Waterfall", "AI Investment Memo"])
    
    with t1:
        st.subheader("Value vs ESG Compliance")
        fig = px.scatter(df, x="ESG_Score", y="Strategic_NPV", size="Investment_Capital", 
                         color="Selected", color_discrete_map={1: '#10b981', 0: '#374151'})
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.subheader("Financial Sensitivity")
        base_val = selected['Strategic_NPV'].sum()
        risk_drag = -(selected['Investment_Capital'].sum() * selected['Market_Volatility'].mean() * vol_stress)
        fig_wat = go.Figure(go.Waterfall(
            orientation = "v", x = ["Baseline", "Volatility", "Stressed NPV"],
            y = [base_val, risk_drag, 0], measure = ["relative", "relative", "total"]
        ))
        st.plotly_chart(fig_wat, use_container_width=True)

    with t3:
        st.subheader("AI Institutional Intelligence")
        target_p = st.selectbox("Select Project", selected['Project_ID'])
        row = selected[selected['Project_ID'] == target_p].iloc[0]
        
        if st.button("Generate Memo"):
            with st.spinner("Consulting AI..."):
                try:
                    # Robust model discovery
                    model_list = genai.list_models()
                    available = [m.name for m in model_list if 'generateContent' in m.supported_generation_methods]
                    
                    # Choose model intelligently
                    chosen_model = "gemini-1.5-flash"
                    if f"models/{chosen_model}" not in available:
                        chosen_model = available[0].replace("models/", "")

                    model = genai.GenerativeModel(chosen_model)
                    prompt = f"Analyze project {target_p} with NPV ${row['Strategic_NPV']:.2f} and ESG {row['ESG_Score']}. Write a 2-sentence thesis."
                    response = model.generate_content(prompt)
                    st.info(f"Analysis via {chosen_model.upper()}")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"AI Error: {str(e)}")

        st.write("---")
        stats = {'budget': selected['Investment_Capital'].sum(), 'npv': selected['Strategic_NPV'].sum(), 'esg': selected['ESG_Score'].mean()}
        pdf_bytes = generate_pdf_report(selected, stats)
        st.download_button("📥 Download Report (PDF)", data=pdf_bytes, file_name="Report.pdf")

if __name__ == "__main__":
    main()
