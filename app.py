# =========================================================
# STRATOS QUANT | Executive Decision Support System
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
GEMINI_API_KEY = "AIzaSyBE3o8PdFRbFoPhRBeQ8MvFR1ImNMpmzu4"
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="STRATOS QUANT", layout="wide")

# INSTITUTIONAL HIGH-CONTRAST CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; }
    .stApp { background-color: #0d1117; color: #f0f6fc; }
    .main-title { 
        font-size: 52px !important; font-weight: 800; letter-spacing: -2.5px; 
        background: linear-gradient(90deg, #58a6ff, #10b981);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-text { color: #8b949e; font-size: 20px; margin-bottom: 30px; }
    .section-header { 
        font-size: 28px; font-weight: 700; color: #ffffff; 
        border-left: 8px solid #238636; padding-left: 15px; margin: 40px 0 20px 0;
    }
    div[data-testid="stMetric"] {
        background: #161b22; border: 2px solid #30363d; border-radius: 12px; padding: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    div[data-testid="stMetricValue"] { font-size: 42px !important; color: #ffffff !important; font-weight: 800; }
    div[data-testid="stMetricLabel"] { font-size: 18px !important; color: #58a6ff !important; text-transform: uppercase; font-weight: 600; }
    .ai-insight-box {
        background: rgba(88, 166, 255, 0.15); border: 1px solid #58a6ff;
        padding: 25px; border-radius: 10px; color: #f0f6fc; margin: 20px 0;
        font-size: 17px; line-height: 1.6;
    }
    section[data-testid="stSidebar"] { background-color: #010409 !important; border-right: 2px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 1. QUANTITATIVE ENGINES
# ----------------------------------------------------

def black_scholes_roa(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def run_optimization(df, budget, esg_hurdle):
    prob = pulp.LpProblem("Portfolio_Opt", pulp.LpMaximize)
    xs = pulp.LpVariable.dicts("Select", df.index, cat=pulp.LpBinary)
    prob += pulp.lpSum([df.loc[i, 'Strategic_Value'] * xs[i] for i in df.index])
    prob += pulp.lpSum([df.loc[i, 'Investment_Capital'] * xs[i] for i in df.index]) <= budget
    prob += pulp.lpSum([df.loc[i, 'ESG_Score'] * xs[i] for i in df.index]) >= esg_hurdle * pulp.lpSum([xs[i] for i in df.index])
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    df['Selected'] = [pulp.value(xs[i]) for i in df.index]
    return df

# ----------------------------------------------------
# 2. APPLICATION EXECUTION
# ----------------------------------------------------

def main():
    with st.sidebar:
        st.markdown("<h1 style='color:#58a6ff; font-size:36px; font-weight:800;'>TERMINAL</h1>", unsafe_allow_html=True)
        nav = st.radio("SELECT ANALYTIC VIEW", 
                       ["SUMMARY", " ML INTELLIGENCE", " SENSITIVITY", " RISK MANAGEMENT", " INSTITUTIONAL THESIS"])
        
        st.markdown("---")
        st.header("DATA OPS")
        up_file = st.file_uploader("Upload Proposals", type="csv")
        use_demo = st.checkbox("Load Demo Data", value=not up_file)
        
        st.header("MARKET BENCHMARKS")
        rf_rate = st.slider("Risk-Free Rate (%)", 0.0, 8.0, 4.2) / 100
        budget = st.number_input("Capital Constraint ($)", value=5500000, step=500000)
        esg_min = st.slider("Sustainability Hurdle (ESG)", 1, 10, 6)

    # Dataset Setup
    if up_file:
        df = pd.read_csv(up_file)
    else:
        np.random.seed(42)
        df = pd.DataFrame({
            'Project_ID': [f"FIN-{i:03d}" for i in range(1, 26)],
            'Department': np.random.choice(['ESG Fintech', 'R&D', 'Infra', 'Digital Assets'], 25),
            'Investment_Capital': np.random.choice([200000, 500000, 750000, 1250000, 2000000], 25),
            'Risk_Score': np.random.uniform(2.0, 9.5, 25),
            'E_Score': np.random.uniform(1, 10, 25), 'S_Score': np.random.uniform(1, 10, 25), 'G_Score': np.random.uniform(1, 10, 25),
            'Volatility': np.random.uniform(0.12, 0.45, 25), 'Strategic_Alignment': np.random.randint(4, 11, 25),
            'Phase_1_Cap': np.random.uniform(0.2, 0.5, 25), 'Phase_2_Cap': np.random.uniform(0.2, 0.4, 25)
        })
        df['ESG_Score'] = (df['E_Score'] + df['S_Score'] + df['G_Score']) / 3
        df['Actual_ROI'] = (df['Strategic_Alignment'] * 1.8) - (df['Risk_Score'] * 0.5) + 11 + np.random.normal(0, 1.5, 25)

    # Calculation Layers
    feats = ['Investment_Capital', 'Risk_Score', 'ESG_Score', 'Volatility', 'Strategic_Alignment']
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(df[feats], df['Actual_ROI'])
    df['Pred_ROI'] = rf_model.predict(df[feats])
    
    wacc = rf_rate + 0.06 
    df['ROA_Value'] = df.apply(lambda x: black_scholes_roa(x['Investment_Capital']*1.35, x['Investment_Capital'], 2, rf_rate, x['Volatility']), axis=1)
    df['Strategic_Value'] = ((df['Investment_Capital'] * (df['Pred_ROI']/100)) / wacc) + df['ROA_Value']
    df['PI'] = df['Strategic_Value'] / df['Investment_Capital']
    df['Sharpe_Score'] = (df['Pred_ROI'] - (rf_rate * 100)) / (df['Volatility'] * 100)

    df = run_optimization(df, budget, esg_min)
    selected = df[df['Selected'] == 1]

    # TOP BRANDING
    st.markdown('<p class="main-title">STRATOS | Capital Allocation Advisor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">AI-Driven Multi-Criteria Decision Support</p>', unsafe_allow_html=True)

    # PAGE ROUTING
    if nav == "SUMMARY":
        st.markdown('<div class="section-header">PORTFOLIO AGGREGATE PERFORMANCE</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAPITAL DEPLOYED", f"${selected['Investment_Capital'].sum():,.0f}")
        c2.metric("STRATEGIC VALUE", f"${selected['Strategic_Value'].sum():,.0f}")
        c3.metric("ESG IMPACT", f"{selected['ESG_Score'].mean():.1f}/10")
        c4.metric("VALUE INDEX (PI)", f"{selected['PI'].mean():.2f}x")
        
        st.markdown('<div class="section-header">FUNDING SCHEDULE</div>', unsafe_allow_html=True)
        st.dataframe(selected[['Project_ID', 'Department', 'Investment_Capital', 'Pred_ROI', 'PI', 'ESG_Score']].style.background_gradient(cmap='Greens'), use_container_width=True)
        
        if st.button("Interpret Summary"):
            with st.spinner("AI analyzing balance..."):
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    res = model.generate_content(f"Explain this portfolio summary to a CEO: Budget ${selected['Investment_Capital'].sum()}, Value ${selected['Strategic_Value'].sum()}, ESG {selected['ESG_Score'].mean():.1f}. Focus on wealth creation.")
                    st.markdown(f"<div class='ai-insight-box'>{res.text}</div>", unsafe_allow_html=True)
                except: st.warning("AI cooling down.")

    elif nav == " ML INTELLIGENCE":
        st.markdown('<div class="section-header">PREDICTIVE ROI LOGIC</div>', unsafe_allow_html=True)
        col_l, col_r = st.columns(2)
        with col_l:
            st.write("**DRIVERS OF VALUE (FEATURE IMPORTANCE)**")
            feat_imp = pd.DataFrame({'Feature': feats, 'Importance': rf_model.feature_importances_})
            st.plotly_chart(px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color_continuous_scale='Blues'), use_container_width=True)
        with col_r:
            st.write("**EFFICIENCY FRONTIER**")
            st.plotly_chart(px.scatter(df, x="ESG_Score", y="Strategic_Value", size="Investment_Capital", color="Selected", color_discrete_map={1:'#58a6ff', 0:'#30363d'}), use_container_width=True)
        
        st.markdown('<div class="section-header">ESG PILLAR RADAR</div>', unsafe_allow_html=True)
        p_means = selected[['E_Score', 'S_Score', 'G_Score']].mean().reset_index()
        p_means.columns = ['Pillar', 'Score']
        st.plotly_chart(px.line_polar(p_means, r='Score', theta='Pillar', line_close=True, range_r=[0,10], color_discrete_sequence=['#238636']), use_container_width=True)
        
        if st.button("Interpret ML Intelligence"):
            with st.spinner("Decoding ML..."):
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    res = model.generate_content(f"Analyze these ML results for an investor: Top drivers are {feat_imp.nlargest(2, 'Importance')['Feature'].tolist()}. ESG Radar averages are {p_means['Score'].tolist()}.")
                    st.markdown(f"<div class='ai-insight-box'>{res.text}</div>", unsafe_allow_html=True)
                except: st.warning("AI cooling down.")

    elif nav == " SENSITIVITY":
        st.markdown('<div class="section-header">VALUE SENSITIVITY: BUDGET VS SUSTAINABILITY</div>', unsafe_allow_html=True)
        b_range = np.linspace(budget * 0.8, budget * 1.2, 5)
        e_range = np.linspace(4, 9, 5)
        h_map = []
        for b in b_range:
            row = []
            for e in e_range:
                temp = run_optimization(df.copy(), b, e)
                row.append(temp[temp['Selected'] == 1]['Strategic_Value'].sum())
            h_map.append(row)
        
        st.plotly_chart(px.imshow(h_map, x=[f"ESG {e:.1f}" for e in e_range], y=[f"${b/1e6:.1f}M" for b in b_range], color_continuous_scale='RdYlGn'), use_container_width=True)
        
        st.markdown('<div class="section-header">3-YEAR CASH OUTLAY SCHEDULE</div>', unsafe_allow_html=True)
        selected['Y1'] = selected['Investment_Capital'] * selected['Phase_1_Cap']
        selected['Y2'] = selected['Investment_Capital'] * selected['Phase_2_Cap']
        selected['Y3'] = selected['Investment_Capital'] - (selected['Y1'] + selected['Y2'])
        ph_sum = selected[['Y1', 'Y2', 'Y3']].sum().reset_index()
        ph_sum.columns = ['Year', 'Outlay']
        st.plotly_chart(px.area(ph_sum, x='Year', y='Outlay', color_discrete_sequence=['#1f6feb']), use_container_width=True)

        if st.button("Interpret Sensitivity"):
            with st.spinner("Analyzing trade-offs..."):
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    res = model.generate_content(f"Explain the sensitivity heatmap. Budget is ${budget} and ESG hurdle is {esg_min}. How does changing these impact the ${selected['Strategic_Value'].sum()} total value?")
                    st.markdown(f"<div class='ai-insight-box'>{res.text}</div>", unsafe_allow_html=True)
                except: st.warning("AI cooling down.")

    elif nav == " RISK MANAGEMENT":
        st.markdown('<div class="section-header">STRATEGIC RISK-RETURN QUADRANT</div>', unsafe_allow_html=True)
        m_r, m_p = df['Risk_Score'].median(), df['PI'].median()
        fig_q = px.scatter(df, x="Risk_Score", y="PI", color="Selected", size="Investment_Capital", text="Project_ID", color_discrete_map={1:'#238636', 0:'#30363d'})
        fig_q.add_hrect(y0=m_p, y1=df['PI'].max()*1.2, x0=0, x1=m_r, fillcolor="green", opacity=0.08, layer="below", annotation_text="STRATEGIC CORE")
        fig_q.add_hrect(y0=0, y1=m_p, x0=m_r, x1=10, fillcolor="red", opacity=0.08, layer="below", annotation_text="VALUE TRAPS")
        st.plotly_chart(fig_q, use_container_width=True)
        
        st.markdown('<div class="section-header">DOWNSIDE RISK (VaR & SHARPE)</div>', unsafe_allow_html=True)
        mu_p = (selected['Pred_ROI'] / 100).mean()
        sig_p = (selected['Pred_ROI'] / 100).std()
        k1, k2 = st.columns(2)
        k1.metric("VAR (95% CONFIDENCE)", f"{norm.ppf(0.05, mu_p, sig_p):.2%}")
        k2.metric("SHARPE RATIO AVG", f"{selected['Sharpe_Score'].mean():.2f}")

        if st.button("Interpret Risk Exposure"):
            with st.spinner("Decoding Risk..."):
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    res = model.generate_content(f"Analyze risk for a CFO. Portfolio VaR is {norm.ppf(0.05, mu_p, sig_p):.2%} and Sharpe Ratio is {selected['Sharpe_Score'].mean():.2f}. Explain what this means for capital safety.")
                    st.markdown(f"<div class='ai-insight-box'>{res.text}</div>", unsafe_allow_html=True)
                except: st.warning("AI cooling down.")

    elif nav == " INSTITUTIONAL THESIS":
        st.markdown('<div class="section-header">PROJECT DEEP-DIVE THESIS</div>', unsafe_allow_html=True)
        target = st.selectbox("SELECT PROJECT FOR QUANT ANALYSIS", selected['Project_ID'])
        r = selected[selected['Project_ID'] == target].iloc[0]
        
        if st.button("Interpret Project Thesis"):
            with st.spinner("AI Quant at work..."):
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    res = model.generate_content(f"Deep Quant analysis for project {target}. Strategic Value ${r['Strategic_Value']:.2f}, Profitability Index {r['PI']:.2f}, Risk Score {r['Risk_Score']:.1f}. Provide a professional investment thesis.")
                    st.markdown(f"<div class='ai-insight-box'>{res.text}</div>", unsafe_allow_html=True)
                except: st.error("AI rate limit. Wait 60s.")

    st.markdown("---")
    st.download_button(" DOWNLOAD (CSV)", selected.to_csv(index=False), file_name="STRATOS_Approved_Portfolio.csv", mime="text/csv")

if __name__ == "__main__":
    main()
