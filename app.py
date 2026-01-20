# =========================================================
# FINCAP-AI | Intelligent Capital Allocation Advisor
# MSc Finance & Analytics | Executive Decision Support
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

# ULTIMATE EXECUTIVE TERMINAL CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; }
    .stApp { background-color: #0d1117; color: #f0f6fc; }
    
    /* Title & Headers */
    .main-title { 
        font-size: 48px !important; font-weight: 800; letter-spacing: -2.5px; 
        background: linear-gradient(90deg, #58a6ff, #10b981);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .page-desc { color: #8b949e; font-size: 18px; margin-bottom: 30px; font-weight: 400; }
    .section-header { 
        font-size: 26px; font-weight: 700; color: #ffffff; 
        border-left: 6px solid #238636; padding-left: 15px; margin: 35px 0 15px 0;
    }
    
    /* Metrics High-Visibility */
    div[data-testid="stMetric"] {
        background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    div[data-testid="stMetricValue"] { font-size: 38px !important; color: #ffffff !important; }
    
    /* Sidebar Focus */
    section[data-testid="stSidebar"] { background-color: #010409 !important; border-right: 2px solid #30363d; }
    
    /* AI Box Styling */
    .ai-box {
        background: rgba(88, 166, 255, 0.1); border: 1px solid #58a6ff;
        padding: 20px; border-radius: 8px; color: #e2e8f0; margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 1. QUANTITATIVE FINANCE ENGINES
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
# 2. MAIN APPLICATION LOGIC
# ----------------------------------------------------

def main():
    with st.sidebar:
        st.markdown("<h1 style='color:#58a6ff; font-size:32px;'>STRATOS NAV</h1>", unsafe_allow_html=True)
        nav = st.radio("DASHBOARD NAVIGATION", ["🚀 EXECUTIVE SUMMARY", "🧠 ML ANALYTICS", "📊 SENSITIVITY", "🛡️ RISK QUADRANT", "🤖 AI EXPERT THESIS"])
        
        st.markdown("---")
        st.header("DATA CONTROL")
        up_file = st.file_uploader("Upload Project Dataset", type="csv")
        use_demo = st.checkbox("Load Institutional Data", value=not up_file)
        
        st.header("MACRO PARAMETERS")
        rf_rate = st.slider("Risk-Free Rate (Rf)", 0.0, 8.0, 4.2) / 100
        budget = st.number_input("Capital Constraint ($)", value=5500000, step=500000)
        esg_min = st.slider("ESG Sustainability Hurdle", 1, 10, 6)

    # Dataset Setup
    if up_file:
        df = pd.read_csv(up_file)
    else:
        np.random.seed(42)
        df = pd.DataFrame({
            'Project_ID': [f"FIN-{i:03d}" for i in range(1, 26)],
            'Department': np.random.choice(['ESG Fintech', 'R&D', 'Infrastructure', 'Digital Assets'], 25),
            'Investment_Capital': np.random.choice([200000, 500000, 750000, 1250000, 2000000], 25),
            'Risk_Score': np.random.uniform(2.0, 9.5, 25),
            'E_Score': np.random.uniform(1, 10, 25),
            'S_Score': np.random.uniform(1, 10, 25),
            'G_Score': np.random.uniform(1, 10, 25),
            'Volatility': np.random.uniform(0.12, 0.45, 25),
            'Strategic_Alignment': np.random.randint(4, 11, 25),
            'Phase_1_Cap': np.random.uniform(0.2, 0.5, 25), 
            'Phase_2_Cap': np.random.uniform(0.2, 0.4, 25)
        })
        df['ESG_Score'] = (df['E_Score'] + df['S_Score'] + df['G_Score']) / 3
        df['Actual_ROI'] = (df['Strategic_Alignment'] * 1.8) - (df['Risk_Score'] * 0.5) + 11 + np.random.normal(0, 1.5, 25)

    # AI/ML Calculations
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

    # TOP HEADER
    st.markdown('<p class="main-title">STRATOS | Capital Allocation Advisor</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Institutional Multi-Criteria Decision Support System for MSC Finance & Analytics</p>', unsafe_allow_html=True)

    # ----------------------------------------------------
    # ROUTING PAGES
    # ----------------------------------------------------

    if nav == "🚀 EXECUTIVE SUMMARY":
        st.markdown('<div class="section-header">PORTFOLIO PERFORMANCE METRICS</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAPITAL UTILIZED", f"${selected['Investment_Capital'].sum():,.0f}", help="Total budget currently deployed.")
        c2.metric("STRATEGIC NPV", f"${selected['Strategic_Value'].sum():,.0f}", help="Sum of traditional NPV plus the value of future growth options.")
        c3.metric("PORTFOLIO ESG", f"{selected['ESG_Score'].mean():.1f}/10", help="Average sustainability rating of funded projects.")
        c4.metric("VALUE INDEX (PI)", f"{selected['PI'].mean():.2f}x", help="Value created per dollar invested. Institutional hurdle is 1.0x.")
        
        st.markdown('<div class="section-header">APPROVED INVESTMENT SCHEDULE</div>', unsafe_allow_html=True)
        st.dataframe(selected[['Project_ID', 'Department', 'Investment_Capital', 'Pred_ROI', 'PI', 'ESG_Score']].style.background_gradient(cmap='Greens'), use_container_width=True)
        
        if st.button("🤖 AI Analysis: Evaluate Portfolio Health"):
            with st.spinner("AI analyzing portfolio balance..."):
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    res = model.generate_content(f"Explain this portfolio summary to a CEO: Budget used ${selected['Investment_Capital'].sum()}, Strategic Value ${selected['Strategic_Value'].sum()}, ESG {selected['ESG_Score'].mean():.1f}. Is this a balanced use of capital?")
                    st.markdown(f"<div class='ai-box'>{res.text}</div>", unsafe_allow_html=True)
                except: st.warning("AI cooling down. Wait 60s.")

    elif nav == "🧠 ML ANALYTICS":
        st.markdown('<div class="section-header">PREDICTIVE ROI INTELLIGENCE</div>', unsafe_allow_html=True)
        col_l, col_r = st.columns(2)
        with col_l:
            st.write("**FEATURE IMPORTANCE (WHY THE MODEL PICKS PROJECTS)**")
            feat_imp = pd.DataFrame({'Feature': feats, 'Importance': rf_model.feature_importances_})
            st.plotly_chart(px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color_continuous_scale='Blues'), use_container_width=True)
        with col_r:
            st.write("**EFFICIENCY FRONTIER (STRATEGIC VALUE VS ESG)**")
            st.plotly_chart(px.scatter(df, x="ESG_Score", y="Strategic_Value", size="Investment_Capital", color="Selected", color_discrete_map={1:'#58a6ff', 0:'#30363d'}), use_container_width=True)
        
        st.markdown('<div class="section-header">ESG PILLAR BREAKDOWN</div>', unsafe_allow_html=True)
        p_means = selected[['E_Score', 'S_Score', 'G_Score']].mean().reset_index()
        p_means.columns = ['Pillar', 'Score']
        
        st.plotly_chart(px.line_polar(p_means, r='Score', theta='Pillar', line_close=True, range_r=[0,10], color_discrete_sequence=['#238636']), use_container_width=True)
        
        if st.button("🤖 AI Analysis: Explain Machine Learning Logic"):
            with st.spinner("Translating ML insights..."):
                model = genai.GenerativeModel("gemini-1.5-flash")
                res = model.generate_content("Explain to a manager why 'Feature Importance' and 'ESG Pillars' are better than just looking at raw ROI.")
                st.markdown(f"<div class='ai-box'>{res.text}</div>", unsafe_allow_html=True)

    elif nav == "📊 SENSITIVITY":
        st.markdown('<div class="section-header">VALUE SENSITIVITY MATRIX: BUDGET VS ESG</div>', unsafe_allow_html=True)
        b_range = np.linspace(budget * 0.8, budget * 1.2, 5)
        e_range = np.linspace(4, 9, 5)
        h_map = []
        for b in b_range:
            row = []
            for e in e_range:
                temp = run_optimization(df.copy(), b, e)
                row.append(temp[temp['Selected'] == 1]['Strategic_Value'].sum())
            h_map.append(row)
        
        st.plotly_chart(px.imshow(h_map, x=[f"ESG {e:.1f}" for e in e_range], y=[f"${b/1e6:.1f}M" for b in b_range], color_continuous_scale='RdYlGn', labels=dict(x="ESG Hurdle", y="Budget", color="NPV")), use_container_width=True)
        
        st.markdown('<div class="section-header">CAPITAL PHASING SCHEDULE (NEXT 3 YEARS)</div>', unsafe_allow_html=True)
        selected['Y1'] = selected['Investment_Capital'] * selected['Phase_1_Cap']
        selected['Y2'] = selected['Investment_Capital'] * selected['Phase_2_Cap']
        selected['Y3'] = selected['Investment_Capital'] - (selected['Y1'] + selected['Y2'])
        ph_sum = selected[['Y1', 'Y2', 'Y3']].sum().reset_index()
        ph_sum.columns = ['Year', 'Outlay']
        st.plotly_chart(px.area(ph_sum, x='Year', y='Outlay', color_discrete_sequence=['#1f6feb']), use_container_width=True)
        
        if st.button("🤖 AI Analysis: Explain Multi-Period Phasing"):
            with st.spinner("Analyzing trade-offs..."):
                model = genai.GenerativeModel("gemini-1.5-flash")
                res = model.generate_content("Explain the 'Capital Phasing' chart. Why is it important for a company to know their year-by-year cash requirements instead of just the total cost?")
                st.markdown(f"<div class='ai-box'>{res.text}</div>", unsafe_allow_html=True)

    elif nav == "🛡️ RISK QUADRANT":
        st.markdown('<div class="section-header">STRATEGIC RISK-RETURN QUADRANT</div>', unsafe_allow_html=True)
        m_r, m_p = df['Risk_Score'].median(), df['PI'].median()
        
        fig_q = px.scatter(df, x="Risk_Score", y="PI", color="Selected", size="Investment_Capital", text="Project_ID", color_discrete_map={1:'#238636', 0:'#30363d'})
        fig_q.add_hrect(y0=m_p, y1=df['PI'].max()*1.2, x0=0, x1=m_r, fillcolor="green", opacity=0.08, layer="below", annotation_text="STRATEGIC CORE")
        fig_q.add_hrect(y0=0, y1=m_p, x0=m_r, x1=10, fillcolor="red", opacity=0.08, layer="below", annotation_text="VALUE TRAPS")
        st.plotly_chart(fig_q, use_container_width=True)
        
        st.markdown('<div class="section-header">RISK-ADJUSTED EFFICIENCY (SHARPE SCORES)</div>', unsafe_allow_html=True)
        st.plotly_chart(px.bar(selected.sort_values('Sharpe_Score', ascending=False), x='Project_ID', y='Sharpe_Score', color='Sharpe_Score', color_continuous_scale='Viridis'), use_container_width=True)
        
        if st.button("🤖 AI Analysis: Interpret Risk Quadrants"):
            with st.spinner("Decoding risk paths..."):
                model = genai.GenerativeModel("gemini-1.5-flash")
                res = model.generate_content("Explain the Risk-Return Quadrant. Why do we fund the Green 'Strategic Core' and avoid the Red 'Value Traps'?")
                st.markdown(f"<div class='ai-box'>{res.text}</div>", unsafe_allow_html=True)

    elif nav == "🤖 AI THESIS":
        st.markdown('<div class="section-header">INSTITUTIONAL INVESTMENT THESIS</div>', unsafe_allow_html=True)
        target = st.selectbox("SELECT PROJECT FOR DEEP-DIVE", selected['Project_ID'])
        r = selected[selected['Project_ID'] == target].iloc[0]
        
        if st.button("EXECUTE AI ANALYSIS"):
            with st.spinner("Accessing LLM Logic..."):
                try:
                    m_list = genai.list_models()
                    m_name = [m.name for m in m_list if 'generateContent' in m.supported_generation_methods][0]
                    model = genai.GenerativeModel(m_name)
                    res = model.generate_content(f"Analyze project {target}. Strategic Value ${r['Strategic_Value']:.2f}, PI {r['PI']:.2f}, Sharpe {r['Sharpe_Score']:.2f}. Discuss trade-offs.")
                    st.markdown(f"<div class='ai-box'>{res.text}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.warning("⏳ AI rate limit reached. Wait 60s.")

    # Footer
    st.markdown("---")
    st.download_button("📥 DOWNLOAD OPTIMIZED PORTFOLIO (CSV)", selected.to_csv(index=False), file_name="Approved_Portfolio.csv", mime="text/csv")

if __name__ == "__main__":
    main()
