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
import google.generativeai as genai
from scipy.stats import norm
import io

# ----------------------------------------------------
# 0. BACKEND CONFIGURATION & API
# ----------------------------------------------------
GEMINI_API_KEY = "AIzaSyBEFV8Q9A9DvRUMzB9tD3KlUPvsv_60j60"
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="STRATOS QUANT | MSc Capital Advisor", layout="wide")

# PROFESSIONAL INSTITUTIONAL CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; }
    .stApp { background-color: #0d1117; color: #f0f6fc; }
    
    .main-title { 
        font-size: 42px !important; font-weight: 800; letter-spacing: -2px; 
        background: linear-gradient(90deg, #58a6ff, #1f6feb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .section-header { 
        font-size: 24px; font-weight: 700; color: #ffffff; 
        border-left: 5px solid #238636; padding-left: 15px; margin: 25px 0 15px 0;
    }
    
    div[data-testid="stMetric"] {
        background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    div[data-testid="stMetricValue"] { font-size: 32px !important; color: #ffffff !important; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 16px !important; color: #8b949e !important; text-transform: uppercase; letter-spacing: 1px; }

    section[data-testid="stSidebar"] { background-color: #010409 !important; border-right: 1px solid #30363d; }
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
        st.markdown("<h1 style='color:#58a6ff;'>NAVIGATOR</h1>", unsafe_allow_html=True)
        # NAVIGATION DROPDOWN
        nav = st.radio("SELECT VIEW", ["🚀 EXECUTIVE SUMMARY", "🧠 ML ANALYTICS", "📊 SENSITIVITY", "🛡️ RISK QUADRANT", "🤖 AI THESIS"])
        
        st.markdown("---")
        st.header("DATA MANAGEMENT")
        up_file = st.file_uploader("Upload Proposals (CSV)", type="csv")
        use_demo = st.checkbox("Load Institutional Demo", value=not up_file)
        
        st.header("FINANCIAL SETTINGS")
        rf_rate = st.slider("Risk-Free Rate (Rf)", 0.0, 8.0, 4.2) / 100
        budget = st.number_input("Capital Pool ($)", value=5500000, step=500000)
        esg_min = st.slider("Portfolio ESG Hurdle", 1, 10, 6)

    # Data Setup
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

    # Calculation Engine
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

    st.markdown('<p class="main-title">STRATOS | Capital Allocation Advisor</p>', unsafe_allow_html=True)

    # ----------------------------------------------------
    # VIEW ROUTING
    # ----------------------------------------------------

    if nav == "🚀 EXECUTIVE SUMMARY":
        st.markdown('<div class="section-header">PORTFOLIO OVERVIEW</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAPITAL DEPLOYED", f"${selected['Investment_Capital'].sum():,.0f}")
        c2.metric("STRATEGIC NPV", f"${selected['Strategic_Value'].sum():,.0f}")
        c3.metric("AVG ESG IMPACT", f"{selected['ESG_Score'].mean():.1f}/10")
        c4.metric("VALUE INDEX (PI)", f"{selected['PI'].mean():.2f}x")
        
        st.markdown('<div class="section-header">APPROVED INVESTMENT SCHEDULE</div>', unsafe_allow_html=True)
        st.dataframe(selected[['Project_ID', 'Department', 'Investment_Capital', 'Pred_ROI', 'PI', 'ESG_Score']].style.background_gradient(cmap='Greens'), use_container_width=True)
        
        st.markdown('<div class="section-header">PORTFOLIO RISK PROFILE</div>', unsafe_allow_html=True)
        mu_p = (selected['Pred_ROI'] / 100).mean()
        sig_p = (selected['Pred_ROI'] / 100).std()
        r1, r2, r3 = st.columns(3)
        r1.metric("EXPECTED RETURN", f"{mu_p:.2%}")
        r2.metric("VAR (95%)", f"{norm.ppf(0.05, mu_p, sig_p):.2%}")
        r3.metric("SHARPE AVG", f"{selected['Sharpe_Score'].mean():.2f}")

    elif nav == "🧠 ML ANALYTICS":
        st.markdown('<div class="section-header">PREDICTIVE INTELLIGENCE</div>', unsafe_allow_html=True)
        col_l, col_r = st.columns(2)
        with col_l:
            st.write("**FEATURE IMPORTANCE**")
            feat_imp = pd.DataFrame({'Feature': feats, 'Importance': rf_model.feature_importances_})
            st.plotly_chart(px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color_continuous_scale='Blues'), use_container_width=True)
        with col_r:
            st.write("**STRATEGIC EFFICIENCY FRONTIER**")
            st.plotly_chart(px.scatter(df, x="ESG_Score", y="Strategic_Value", size="Investment_Capital", color="Selected", color_discrete_map={1:'#58a6ff', 0:'#30363d'}), use_container_width=True)
        
        st.markdown('<div class="section-header">ESG PILLAR BREAKDOWN</div>', unsafe_allow_html=True)
        p_means = selected[['E_Score', 'S_Score', 'G_Score']].mean().reset_index()
        p_means.columns = ['Pillar', 'Score']
        st.plotly_chart(px.line_polar(p_means, r='Score', theta='Pillar', line_close=True, range_r=[0,10], color_discrete_sequence=['#238636']), use_container_width=True)

    elif nav == "📊 SENSITIVITY":
        st.markdown('<div class="section-header">SENSITIVITY MATRIX</div>', unsafe_allow_html=True)
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
        
        st.markdown('<div class="section-header">3-YEAR CAPITAL PHASING</div>', unsafe_allow_html=True)
        selected['Y1'] = selected['Investment_Capital'] * selected['Phase_1_Cap']
        selected['Y2'] = selected['Investment_Capital'] * selected['Phase_2_Cap']
        selected['Y3'] = selected['Investment_Capital'] - (selected['Y1'] + selected['Y2'])
        ph_sum = selected[['Y1', 'Y2', 'Y3']].sum().reset_index()
        ph_sum.columns = ['Year', 'Outlay']
        st.plotly_chart(px.area(ph_sum, x='Year', y='Outlay', color_discrete_sequence=['#1f6feb']), use_container_width=True)

    elif nav == "🛡️ RISK QUADRANT":
        st.markdown('<div class="section-header">RISK-RETURN QUADRANT</div>', unsafe_allow_html=True)
        m_r, m_p = df['Risk_Score'].median(), df['PI'].median()
        fig_q = px.scatter(df, x="Risk_Score", y="PI", color="Selected", size="Investment_Capital", text="Project_ID", color_discrete_map={1:'#238636', 0:'#30363d'})
        fig_q.add_hrect(y0=m_p, y1=df['PI'].max()*1.2, x0=0, x1=m_r, fillcolor="green", opacity=0.08, annotation_text="STRATEGIC CORE")
        fig_q.add_hrect(y0=0, y1=m_p, x0=m_r, x1=10, fillcolor="red", opacity=0.08, annotation_text="VALUE TRAPS")
        st.plotly_chart(fig_q, use_container_width=True)
        
        st.markdown('<div class="section-header">CAPITAL ALLOCATION MIX</div>', unsafe_allow_html=True)
        st.plotly_chart(px.pie(selected, names='Department', values='Investment_Capital', hole=0.5, color_discrete_sequence=px.colors.sequential.Blues_r), use_container_width=True)

    elif nav == "🤖 AI THESIS":
        st.markdown('<div class="section-header">INSTITUTIONAL AI THESIS</div>', unsafe_allow_html=True)
        target = st.selectbox("SELECT PROJECT", selected['Project_ID'])
        r = selected[selected['Project_ID'] == target].iloc[0]
        
        if st.button("EXECUTE ANALYSIS"):
            with st.spinner("AI analyzing fundamentals..."):
                try:
                    m_list = genai.list_models()
                    m_name = [m.name for m in m_list if 'generateContent' in m.supported_generation_methods][0]
                    model = genai.GenerativeModel(m_name)
                    res = model.generate_content(f"Analyze project {target}. Strategic Value ${r['Strategic_Value']:.2f}, PI {r['PI']:.2f}, Sharpe {r['Sharpe_Score']:.2f}.")
                    st.success(f"Analyst: {m_name.upper()}")
                    st.markdown(res.text)
                except Exception as e:
                    if "429" in str(e):
                        st.warning("⏳ AI Analyst is cooling down. Please wait 60 seconds.")
                    else:
                        st.error(f"Error: {e}")

    # Footer
    st.markdown("---")
    st.download_button("📥 DOWNLOAD PORTFOLIO (CSV)", selected.to_csv(index=False), file_name="STRATOS_Approved_Portfolio.csv", mime="text/csv")

if __name__ == "__main__":
    main()
