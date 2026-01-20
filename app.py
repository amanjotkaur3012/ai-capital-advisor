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
    prob += pulp.lpSum([df.loc[i, 'ESG_Score'] * xs[i] for i in df.index]) >= esg_hurdle * pulp.lpSum([xs[i] for i in df.index])
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    df['Selected'] = [pulp.value(xs[i]) for i in df.index]
    return df

# ----------------------------------------------------
# 2. MAIN APPLICATION LOGIC
# ----------------------------------------------------

def main():
    st.markdown('<p class="main-title">STRATOS | Capital Allocation Advisor</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8b949e; margin-bottom: 30px;">Institutional Strategy & Multi-Criteria Decision Support System</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Data Management")
        up_file = st.file_uploader("Upload Project Dataset (CSV)", type="csv")
        use_demo = st.checkbox("Load MSc Institutional Demo", value=not up_file)
        
        st.header("Financial Benchmarks")
        rf_rate = st.slider("Risk-Free Rate (Rf)", 0.0, 8.0, 4.2) / 100
        budget = st.number_input("Capital Constraint ($)", value=5500000, step=500000)
        esg_min = st.slider("ESG Sustainability Hurdle", 1, 10, 6)

    # Dataset Generation/Loading
    if up_file:
        df = pd.read_csv(up_file)
    else:
        np.random.seed(42)
        df = pd.DataFrame({
            'Project_ID': [f"FIN-{i:03d}" for i in range(1, 26)],
            'Department': np.random.choice(['ESG Fintech', 'R&D', 'Core Infrastructure', 'Digital Assets'], 25),
            'Investment_Capital': np.random.choice([200000, 500000, 750000, 1250000, 2000000], 25),
            'Risk_Score': np.random.uniform(2.0, 9.5, 25),
            # ADDITION: ESG Pillar Breakdown Data
            'E_Score': np.random.uniform(1, 10, 25),
            'S_Score': np.random.uniform(1, 10, 25),
            'G_Score': np.random.uniform(1, 10, 25),
            'Volatility': np.random.uniform(0.12, 0.45, 25),
            'Strategic_Alignment': np.random.randint(4, 11, 25),
            # ADDITION: Phasing Percentages
            'Phase_1_Cap': np.random.uniform(0.2, 0.5, 25), 
            'Phase_2_Cap': np.random.uniform(0.2, 0.4, 25)
        })
        df['ESG_Score'] = (df['E_Score'] + df['S_Score'] + df['G_Score']) / 3
        df['Actual_ROI'] = (df['Strategic_Alignment'] * 1.8) - (df['Risk_Score'] * 0.5) + 11 + np.random.normal(0, 1.5, 25)

    # ML Forecaster
    feats = ['Investment_Capital', 'Risk_Score', 'ESG_Score', 'Volatility', 'Strategic_Alignment']
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(df[feats], df['Actual_ROI'])
    df['Pred_ROI'] = rf_model.predict(df[feats])
    feat_imp = pd.DataFrame({'Feature': feats, 'Importance': rf_model.feature_importances_})

    # Quantitative Metrics
    wacc = rf_rate + 0.06 
    df['ROA_Value'] = df.apply(lambda x: black_scholes_roa(x['Investment_Capital']*1.35, x['Investment_Capital'], 2, rf_rate, x['Volatility']), axis=1)
    df['NPV'] = (df['Investment_Capital'] * (df['Pred_ROI']/100)) / wacc
    df['Strategic_Value'] = df['NPV'] + df['ROA_Value']
    df['PI'] = df['Strategic_Value'] / df['Investment_Capital']
    
    # ADDITION: Sharpe Score Calculation
    df['Sharpe_Score'] = (df['Pred_ROI'] - (rf_rate * 100)) / (df['Volatility'] * 100)

    # Solver Implementation
    df = run_optimization(df, budget, esg_min)
    selected = df[df['Selected'] == 1]

    # --- DASHBOARD TABS ---
    t1, t2, t3, t4, t5 = st.tabs(["🚀 Summary", "🧠 ML Analytics", "📊 Sensitivity", "🛡️ Risk Quadrant", "🤖 AI Thesis"])
    
    with t1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capital Allocated", f"${selected['Investment_Capital'].sum():,.0f}")
        c2.metric("Portfolio Strategic NPV", f"${selected['Strategic_Value'].sum():,.0f}")
        c3.metric("Avg ESG Impact", f"{selected['ESG_Score'].mean():.1f}")
        c4.metric("Avg PI", f"{selected['PI'].mean():.2f}x")
        
        st.write("### 📋 Final Investment Schedule")
        st.dataframe(selected[['Project_ID', 'Department', 'Investment_Capital', 'Pred_ROI', 'PI', 'ESG_Score']].style.background_gradient(cmap='Greens'))
        
        # Portfolio Risk Snapshot
        st.markdown("---")
        st.markdown("### 📉 Portfolio Risk Snapshot")
        portfolio_returns = selected['Pred_ROI'] / 100
        mu_p = portfolio_returns.mean()
        sigma_p = portfolio_returns.std()
        VaR_95 = norm.ppf(0.05, mu_p, sigma_p)
        CVaR_95 = mu_p - (sigma_p * norm.pdf(norm.ppf(0.05)) / 0.05)

        r1, r2, r3 = st.columns(3)
        r1.metric("Expected Return", f"{mu_p:.2%}")
        r2.metric("VaR (95%)", f"{VaR_95:.2%}")
        r3.metric("CVaR (95%)", f"{CVaR_95:.2%}")
        
        if st.button("🤖 Explain My Portfolio Health"):
            with st.spinner("Analyzing portfolio balance..."):
                try:
                    m_list = genai.list_models()
                    available = [m.name for m in m_list if 'generateContent' in m.supported_generation_methods]
                    m_name = "gemini-1.5-flash" if f"models/gemini-1.5-flash" in available else available[0].replace("models/", "")
                    model = genai.GenerativeModel(m_name)
                    summary_prompt = f"Explain this portfolio summary in 3 bullet points: Budget used ${selected['Investment_Capital'].sum()}, NPV ${selected['Strategic_Value'].sum()}, Average PI {selected['PI'].mean():.2f}."
                    response = model.generate_content(summary_prompt)
                    st.info(response.text)
                except Exception as e: st.error("AI Service busy.")

    with t2:
        col_l, col_r = st.columns(2)
        with col_l:
            st.write("**Feature Importance (ML Forecasting)**")
            fig_i = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Teal')
            st.plotly_chart(fig_i, use_container_width=True)
        with col_r:
            st.write("**Efficiency Frontier**")
            fig_f = px.scatter(df, x="ESG_Score", y="Strategic_Value", size="Investment_Capital", color="Selected",
                               color_discrete_map={1:'#10b981', 0:'#ff4b4b'}, labels={"Strategic_Value": "Strategic Value ($)"})
            st.plotly_chart(fig_f, use_container_width=True)
            
        # ADDITION: ESG Pillar Radar Chart
        st.markdown("---")
        st.write("### 🌿 ESG Pillar Breakdown (Selected Portfolio)")
        
        pillar_means = selected[['E_Score', 'S_Score', 'G_Score']].mean().reset_index()
        pillar_means.columns = ['Pillar', 'Average Score']
        fig_radar = px.line_polar(pillar_means, r='Average Score', theta='Pillar', line_close=True, range_r=[0,10])
        fig_radar.update_traces(fill='toself', line_color='#10b981')
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("### 🏆 Capital Efficiency Ranking")
        eff = selected.sort_values("PI", ascending=False)
        st.dataframe(eff[['Project_ID', 'Department', 'PI', 'Pred_ROI', 'ESG_Score']], use_container_width=True)
            
        if st.button("🤖 Explain the AI Logic"):
            with st.spinner("Translating ML insights..."):
                model = genai.GenerativeModel("gemini-1.5-flash")
                ml_prompt = "Explain why 'Feature Importance' and an 'Efficiency Frontier' matter for project selection."
                response = model.generate_content(ml_prompt)
                st.info(response.text)

    with t3:
        st.subheader("Sensitivity Matrix: The Budget-ESG Trade-off")
        b_range = np.linspace(budget * 0.8, budget * 1.2, 5)
        e_range = np.linspace(4, 9, 5)
        h_map = []
        for b in b_range:
            row = []
            for e in e_range:
                temp = run_optimization(df.copy(), b, e)
                row.append(temp[temp['Selected'] == 1]['Strategic_Value'].sum())
            h_map.append(row)
        
        
        fig_h = px.imshow(h_map, x=[f"ESG {e:.1f}" for e in e_range], y=[f"${b/1e6:.1f}M" for b in b_range], color_continuous_scale='RdYlGn', labels=dict(x="Sustainability Hurdle", y="Budget Capacity", color="Total Value"))
        st.plotly_chart(fig_h, use_container_width=True)
        
        # ADDITION: Phasing Area Chart
        st.markdown("---")
        st.write("### 📅 Capital Phasing Schedule (Next 3 Years)")
        selected['Y1'] = selected['Investment_Capital'] * selected['Phase_1_Cap']
        selected['Y2'] = selected['Investment_Capital'] * selected['Phase_2_Cap']
        selected['Y3'] = selected['Investment_Capital'] - (selected['Y1'] + selected['Y2'])
        phasing_summary = selected[['Y1', 'Y2', 'Y3']].sum().reset_index()
        phasing_summary.columns = ['Year', 'Total Outlay']
        fig_phase = px.area(phasing_summary, x='Year', y='Total Outlay', color_discrete_sequence=['#3b82f6'])
        st.plotly_chart(fig_phase, use_container_width=True)

        st.markdown("### 🌍 Macro Scenario Stress Test")
        scenarios = {"Recession": 0.7, "Base Case": 1.0, "Economic Boom": 1.3, "Rate Shock": 0.85}
        scenario_results = {k: (selected['Strategic_Value'] * v).sum() for k, v in scenarios.items()}
        fig_s = px.bar(x=scenario_results.keys(), y=scenario_results.values(), title="Scenario Sensitivity")
        st.plotly_chart(fig_s, use_container_width=True)

        if st.button("🤖 Explain this Heatmap"):
            with st.spinner("Interpreting matrix..."):
                model = genai.GenerativeModel("gemini-1.5-flash")
                h_prompt = "Explain this 'Budget vs ESG' heatmap and how it helps decision making."
                response = model.generate_content(h_prompt)
                st.info(response.text)

    with t4:
        st.subheader("Strategic Risk-Return Quadrant")
        m_risk = df['Risk_Score'].median()
        m_pi = df['PI'].median()
        
        
        fig_q = px.scatter(df, x="Risk_Score", y="PI", color="Selected", size="Investment_Capital", hover_name="Project_ID", text="Project_ID", color_discrete_map={1: '#10b981', 0: '#455a64'})
        fig_q.update_traces(textposition='top center')
        fig_q.add_hrect(y0=m_pi, y1=df['PI'].max()*1.2, x0=0, x1=m_risk, fillcolor="green", opacity=0.08, layer="below", annotation_text="STRATEGIC CORE")
        fig_q.add_hrect(y0=0, y1=m_pi, x0=m_risk, x1=10, fillcolor="red", opacity=0.08, layer="below", annotation_text="VALUE TRAPS")
        st.plotly_chart(fig_q, use_container_width=True)
        
        # ADDITION: Sharpe Ratio Bar Chart
        st.markdown("---")
        st.write("### ⚖️ Risk-Adjusted Efficiency (Sharpe Score)")
        fig_sharpe = px.bar(selected.sort_values('Sharpe_Score', ascending=False), x='Project_ID', y='Sharpe_Score', color='Sharpe_Score', color_continuous_scale='Viridis')
        st.plotly_chart(fig_sharpe, use_container_width=True)

        st.markdown("### 🍩 Portfolio Capital Allocation Mix")
        fig_mix = px.pie(selected, names='Department', values='Investment_Capital', hole=0.45)
        st.plotly_chart(fig_mix, use_container_width=True)

        if st.button("🤖 Explain this Risk Chart"):
            with st.spinner("Decoding Risk-Return paths..."):
                model = genai.GenerativeModel("gemini-1.5-flash")
                q_prompt = "Explain this Risk-Return Quadrant for a beginner."
                response = model.generate_content(q_prompt)
                st.info(response.text)

with t5:
        st.subheader("Deep-Dive AI Investment Thesis")
        target = st.selectbox("Select Project for AI Deep Dive", selected['Project_ID'])
        r = selected[selected['Project_ID'] == target].iloc[0]
        
        if st.button("🖋️ Generate Expert Memo"):
            with st.spinner("Accessing Institutional Intelligence..."):
                try:
                    m_list = genai.list_models()
                    available = [m.name for m in m_list if 'generateContent' in m.supported_generation_methods]
                    m_name = "gemini-1.5-flash" if f"models/gemini-1.5-flash" in available else available[0].replace("models/", "")
                    
                    model = genai.GenerativeModel(m_name)
                    prompt = f"Senior Quant perspective: Analyze {target}. PI {r['PI']:.2f}, ESG {r['ESG_Score']}, Sharpe {r['Sharpe_Score']:.2f}. Discuss trade-offs."
                    response = model.generate_content(prompt)
                    st.success(f"Analyst: {m_name.upper()}")
                    st.markdown(response.text)
                except Exception as e:
                    if "429" in str(e):
                        st.warning("⏳ **AI Analyst is cooling down.** We have reached the free-tier limit (5 requests/min). Please wait 60 seconds before generating the next memo.")
                    else:
                        st.error(f"Connection Error: {e}")

    st.markdown("---")
    st.markdown("### 📥 Export for Investment Committee")
    st.download_button("Download Approved Portfolio (CSV)", selected.to_csv(index=False), file_name="Approved_Portfolio.csv", mime="text/csv")

if __name__ == "__main__":
    main()
