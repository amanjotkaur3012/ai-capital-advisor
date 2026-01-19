# =========================================================
# FINCAP-AI | Intelligent Capital Allocation Advisor
# MSc Finance & Analytics Live Project
# Author: Aman

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import pulp
import google.generativeai as genai
from scipy.stats import norm
from fpdf import FPDF
import io

# ----------------------------------------------------
# 0. CORE CONFIGURATION & API INTEGRATION
# ----------------------------------------------------
# Your provided API Key is hardcoded here to ensure background operation.
GEMINI_API_KEY = "AIzaSyDAz2r4IRQT5dv3zNq-uuRO7D2O86nNueE"
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="STRATOS QUANT | MSc Finance Advisor", layout="wide")

# Custom CSS for "Bloomberg-Style" Terminal UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; }
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .main-title { 
        font-size: 36px; font-weight: 800; letter-spacing: -1.5px; 
        background: linear-gradient(90deg, #10b981, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    div[data-testid="stMetric"] {
        background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 22px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22; border: 1px solid #30363d; border-radius: 5px 5px 0 0; padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 1. QUANTITATIVE FINANCE ENGINES
# ----------------------------------------------------

def black_scholes_roa(S, K, T, r, sigma):
    """Real Options Valuation: Option to Expand."""
    if T <= 0 or sigma <= 0 or S <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def run_optimization(df, budget, esg_hurdle):
    """Mixed Integer Programming for Capital Allocation."""
    prob = pulp.LpProblem("Portfolio_Opt", pulp.LpMaximize)
    xs = pulp.LpVariable.dicts("Select", df.index, cat=pulp.LpBinary)
    
    # Objective: Maximize Total Strategic Value (NPV + ROA)
    prob += pulp.lpSum([df.loc[i, 'Strategic_Value'] * xs[i] for i in df.index])
    
    # Constraints
    prob += pulp.lpSum([df.loc[i, 'Investment_Capital'] * xs[i] for i in df.index]) <= budget
    prob += pulp.lpSum([df.loc[i, 'ESG_Score'] * xs[i] for i in df.index]) >= esg_hurdle * pulp.lpSum([xs[i] for i in df.index])
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    df['Selected'] = [pulp.value(xs[i]) for i in df.index]
    return df

# ----------------------------------------------------
# 2. MAIN APP ARCHITECTURE
# ----------------------------------------------------

def main():
    st.markdown('<p class="main-title">STRATOS | Capital Allocation Advisor</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8b949e; margin-bottom: 25px;">Institutional Strategy & Asset-Liability Management Framework</p>', unsafe_allow_html=True)

    # --- SIDEBAR: DATA & PARAMETERS ---
    with st.sidebar:
        st.header("Data Ingestion")
        up_file = st.file_uploader("Upload Project CSV", type="csv")
        use_demo = st.checkbox("Load MSc Institutional Data", value=not up_file)
        
        st.header("Risk & Macro")
        rf_rate = st.slider("Risk-Free Rate (Rf)", 0.0, 8.0, 4.2) / 100
        budget = st.number_input("Capital Pool ($)", value=5000000, step=500000)
        esg_min = st.slider("Portfolio ESG Hurdle", 1, 10, 6)

    # Data Initialization
    if up_file:
        df = pd.read_csv(up_file)
    else:
        # Create High-Fidelity Data
        np.random.seed(42)
        df = pd.DataFrame({
            'Project_ID': [f"FIN-{i:03d}" for i in range(1, 26)],
            'Department': np.random.choice(['ESG Fintech', 'R&D', 'Infrastructure', 'Digital Assets'], 25),
            'Investment_Capital': np.random.choice([250000, 500000, 1000000, 1500000], 25),
            'Risk_Score': np.random.uniform(2.0, 9.0, 25),
            'ESG_Score': np.random.uniform(3.0, 10.0, 25),
            'Volatility': np.random.uniform(0.12, 0.40, 25),
            'Strategic_Alignment': np.random.randint(4, 11, 25)
        })
        df['Actual_ROI'] = (df['Strategic_Alignment'] * 1.8) - (df['Risk_Score'] * 0.5) + 10 + np.random.normal(0, 1, 25)

    # ML ROI Forecaster
    features = ['Investment_Capital', 'Risk_Score', 'ESG_Score', 'Volatility', 'Strategic_Alignment']
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(df[features], df['Actual_ROI'])
    df['Pred_ROI'] = rf_model.predict(df[features])
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': rf_model.feature_importances_})

    # Financial Metric Layers
    wacc = rf_rate + 0.055 
    df['ROA_Value'] = df.apply(lambda x: black_scholes_roa(x['Investment_Capital']*1.3, x['Investment_Capital'], 2, rf_rate, x['Volatility']), axis=1)
    df['NPV'] = (df['Investment_Capital'] * (df['Pred_ROI']/100)) / wacc
    df['Strategic_Value'] = df['NPV'] + df['ROA_Value']
    df['PI'] = df['Strategic_Value'] / df['Investment_Capital']

    # Optimization
    df = run_optimization(df, budget, esg_min)
    selected = df[df['Selected'] == 1]

    # --- DASHBOARD TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Portfolio Summary", "🧠 ML Analytics", "📊 Sensitivity Heatmap", "🤖 AI Thesis"])
    
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capital Utilized", f"${selected['Investment_Capital'].sum():,.0f}", f"{selected['Investment_Capital'].sum()/budget*100:.1f}%")
        c2.metric("Portfolio NPV", f"${selected['Strategic_Value'].sum():,.0f}")
        c3.metric("Avg ESG Impact", f"{selected['ESG_Score'].mean():.1f}/10")
        c4.metric("Avg PI", f"{selected['PI'].mean():.2f}x")
        
        st.write("### Strategic Selection Matrix")
        st.dataframe(selected[['Project_ID', 'Department', 'Investment_Capital', 'Pred_ROI', 'PI', 'ESG_Score']].style.background_gradient(cmap='Greens'))

    with tab2:
        col_l, col_r = st.columns(2)
        with col_l:
            st.write("**Feature Importance (Random Forest)**")
            fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Teal')
            st.plotly_chart(fig_imp, use_container_width=True)
        with col_r:
            st.write("**Efficient Frontier: Strategic NPV vs ESG**")
            fig_scat = px.scatter(df, x="ESG_Score", y="Strategic_Value", size="Investment_Capital", color="Selected",
                                  hover_name="Project_ID", color_discrete_map={1:'#10b981', 0:'#ff4b4b'})
            st.plotly_chart(fig_scat, use_container_width=True)

    with tab3:
        st.subheader("Multi-Scenario Sensitivity Matrix")
        st.write("Analysis of Portfolio NPV across Budget Constraints and ESG Hurdles.")
        
        # Generating Heatmap Data
        b_range = np.linspace(budget * 0.7, budget * 1.3, 5)
        e_range = np.linspace(4, 9, 5)
        heatmap_data = []

        for b in b_range:
            row = []
            for e in e_range:
                temp_df = run_optimization(df.copy(), b, e)
                row.append(temp_df[temp_df['Selected'] == 1]['Strategic_Value'].sum())
            heatmap_data.append(row)

        fig_heat = px.imshow(heatmap_data, 
                             x=[f"ESG {e:.1f}" for e in e_range], 
                             y=[f"${b/1e6:.1f}M" for b in b_range],
                             labels=dict(x="ESG Hurdle", y="Budget Capacity", color="Total NPV"),
                             color_continuous_scale='Viridis')
        st.plotly_chart(fig_heat, use_container_width=True)
        

    with tab4:
        st.subheader("Institutional AI Investment Thesis")
        target = st.selectbox("Select Project for Deep Dive", selected['Project_ID'])
        r = selected[selected['Project_ID'] == target].iloc[0]
        
        if st.button("Generate Strategic Memo"):
            with st.spinner("Accessing Gemini Institutional Logic..."):
                try:
                    # FIX: Dynamic Discovery to prevent 'NotFound' traceback
                    model_list = genai.list_models()
                    available = [m.name for m in model_list if 'generateContent' in m.supported_generation_methods]
                    chosen_model = "gemini-1.5-flash"
                    if f"models/{chosen_model}" not in available:
                        chosen_model = available[0].replace("models/", "")

                    model = genai.GenerativeModel(chosen_model)
                    prompt = f"""
                    Role: Senior Investment Committee Member. 
                    Project: {target} in {r['Department']}. 
                    Metrics: Capital ${r['Investment_Capital']}, Strategic NPV ${r['Strategic_Value']:.2f}, 
                    ESG {r['ESG_Score']}/10, PI {r['PI']:.2f}.
                    Context: WACC {wacc*100:.1f}%.
                    Write a 2-sentence institutional thesis and identify 1 hidden risk factor.
                    """
                    response = model.generate_content(prompt)
                    st.info(f"Generated via {chosen_model.upper()}")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"AI Connectivity Error: {e}")

if __name__ == "__main__":
    main()
