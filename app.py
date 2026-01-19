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
from sklearn.tree import DecisionTreeRegressor
import pulp
import google.generativeai as genai
from scipy.stats import norm
from fpdf import FPDF
import io

# ----------------------------------------------------
# 0. CONFIGURATION & MODELS
# ----------------------------------------------------
GEMINI_API_KEY = "AIzaSyDAz2r4IRQT5dv3zNq-uuRO7D2O86nNueE"
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="STRATOS QUANT | Enterprise Capital Advisor", layout="wide")

# Institutional Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono&family=Public+Sans:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Public Sans', sans-serif; background-color: #0d1117; }
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .main-title { font-size: 34px; font-weight: 800; letter-spacing: -1px; color: #ffffff; margin-bottom: 0px; }
    .metric-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; text-align: center; }
    .stButton>button { width: 100%; border-radius: 4px; background-color: #238636; color: white; border: none; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 1. FINANCE & ML LOGIC
# ----------------------------------------------------

def black_scholes_roa(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

@st.cache_data
def generate_dummy_data():
    """Restores the capability to generate high-quality dummy data if no file is uploaded."""
    data = {
        'Project_ID': [f"EXT-{i:03d}" for i in range(1, 26)],
        'Department': np.random.choice(['FinTech', 'Green Energy', 'Infrastructure', 'R&D', 'Operations'], 25),
        'Investment_Capital': np.random.choice([250000, 500000, 750000, 1000000, 1500000], 25),
        'Risk_Score': np.random.uniform(1.5, 9.5, 25),
        'ESG_Score': np.random.uniform(2, 10, 25),
        'Carbon_Offset_MT': np.random.uniform(50, 500, 25),
        'Market_Volatility': np.random.uniform(0.1, 0.45, 25),
        'Historical_Success_Rate': np.random.uniform(0.6, 0.95, 25),
        'Strategic_Alignment': np.random.randint(1, 11, 25)
    }
    # Create target ROI for training
    df = pd.DataFrame(data)
    df['Actual_ROI'] = (df['Strategic_Alignment'] * 2) + (df['Historical_Success_Rate'] * 10) - df['Risk_Score'] + np.random.normal(0, 2, 25)
    return df

def train_and_predict(df):
    features = ['Investment_Capital', 'Risk_Score', 'ESG_Score', 'Market_Volatility', 'Historical_Success_Rate']
    target = 'Actual_ROI'
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(df[features], df[target])
    
    df['Pred_ROI'] = rf.predict(df[features])
    importances = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_})
    return df, importances

# ----------------------------------------------------
# 2. APP FLOW
# ----------------------------------------------------

def main():
    st.markdown('<p class="main-title">STRATOS | Capital Allocation Advisor</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8b949e;">Multi-Criteria AI Optimization & Strategic Risk Management</p>', unsafe_allow_html=True)
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("1. Data Ingestion")
        uploaded_file = st.file_uploader("Upload Project Proposals (CSV)", type=["csv"])
        load_dummy = st.checkbox("Use Strategic Dummy Data", value=not uploaded_file)
        
        st.header("2. Macro Parameters")
        rf_rate = st.slider("Risk-Free Rate (Rf)", 0.0, 8.0, 4.2) / 100
        mkt_prem = st.slider("Equity Risk Premium", 3.0, 10.0, 5.5) / 100
        
        st.header("3. Allocation Constraints")
        budget = st.number_input("Capital Pool ($)", value=5000000)
        min_esg = st.slider("Min Portfolio ESG Score", 1, 10, 6)
        
    # Data Loading Logic
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    elif load_dummy:
        df = generate_dummy_data()
    else:
        st.info("Please upload data or select 'Use Dummy Data' to begin.")
        st.stop()

    # ML ROI Prediction
    df, feature_importance = train_and_predict(df)
    
    # Financial Engineering Calculations
    wacc = rf_rate + (1.2 * mkt_prem)
    df['ROA_Value'] = df.apply(lambda x: black_scholes_roa(x['Investment_Capital']*1.3, x['Investment_Capital'], 2, rf_rate, x['Market_Volatility']), axis=1)
    df['NPV'] = (df['Investment_Capital'] * (df['Pred_ROI']/100)) / wacc
    df['Strategic_NPV'] = df['NPV'] + df['ROA_Value']
    df['Profitability_Index'] = df['Strategic_NPV'] / df['Investment_Capital']
    
    # Optimization (Mixed Integer Programming)
    prob = pulp.LpProblem("Capital_Allocation", pulp.LpMaximize)
    xs = pulp.LpVariable.dicts("Select", df.index, cat=pulp.LpBinary)
    
    prob += pulp.lpSum([df.loc[i, 'Strategic_NPV'] * xs[i] for i in df.index])
    prob += pulp.lpSum([df.loc[i, 'Investment_Capital'] * xs[i] for i in df.index]) <= budget
    prob += pulp.lpSum([df.loc[i, 'ESG_Score'] * xs[i] for i in df.index]) >= min_esg * pulp.lpSum([xs[i] for i in df.index])
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    df['Selected'] = [pulp.value(xs[i]) for i in df.index]
    selected_df = df[df['Selected'] == 1]

    # --- MAIN DASHBOARD ---
    t1, t2, t3, t4 = st.tabs(["🚀 Portfolio Summary", "📈 ML Insights", "🛡️ Stress Test", "📝 AI Memo"])
    
    with t1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Projects Funded", f"{len(selected_df)} / {len(df)}")
        c2.metric("Budget Utilization", f"${selected_df['Investment_Capital'].sum():,.0f}", f"{selected_df['Investment_Capital'].sum()/budget*100:.1f}%")
        c3.metric("Agg. Strategic NPV", f"${selected_df['Strategic_NPV'].sum():,.0f}")
        c4.metric("Avg ESG Impact", f"{selected_df['ESG_Score'].mean():.1f}")
        
        st.write("### Optimized Investment Schedule")
        st.dataframe(selected_df[['Project_ID', 'Department', 'Investment_Capital', 'Pred_ROI', 'Profitability_Index', 'ESG_Score']].style.background_gradient(cmap='Greens'))
        
        # Download Data
        csv = selected_df.to_csv(index=False).encode('utf-8')
        st.download_button("Export Optimized Portfolio (CSV)", csv, "portfolio.csv", "text/csv")

    with t2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Model Explainability: Feature Importance**")
            fig_imp = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig_imp, use_container_width=True)
        with col_b:
            st.write("**Strategic Frontier: Value vs ESG**")
            fig_scat = px.scatter(df, x="ESG_Score", y="Strategic_NPV", size="Investment_Capital", color="Selected",
                                  hover_name="Project_ID", color_discrete_map={1:'#00e676', 0:'#ff1744'})
            st.plotly_chart(fig_scat, use_container_width=True)

    with t3:
        st.subheader("Monte Carlo Simulation (1,000 Paths)")
        # Simulating volatility in NPV
        sim_results = []
        for _ in range(1000):
            shocks = np.random.normal(0, selected_df['Market_Volatility'].mean(), len(selected_df))
            sim_npv = (selected_df['Strategic_NPV'] * (1 + shocks)).sum()
            sim_results.append(sim_npv)
        
        fig_sim = px.histogram(sim_results, title="Portfolio Value Distribution under Volatility Stress", color_discrete_sequence=['#3b82f6'])
        st.plotly_chart(fig_sim, use_container_width=True)
        st.write(f"**Value-at-Risk (VaR) 95%:** ${np.percentile(sim_results, 5):,.0f}")

    with t4:
        st.subheader("Institutional AI Investment Thesis")
        target_p = st.selectbox("Select Project for Deep Dive", selected_df['Project_ID'])
        row = selected_df[selected_df['Project_ID'] == target_p].iloc[0]
        
        if st.button("Generate Institutional Memo"):
            with st.spinner("AI analyzing project fundamentals..."):
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"""
                Analyze investment project {target_p}. 
                Fundamentals: Capital ${row['Investment_Capital']}, Predicted ROI {row['Pred_ROI']:.2f}%, 
                ESG {row['ESG_Score']}/10, Profitability Index {row['Profitability_Index']:.2f}.
                Context: WACC is {wacc*100:.1f}%. 
                Write a 3-bullet thesis: 1. Financial Viability, 2. ESG/Strategic alignment, 3. Critical Risk Factor.
                """
                response = model.generate_content(prompt)
                st.info(response.text)

if __name__ == "__main__":
    main()
