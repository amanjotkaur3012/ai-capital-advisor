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

# ----------------------------------------------------
# 0. CORE CONFIGURATION & API
# ----------------------------------------------------
GEMINI_API_KEY = "AIzaSyBEFV8Q9A9DvRUMzB9tD3KlUPvsv_60j60"
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="STRATOS QUANT | MSc Capital Advisor", layout="wide")

# Bloomberg Terminal Style CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0b0f19; }
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    .main-title { 
        font-size: 32px; font-weight: 800; letter-spacing: -1px; 
        background: linear-gradient(90deg, #10b981, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    div[data-testid="stMetric"] {
        background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 1. QUANTITATIVE FINANCE ENGINES
# ----------------------------------------------------

def black_scholes_roa(S, K, T, r, sigma):
    """MSc Finance: Real Options Value (Option to Expand)"""
    if T <= 0 or sigma <= 0 or S <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def run_optimization(df, budget, esg_hurdle):
    """Mixed Integer Programming for constrained capital allocation."""
    prob = pulp.LpProblem("Cap_Alloc", pulp.LpMaximize)
    xs = pulp.LpVariable.dicts("Select", df.index, cat=pulp.LpBinary)
    prob += pulp.lpSum([df.loc[i, 'Strategic_Value'] * xs[i] for i in df.index])
    prob += pulp.lpSum([df.loc[i, 'Investment_Capital'] * xs[i] for i in df.index]) <= budget
    prob += pulp.lpSum([df.loc[i, 'ESG_Score'] * xs[i] for i in df.index]) >= esg_hurdle * pulp.lpSum([xs[i] for i in df.index])
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    df['Selected'] = [pulp.value(xs[i]) for i in df.index]
    return df

# ----------------------------------------------------
# 2. MAIN APPLICATION
# ----------------------------------------------------

def main():
    st.markdown('<p class="main-title">STRATOS | Institutional Capital Advisor</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8b949e;">Multi-Criteria AI Optimization & Strategic Risk Management</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Data Control")
        up_file = st.file_uploader("Upload Portfolio CSV", type="csv")
        use_demo = st.checkbox("Load Institutional Data", value=not up_file)
        
        st.header("Macro Drivers")
        rf_rate = st.slider("Risk-Free Rate (Rf)", 0.0, 8.0, 4.2) / 100
        budget = st.number_input("Capital Pool ($)", value=5000000, step=500000)
        esg_min = st.slider("ESG Portfolio Hurdle", 1, 10, 6)

    # Dataset Logic
    if up_file:
        df = pd.read_csv(up_file)
    else:
        np.random.seed(42)
        df = pd.DataFrame({
            'Project_ID': [f"PRJ-{i:03d}" for i in range(1, 26)],
            'Department': np.random.choice(['FinTech', 'Green Energy', 'Infra', 'R&D'], 25),
            'Investment_Capital': np.random.choice([250000, 500000, 800000, 1500000], 25),
            'Risk_Score': np.random.uniform(2.0, 9.5, 25),
            'ESG_Score': np.random.uniform(2.0, 10.0, 25),
            'Market_Volatility': np.random.uniform(0.1, 0.45, 25),
            'Strategic_Alignment': np.random.randint(3, 11, 25)
        })
        df['Actual_ROI'] = (df['Strategic_Alignment'] * 2) - (df['Risk_Score'] * 0.4) + 10 + np.random.normal(0, 1, 25)

    # ML ROI Forecasting
    features = ['Investment_Capital', 'Risk_Score', 'ESG_Score', 'Market_Volatility', 'Strategic_Alignment']
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(df[features], df['Actual_ROI'])
    df['Pred_ROI'] = rf_model.predict(df[features])

    # Financial Engineering
    wacc = rf_rate + 0.06
    df['ROA_Value'] = df.apply(lambda x: black_scholes_roa(x['Investment_Capital']*1.25, x['Investment_Capital'], 2, rf_rate, x['Market_Volatility']), axis=1)
    df['NPV'] = (df['Investment_Capital'] * (df['Pred_ROI']/100)) / wacc
    df['Strategic_Value'] = df['NPV'] + df['ROA_Value']
    df['PI'] = df['Strategic_Value'] / df['Investment_Capital']

    # Optimization Execution
    df = run_optimization(df, budget, esg_min)
    selected = df[df['Selected'] == 1]

    # --- DASHBOARD TABS ---
    t1, t2, t3, t4, t5 = st.tabs(["🚀 Summary", "🧠 ML Analytics", "📊 Sensitivity", "🛡️ Risk Quadrant", "🤖 AI Thesis"])
    
    with t1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capital Allocated", f"${selected['Investment_Capital'].sum():,.0f}", f"{selected['Investment_Capital'].sum()/budget*100:.1f}%")
        c2.metric("Portfolio Strategic NPV", f"${selected['Strategic_Value'].sum():,.0f}")
        c3.metric("Avg ESG Impact", f"{selected['ESG_Score'].mean():.1f}")
        c4.metric("Agg. PI", f"{selected['PI'].mean():.2f}x")
        st.dataframe(selected[['Project_ID', 'Department', 'Investment_Capital', 'Pred_ROI', 'PI', 'ESG_Score']].style.background_gradient(cmap='Blues'))

    with t2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Feature Importance (Predictive ROI)**")
            feat_imp = pd.DataFrame({'Feature': features, 'Importance': rf_model.feature_importances_})
            st.plotly_chart(px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color_continuous_scale='Viridis'), use_container_width=True)
        with col_b:
            st.write("**Efficient Frontier**")
            st.plotly_chart(px.scatter(df, x="ESG_Score", y="Strategic_Value", color="Selected", size="Investment_Capital"), use_container_width=True)

    with t3:
        st.subheader("Sensitivity Matrix: Budget vs ESG")
        b_range = np.linspace(budget * 0.8, budget * 1.2, 5)
        e_range = np.linspace(4, 9, 5)
        h_map = []
        for b in b_range:
            row = []
            for e in e_range:
                temp = run_optimization(df.copy(), b, e)
                row.append(temp[temp['Selected'] == 1]['Strategic_Value'].sum())
            h_map.append(row)
        
        fig_heat = px.imshow(h_map, x=[f"ESG {e:.1f}" for e in e_range], y=[f"${b/1e6:.1f}M" for b in b_range], color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_heat, use_container_width=True)

    with t4:
        st.subheader("Strategic Risk-Return Quadrant")
        
        # Calculate Medians for clean quadrant lines
        median_risk = df['Risk_Score'].median()
        median_pi = df['PI'].median()
        
        # Create the scatter plot with improved labeling
        fig_quad = px.scatter(
            df, 
            x="Risk_Score", 
            y="PI", 
            color="Selected", 
            size="Investment_Capital",
            hover_name="Project_ID", 
            text="Project_ID", # Keep labels but we will style them
            color_discrete_map={1: '#10b981', 0: '#334155'},
            labels={"PI": "Profitability Index (Value)", "Risk_Score": "Risk Rating (1-10)"},
            category_orders={"Selected": [1, 0]}
        )
        
        # CLEANUP: Position labels so they don't sit directly on the bubbles
        fig_quad.update_traces(
            textposition='top center',
            marker=dict(line=dict(width=1, color='DarkSlateGrey')),
            selector=dict(mode='markers+text')
        )

        # ADD STRATEGIC REGIONS: Using Shapes to color-code quadrants
        fig_quad.add_hrect(y0=median_pi, y1=df['PI'].max()*1.1, x0=0, x1=median_risk, 
                          fillcolor="green", opacity=0.05, layer="below", line_width=0,
                          annotation_text="STRATEGIC CORE")
        
        fig_quad.add_hrect(y0=0, y1=median_pi, x0=median_risk, x1=10, 
                          fillcolor="red", opacity=0.05, layer="below", line_width=0,
                          annotation_text="VALUE TRAPS")

        # Layout Refinement
        fig_quad.update_layout(
            showlegend=True,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(gridcolor='#30363d', zeroline=False),
            yaxis=dict(gridcolor='#30363d', zeroline=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_quad, use_container_width=True)
        
        # ADD AN EXPLANATORY LEGEND BELOW THE CHART
        st.markdown("""
        #### 🔍 How to Read This Quadrant
        * **Top-Left (High Value, Low Risk):** Priority funding. These projects offer high returns relative to their capital cost.
        * **Top-Right (High Value, High Risk):** Speculative growth. High potential but requires strict risk-mitigation.
        * **Bottom-Left (Low Value, Low Risk):** Defensive plays. Safe but offer marginal strategic wealth creation.
        * **Bottom-Right (Low Value, High Risk):** **Value Traps.** Avoid these; they consume capital without adequate risk-adjusted returns.
        """)

    with t5:
        st.subheader("AI Dealer Memo")
        target = st.selectbox("Select Project", selected['Project_ID'])
        r = selected[selected['Project_ID'] == target].iloc[0]
        
        if st.button("Generate Memo"):
            with st.spinner("Accessing Gemini..."):
                try:
                    # DYNAMIC DISCOVERY: Fixes the 'NotFound' error by checking authorized models
                    model_list = genai.list_models()
                    available = [m.name for m in model_list if 'generateContent' in m.supported_generation_methods]
                    model_name = "gemini-1.5-flash" if f"models/gemini-1.5-flash" in available else available[0].replace("models/", "")
                    
                    llm = genai.GenerativeModel(model_name)
                    prompt = f"Role: Senior Quant. Project {target}. PI {r['PI']:.2f}, ESG {r['ESG_Score']}. Write a 3-sentence investment thesis."
                    res = llm.generate_content(prompt)
                    st.success(f"Analyst: {model_name.upper()}")
                    st.write(res.text)
                except Exception as e:
                    st.error(f"AI Connectivity Error: {e}")

if __name__ == "__main__":
    main()
