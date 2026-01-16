import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pulp
import google.generativeai as genai
import plotly.express as px

# --- INITIAL CONFIG ---
st.set_page_config(page_title="AI Capital Allocator", layout="wide")

# Setup Gemini (User provides API key in sidebar)
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
if api_key:
    genai.configure(api_key=api_key)

# --- HELPER FUNCTIONS ---
def calculate_npv(row, wacc=0.1):
    """Simple NPV calculation: Total Returns / (1 + WACC)^Years - Initial Cost"""
    years = row['Project_Duration']
    return (row['Expected_Annual_Return'] * ((1 - (1 + wacc)**-years) / wacc)) - row['Cost']

def run_optimization(df, budget):
    """Linear Programming to maximize NPV under budget constraints"""
    prob = pulp.LpProblem("Capital_Allocation", pulp.LpMaximize)
    
    # Decision variables: 1 if we invest, 0 otherwise
    decisions = pulp.LpVariable.dicts("Invest", df.index, cat='Binary')
    
    # Objective: Maximize total Predicted NPV
    prob += pulp.lpSum([decisions[i] * df.loc[i, 'Predicted_NPV'] for i in df.index])
    
    # Constraint: Total cost <= Budget
    prob += pulp.lpSum([decisions[i] * df.loc[i, 'Cost'] for i in df.index]) <= budget
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return [pulp.value(decisions[i]) for i in df.index]

def get_ai_justification(project_name, roi, npv, risk_score):
    """Generate LLM commentary for the project decision"""
    if not api_key: return "API Key missing."
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Act as a CFO. Project: {project_name}. Predicted ROI: {roi}%, NPV: ${npv}, Risk: {risk_score}/10.
    Provide a concise (2 sentence) investment justification and one specific risk.
    Format: Justification: [text] | Risk: [text]
    """
    response = model.generate_content(prompt)
    return response.text

# --- APP UI ---
st.title("🚀 AI-Driven Capital Allocation Advisor")
st.markdown("Optimization of project portfolios using Machine Learning and Linear Programming.")

# 1. DATA INPUT
st.header("1. Project Portfolio Input")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Project Proposals")
    # Sample data generator for demo purposes
    if st.button("Generate Sample Data"):
        sample_df = pd.DataFrame({
            'Project_Name': [f"Project {chr(65+i)}" for i in range(8)],
            'Cost': [100000, 250000, 50000, 150000, 300000, 120000, 80000, 200000],
            'Expected_Annual_Return': [25000, 70000, 15000, 40000, 65000, 30000, 22000, 55000],
            'Project_Duration': [5, 4, 3, 5, 6, 4, 3, 5],
            'Risk_Score': [3, 7, 2, 5, 8, 4, 3, 6]
        })
        st.session_state['data'] = sample_df
    
    uploaded_file = st.file_uploader("Upload Proposals CSV", type="csv")
    if uploaded_file:
        st.session_state['data'] = pd.read_csv(uploaded_file)

if 'data' in st.session_state:
    df = st.session_state['data']
    st.dataframe(df, use_container_width=True)

    # 2. ML & FINANCE LOGIC
    st.header("2. Financial Analysis & Prediction")
    wacc = st.slider("Cost of Capital (WACC %)", 5, 20, 10) / 100
    budget = st.number_input("Total Investment Budget ($)", min_value=10000, value=500000)

    # Calculate NPV and Mock ROI for ML Training
    df['NPV'] = df.apply(lambda x: calculate_npv(x, wacc), axis=1)
    df['ROI_Calculated'] = (df['NPV'] / df['Cost']) * 100

    # Train a simple Random Forest to 'predict' NPV based on Cost, Risk, and Duration
    X = df[['Cost', 'Project_Duration', 'Risk_Score']]
    y = df['NPV']
    model = RandomForestRegressor(n_estimators=100).fit(X, y)
    df['Predicted_NPV'] = model.predict(X)

    # 3. OPTIMIZATION
    st.header("3. Optimal Allocation (The 'Greenlight' List)")
    if st.button("Run Optimizer"):
        df['Decision_Binary'] = run_optimization(df, budget)
        df['Decision'] = df['Decision_Binary'].apply(lambda x: "✅ INVEST" if x == 1 else "❌ REJECT")
        
        funded_df = df[df['Decision_Binary'] == 1]
        
        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("Total NPV Created", f"${funded_df['NPV'].sum():,.2f}")
        k2.metric("Budget Utilized", f"${funded_df['Cost'].sum():,.2f}")
        k3.metric("Projects Funded", len(funded_df))

        # 4. AI JUSTIFICATION
        st.subheader("CFO Insight (Gen-AI)")
        with st.spinner("Generating AI justifications..."):
            ai_summaries = []
            for idx, row in df.iterrows():
                if row['Decision_Binary'] == 1:
                    summary = get_ai_justification(row['Project_Name'], round(row['ROI_Calculated'],2), round(row['NPV'],2), row['Risk_Score'])
                    ai_summaries.append(summary)
                else:
                    ai_summaries.append("Project exceeds budget or ROI threshold.")
            df['AI_Insight'] = ai_summaries

        st.dataframe(df[['Project_Name', 'Cost', 'Decision', 'AI_Insight']], use_container_width=True)

        # Visualization
        fig = px.bar(df, x='Project_Name', y='NPV', color='Decision', 
                     title="Project Value vs Decision", barmode='group')
        st.plotly_chart(fig, use_container_width=True)

        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Allocation Report", data=csv, file_name="allocation_report.csv", mime="text/csv")
