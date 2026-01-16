import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import pulp
import sqlite3

# ----------------------------------------------------
# 1. Page Configuration & Professional CSS
# ----------------------------------------------------
st.set_page_config(page_title="CAPITAL-AI | MSc Edition", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: #0f172a; }
    h1, h2, h3 { color: #f8fafc !important; font-weight: 700; }
    p, span, label { color: #cbd5e1 !important; }
    
    /* Metrics Card */
    div[data-testid="stMetric"] {
        background-color: #1e293b; 
        border: 1px solid #334155;
        border-radius: 8px; 
        padding: 20px;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #334155; }
    
    .brand-text {
        font-size: 24px; font-weight: 800;
        background: -webkit-linear-gradient(0deg, #38bdf8, #818cf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. AI Logic: The Linear & Tree Hybrid
# ----------------------------------------------------
def train_hybrid_models(df_hist):
    features = ['Investment_Capital', 'Risk_Score', 'Market_Trend_Index']
    target = 'Actual_ROI'
    
    # Model A: Linear Regression for Trend
    lr = LinearRegression()
    lr.fit(df_hist[features], df_hist[target])
    
    # Model B: Decision Tree for Non-Linear Risk
    dt = DecisionTreeRegressor(max_depth=4, random_state=42)
    dt.fit(df_hist[features], df_hist[target])
    
    return lr, dt

def run_optimization(df, budget):
    prob = pulp.LpProblem("Allocation", pulp.LpMaximize)
    vars = pulp.LpVariable.dicts("Select", df.index, cat='Binary')
    
    # Maximize Predicted ROI
    prob += pulp.lpSum([df.loc[i, 'Pred_ROI'] * vars[i] for i in df.index])
    # Within Budget
    prob += pulp.lpSum([df.loc[i, 'Investment_Capital'] * vars[i] for i in df.index]) <= budget
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    df['Selected'] = [int(vars[i].varValue) for i in df.index]
    return df

# ----------------------------------------------------
# 3. Sidebar & Parameters
# ----------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand-text">CAPITAL-AI ADVISOR</div>', unsafe_allow_html=True)
    st.caption("Advanced Decision Support System")
    st.markdown("---")
    budget = st.number_input("Total Capital Pool ($)", 1000000, 5000000, 2500000)
    risk_appetite = st.slider("Max Portfolio Risk", 1.0, 10.0, 5.0)
    st.markdown("---")
    st.info("Algorithms: OLS Regression & CART Decision Tree")

# ----------------------------------------------------
# 4. Mock Data Generation
# ----------------------------------------------------
@st.cache_data
def load_data():
    hist = pd.DataFrame({
        'Investment_Capital': np.random.randint(200, 1000, 100) * 1000,
        'Risk_Score': np.random.uniform(1, 10, 100),
        'Market_Trend_Index': np.random.uniform(0.5, 1.5, 100),
        'Actual_ROI': np.random.uniform(5, 25, 100)
    })
    prop = pd.DataFrame({
        'Project_ID': [f"PRJ-{i:02d}" for i in range(1, 13)],
        'Department': np.random.choice(['R&D', 'Ops', 'Marketing', 'IT'], 12),
        'Investment_Capital': np.random.randint(200, 800, 12) * 1000,
        'Risk_Score': np.random.uniform(1, 9, 12),
        'Market_Trend_Index': np.random.uniform(0.8, 1.4, 12)
    })
    return hist, prop

hist, prop = load_data()
lr_model, dt_model = train_hybrid_models(hist)

# Hybrid Prediction Logic
prop['LR_Pred'] = lr_model.predict(prop[['Investment_Capital', 'Risk_Score', 'Market_Trend_Index']])
prop['DT_Pred'] = dt_model.predict(prop[['Investment_Capital', 'Risk_Score', 'Market_Trend_Index']])
prop['Pred_ROI'] = (prop['LR_Pred'] + prop['DT_Pred']) / 2

# Filter by Risk Appetite before Optimization
prop = prop[prop['Risk_Score'] <= risk_appetite].reset_index(drop=True)
if not prop.empty:
    results = run_optimization(prop, budget)
    portfolio = results[results['Selected'] == 1]
else:
    results = prop
    portfolio = pd.DataFrame()

# ----------------------------------------------------
# 5. Main Dashboard UI
# ----------------------------------------------------
st.title("Executive Capital Allocation Dashboard")

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Recommended Allocation", f"${portfolio['Investment_Capital'].sum():,.0f}")
kpi2.metric("Projected Avg ROI", f"{portfolio['Pred_ROI'].mean():.2f}%" if not portfolio.empty else "0%")
kpi3.metric("Selection Yield", f"{len(portfolio)} / {len(results)} Projects")

st.markdown("---")

col_a, col_b = st.columns([2, 1])

with col_a:
    st.subheader("Optimized Investment Schedule")
    st.dataframe(
        results.style.background_gradient(subset=['Pred_ROI'], cmap='Blues')
        .format({'Investment_Capital': '${:,.0f}', 'Pred_ROI': '{:.2f}%'}),
        use_container_width=True
    )

with col_b:
    st.subheader("Risk-Return Topology")
    fig = px.scatter(results, x="Risk_Score", y="Pred_ROI", size="Investment_Capital", 
                     color="Selected", color_discrete_map={1: '#10b981', 0: '#f43f5e'},
                     template="plotly_dark")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)



st.subheader("Model Interpretation: Linear vs. Decision Tree Weights")
# Showing the coefficients of the Linear model
coef_df = pd.DataFrame({
    'Feature': ['Capital', 'Risk', 'Market Trend'],
    'Impact Weight': lr_model.coef_
})
fig_bar = px.bar(coef_df, x='Impact Weight', y='Feature', orientation='h', template="plotly_dark")
st.plotly_chart(fig_bar, use_container_width=True)



from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import io

# --- SECTION: EXPLAINABLE AI (XAI) ---
st.divider()
st.subheader("🧠 Model Explainability: Decision Tree Logic")
render_col, logic_col = st.columns([2, 1])

with render_col:
    # Creating a visual representation of how the tree splits data
    fig_tree, ax = plt.subplots(figsize=(12, 6))
    fig_tree.patch.set_facecolor('#0f172a')
    plot_tree(dt_model, 
              feature_names=['Capital', 'Risk', 'Trend'], 
              filled=True, 
              rounded=True, 
              fontsize=8,
              precision=2,
              ax=ax)
    # Adjusting colors for Dark Mode UI
    for text in ax.texts:
        text.set_color('white')
    st.pyplot(fig_tree)

with logic_col:
    st.markdown("#### Logic Breakdown")
    st.write("""
    The **CART Decision Tree** identifies non-linear 'breakpoints' in your data. 
    Unlike Linear Regression, which assumes a straight-line relationship, the tree 
    asks: *'Is the Risk Score > 6.5?'* This allows the advisor to penalize high-risk projects more heavily than a simple 
    formula would.
    """)

# --- SECTION: EXPORT & REPORTING ---
st.divider()
st.subheader("📑 Executive Reporting")
if not portfolio.empty:
    # Professional Narrative Generation
    avg_risk = portfolio['Risk_Score'].mean()
    total_npv = portfolio['Investment_Capital'].sum() * (portfolio['Pred_ROI'].mean()/100)
    
    report_text = f"""
    **OFFICIAL ADVISORY SUMMARY**
    
    Based on the Linear Regression baseline and Decision Tree volatility analysis, 
    the system recommends a capital deployment of **${portfolio['Investment_Capital'].sum():,.2f}**.
    
    The portfolio maintains a weighted average risk score of **{avg_risk:.2f}**, 
    which is aligned with current market trends ({prop['Market_Trend_Index'].mean():.2f}x benchmark).
    """
    st.info(report_text)
    
    # Download Button for CSV
    csv = portfolio.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Investment Schedule (CSV)",
        data=csv,
        file_name='strategic_allocation_plan.csv',
        mime='text/csv',
    )
