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
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestRegressor
import pulp
import sqlite3
import google.generativeai as genai

# ----------------------------------------------------
# 1. Page Configuration & CSS
# ----------------------------------------------------
st.set_page_config(
    page_title="FINCAP-AI | AI Capital Allocation Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊" 
)

# --- THEME ENGINE: ULTRA-CLEAN CORPORATE DARK ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Global Settings */
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear_gradient(135deg, #0f172a 0%, #1e293b 100%);
    }

    /* Text Visibility - Corporate White/Grey */
    h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; font-weight: 700; letter-spacing: -0.5px; }
    p, li, span, label, div[data-testid="stMarkdownContainer"] p { color: #cbd5e1 !important; }
    
    /* Metrics & Cards */
    div[data-testid="stMetric"], div[data-testid="stExpander"], div.stDataFrame {
        background-color: #1e293b; 
        border: 1px solid #334155;
        border-radius: 6px; 
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a; 
        border-right: 1px solid #334155;
    }
    
    /* Custom Logo Text */
    .brand-text {
        font-size: 26px;
        font-weight: 800;
        background: -webkit-linear-gradient(0deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1px;
        text-align: center;
        margin-bottom: 5px;
    }
    .brand-sub {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #64748b;
        text-align: center;
        margin-bottom: 30px;
    }

    /* Button Styling */
    .stButton>button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 1px;
        height: 45px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: #1d4ed8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* Analysis Box Styling */
    .analysis-box {
        border-left: 4px solid #38bdf8;
        background-color: rgba(56, 189, 248, 0.05);
        padding: 20px;
        margin-top: 15px;
        border-radius: 0 6px 6px 0;
    }
    .analysis-title {
        font-weight: 700;
        color: #38bdf8 !important;
        margin-bottom: 8px;
        display: block;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 1.5px;
    }
    .analysis-text {
        font-size: 14px;
        line-height: 1.6;
        color: #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Database & Auth Layer
# ----------------------------------------------------
DB_FILE = "capital_planning.db"

def init_db():
    """Creates a local SQL database for persistence."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS scenarios 
                 (id INTEGER PRIMARY KEY, name TEXT, date TEXT, budget REAL, 
                  npv REAL, roi REAL, projects_count INTEGER, wacc REAL)''')
    conn.commit()
    conn.close()

def save_scenario_db(name, budget, npv, roi, count, wacc):
    """Saves scenario to SQLite."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO scenarios (name, date, budget, npv, roi, projects_count, wacc) VALUES (?, datetime('now', 'localtime'), ?, ?, ?, ?, ?)",
              (name, budget, npv, roi, count, wacc))
    conn.commit()
    conn.close()

def get_scenarios_db():
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql("SELECT * FROM scenarios ORDER BY id DESC", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

init_db() # Initialize on load

def check_login():
    """Simple Login Wrapper."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        return False
    return True

# ----------------------------------------------------
# 3. Helper Functions (Logic & AI)
# ----------------------------------------------------

def standardize_columns(df):
    """Maps various user inputs to system standard names."""
    column_map = {
        'Capex': 'Investment_Capital', 'Cost': 'Investment_Capital', 'Budget': 'Investment_Capital',
        'Return': 'Actual_ROI_Pct', 'ROI': 'Actual_ROI_Pct',
        'Strategy': 'Strategic_Alignment', 'Strat': 'Strategic_Alignment',
        'Risk': 'Risk_Score', 'Dept': 'Department', 
        'Mandatory': 'Is_Mandatory', 'Group': 'Exclusion_Group'
    }
    df = df.rename(columns=column_map)
    
    # Ensure Advanced Columns exist
    if 'Is_Mandatory' not in df.columns: df['Is_Mandatory'] = 0
    if 'Exclusion_Group' not in df.columns: df['Exclusion_Group'] = 0
    if 'Project_Description' not in df.columns: df['Project_Description'] = "Standard Project"

    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

@st.cache_data
def get_templates():
    """Generates sample CSV templates."""
    df_h = pd.DataFrame({
        "Investment_Capital": np.random.randint(500000, 5000000, 50),
        "Duration_Months": np.random.randint(6, 36, 50),
        "Risk_Score": np.random.uniform(1, 10, 50).round(1),
        "Strategic_Alignment": np.random.uniform(1, 10, 50).round(1),
        "Market_Trend_Index": np.random.uniform(0.5, 1.5, 50).round(2),
        "Actual_ROI_Pct": np.random.uniform(5, 25, 50).round(1),
        "Actual_NPV": np.random.randint(100000, 2000000, 50)
    })
    
    df_p = pd.DataFrame({
        "Project_ID": [f"PROJ-{i:03d}" for i in range(1, 21)],
        "Department": np.random.choice(['IT', 'R&D', 'Marketing', 'Ops'], 20),
        "Investment_Capital": np.random.randint(500000, 5000000, 20),
        "Duration_Months": np.random.randint(6, 36, 20),
        "Risk_Score": np.random.uniform(1, 10, 20).round(1),
        "Strategic_Alignment": np.random.uniform(1, 10, 20).round(1),
        "Market_Trend_Index": np.random.uniform(0.5, 1.5, 20).round(2),
        "Is_Mandatory": np.random.choice([0, 1], 20, p=[0.9, 0.1]), 
        "Exclusion_Group": 0 
    })
    return df_h, df_p

@st.cache_resource
def train_models(df_hist):
    features = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]
    for f in features:
        if f not in df_hist.columns: df_hist[f] = 0
            
    rf_roi = RandomForestRegressor(n_estimators=200, random_state=42)
    try:
        rf_roi.fit(df_hist[features], df_hist["Actual_ROI_Pct"])
    except Exception as e:
        st.error(f"Training Error: {e}")
        st.stop()
    
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_roi.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return rf_roi, importance

def calculate_dynamic_npv(row, wacc_rate):
    total_return_value = row['Investment_Capital'] * (1 + (row['Pred_ROI'] / 100))
    duration = max(row['Duration_Months'], 1)
    annual_cash_flow = total_return_value / (duration / 12)
    years = duration / 12
    dcf = 0
    for t in range(1, int(years) + 2):
        if t <= years:
            dcf += annual_cash_flow / ((1 + wacc_rate) ** t)
        else:
            remaining_fraction = years - int(years)
            if remaining_fraction > 0:
                dcf += (annual_cash_flow * remaining_fraction) / ((1 + wacc_rate) ** t)
    return dcf - row['Investment_Capital']

def calculate_payback(row):
    total_return_value = row['Investment_Capital'] * (1 + (row['Pred_ROI'] / 100))
    annual_cash_flow = total_return_value / (max(row['Duration_Months'], 1) / 12)
    if annual_cash_flow <= 0: return 99.9 
    return round(row['Investment_Capital'] / annual_cash_flow, 2)

def run_advanced_optimization(df, budget, min_dept_alloc_pct=0.0):
    prob = pulp.LpProblem("Capital_Allocation", pulp.LpMaximize)
    selection_vars = pulp.LpVariable.dicts("Select", df.index, cat='Binary')
    
    prob += pulp.lpSum([df.loc[i, "Dynamic_NPV"] * selection_vars[i] for i in df.index])
    prob += pulp.lpSum([df.loc[i, "Investment_Capital"] * selection_vars[i] for i in df.index]) <= budget
    
    # 1. Mandatory Constraints
    if 'Is_Mandatory' in df.columns:
        for i in df.index:
            if df.loc[i, 'Is_Mandatory'] == 1:
                prob += selection_vars[i] == 1

    # 2. Mutually Exclusive Constraints
    if 'Exclusion_Group' in df.columns:
        groups = df[df['Exclusion_Group'] > 0]['Exclusion_Group'].unique()
        for g in groups:
            group_indices = df[df['Exclusion_Group'] == g].index
            prob += pulp.lpSum([selection_vars[i] for i in group_indices]) <= 1

    # 3. Department Constraints
    if min_dept_alloc_pct > 0 and 'Department' in df.columns:
        departments = df['Department'].unique()
        for dept in departments:
            dept_indices = df[df['Department'] == dept].index
            dept_req = df.loc[dept_indices, "Investment_Capital"].sum()
            target_min = budget * min_dept_alloc_pct
            if dept_req >= target_min:
                prob += pulp.lpSum([df.loc[i, "Investment_Capital"] * selection_vars[i] for i in dept_indices]) >= target_min

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    df["Selected"] = [int(selection_vars[i].varValue) if selection_vars[i].varValue is not None else 0 for i in df.index]
    return df

def generate_ai_memo_text(row, api_key):
    """Real AI Integration via Google Gemini (Auto-Model Selector)"""
    if not api_key:
        return f"Decision: APPROVED\n\nThis project exceeds the required risk-adjusted return threshold and aligns with strategic goals."
    
    try:
        genai.configure(api_key=api_key)
        
        # 1. Try to list available models to find a valid one
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 2. Pick the best available model (Flash > Pro > Default)
        model_name = 'gemini-1.5-flash' # Default target
        if 'models/gemini-1.5-flash' in available_models:
            model_name = 'gemini-1.5-flash'
        elif 'models/gemini-pro' in available_models:
            model_name = 'gemini-pro'
        elif available_models:
            model_name = available_models[0].replace('models/', '') # Fallback to first available
        
        # 3. Generate
        model = genai.GenerativeModel(model_name) 
        prompt = f"""
        Act as a cynical Investment Committee member. Analyze this project:
        Desc: {row.get('Project_Description', 'Standard Project')}
        Cost: {row['Investment_Capital']}
        ROI: {row['Pred_ROI']}%
        Risk: {row['Risk_Score']}/10
        
        Write a concise, 1-sentence approval reason and 1 major risk warning.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

# --- VISUALIZATIONS ---

def generate_professional_waterfall(portfolio):
    base_npv = portfolio['Dynamic_NPV'].sum()
    wacc_impact = -(base_npv * 0.08)
    inflation_impact = -(base_npv * 0.05)
    synergy_gain = (base_npv * 0.12)
    final_npv = base_npv + wacc_impact + inflation_impact + synergy_gain
    
    fig = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = ["relative", "relative", "relative", "relative", "total"],
        x = ["Base Projection", "Cost of Capital Risk", "Inflation Drag", "Synergy Upside", "Risk-Adjusted Final"],
        textposition = "outside",
        text = [f"{base_npv/1e6:.1f}M", f"{wacc_impact/1e6:.1f}M", f"{inflation_impact/1e6:.1f}M", f"{synergy_gain/1e6:.1f}M", f"{final_npv/1e6:.1f}M"],
        y = [base_npv, wacc_impact, inflation_impact, synergy_gain, 0],
        connector = {"line":{"color":"rgb(150, 150, 150)", "width": 1}},
        decreasing = {"marker":{"color":"#f43f5e"}}, # Soft Red
        increasing = {"marker":{"color":"#10b981"}}, # Emerald Green
        totals = {"marker":{"color":"#3b82f6"}}      # Corporate Blue
    ))
    fig.update_layout(
        title = "Financial Value Bridge (Sensitivity Analysis)", 
        showlegend = False, 
        template="plotly_dark", 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)", 
        font=dict(color="#e2e8f0")
    )
    return fig

def generate_benchmark_radar(df_prop):
    """2D Radar Chart."""
    categories = ["Risk Safety (Inv)", "Strategic Fit", "ROI Potential"]
    
    df_norm = df_prop.copy()
    df_norm["Risk Safety (Inv)"] = 11 - df_norm["Risk_Score"] 
    df_norm["Strategic Fit"] = df_norm["Strategic_Alignment"]
    df_norm["ROI Potential"] = df_norm["Pred_ROI"]
    
    max_roi = df_norm["ROI Potential"].max()
    if max_roi > 0:
        df_norm["ROI Potential"] = (df_norm["ROI Potential"] / max_roi) * 10
    
    funded = df_norm[df_norm["Selected"] == 1][categories].mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[8, 8, 8],
        theta=categories,
        fill='toself',
        name='Corporate Benchmark',
        line_color='rgba(255, 255, 255, 0.3)',
        line_dash='dot'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[funded["Risk Safety (Inv)"], funded["Strategic Fit"], funded["ROI Potential"]],
        theta=categories,
        fill='toself',
        name='Selected Portfolio',
        line_color='#00e676',
        opacity=0.8
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], showticklabels=False),
            bgcolor="rgba(255,255,255,0.05)"
        ),
        title="Portfolio Quality vs. Benchmark",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def generate_3d_strategic_triangle(df_prop):
    """3D Scatter Triangle for Topology."""
    funded = df_prop[df_prop["Selected"] == 1].copy()
    funded["Risk_Inv"] = 10 - funded["Risk_Score"] 
    
    fig = go.Figure(data=[go.Scatter3d(
        x=funded['Strategic_Alignment'],
        y=funded['Pred_ROI'],
        z=funded['Risk_Inv'],
        mode='markers',
        marker=dict(
            size=12,
            color=funded['Dynamic_NPV'],
            colorscale='Viridis',
            opacity=0.8
        ),
        text=funded['Project_ID'],
        hovertemplate='<b>%{text}</b><br>Strat: %{x}<br>ROI: %{y}<br>Safe Score: %{z}<extra></extra>'
    )])

    fig.update_layout(
        title="3D Strategic Triangle (Strategy vs ROI vs Risk)",
        scene = dict(
            xaxis_title='Strategy Fit',
            yaxis_title='ROI (%)',
            zaxis_title='Safety (Inv Risk)',
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="gray"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="gray"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="gray"),
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def generate_combo_scenario_chart(scenarios):
    """
    FIXED: Handles SQLite output (lowercase keys) correctly.
    """
    df = pd.DataFrame(scenarios)
    
    # 1. Normalize columns to lowercase to avoid KeyError if DB returns 'npv' but we ask for 'NPV'
    df.columns = [c.lower() for c in df.columns]
    
    fig = go.Figure()
    # 2. Use lowercase keys matching the normalized columns
    fig.add_trace(go.Bar(x=df["name"], y=df["npv"], name="Net Present Value (NPV)", marker_color="#3b82f6", yaxis="y1"))
    fig.add_trace(go.Scatter(x=df["name"], y=df["roi"], name="Avg ROI %", mode='lines+markers', line=dict(color="#fbbf24", width=3), yaxis="y2"))
    
    fig.update_layout(
        title="Scenario Analysis: Value vs Efficiency", 
        yaxis=dict(title="NPV (Currency)", side="left", showgrid=False), 
        yaxis2=dict(title="ROI (%)", side="right", overlaying="y", showgrid=False), 
        template="plotly_dark", 
        paper_bgcolor="rgba(0,0,0,0)", 
        font=dict(color="#e2e8f0"), 
        legend=dict(orientation="h", y=-0.2)
    )
    return fig

# --- PDF GENERATOR ---
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
    
    class PDFReport(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'CAPITALIQ-AI | Strategic Portfolio Report', 0, 1, 'C')
            self.line(10, 20, 200, 20)
            self.ln(10)

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.set_fill_color(200, 220, 255)
            self.cell(0, 6, title, 0, 1, 'L', 1)
            self.ln(4)

        def chapter_body(self, body):
            self.set_font('Arial', '', 10)
            self.multi_cell(0, 5, body)
            self.ln()

    def create_pdf_report(portfolio, rejected, budget, wacc):
        pdf = PDFReport()
        pdf.add_page()
        pdf.chapter_title('1. Capital Allocation Overview')
        summary = f"""Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}\nTotal Budget Authorized: INR {budget:,.2f}\nCapital Deployed: INR {portfolio['Investment_Capital'].sum():,.2f}\nTotal Portfolio NPV: INR {portfolio['Dynamic_NPV'].sum():,.2f}\nWACC: {wacc*100:.1f}%\nAverage ROI: {portfolio['Pred_ROI'].mean():.2f}%"""
        pdf.chapter_body(summary)
        pdf.chapter_title('2. Approved Investment Schedule')
        pdf.set_font('Arial', 'B', 9)
        pdf.cell(40, 7, 'Project ID', 1); pdf.cell(50, 7, 'Department', 1); pdf.cell(50, 7, 'Investment (INR)', 1); pdf.cell(30, 7, 'ROI (%)', 1); pdf.ln()
        pdf.set_font('Arial', '', 9)
        for _, row in portfolio.iterrows():
            pdf.cell(40, 6, str(row['Project_ID']), 1); pdf.cell(50, 6, str(row['Department']), 1); pdf.cell(50, 6, f"{row['Investment_Capital']:,.0f}", 1); pdf.cell(30, 6, f"{row['Pred_ROI']:.1f}", 1); pdf.ln()
        pdf.ln()
        pdf.chapter_title('3. Deferred / Rejected Proposals')
        pdf.set_font('Arial', '', 9)
        for _, row in rejected.iterrows():
            pdf.cell(0, 5, f"[X] {row['Project_ID']} ({row['Department']}) - Low Efficiency Score: {row['Efficiency']:.2f}", 0, 1)
        return pdf.output(dest='S').encode('latin-1')

except ImportError:
    PDF_AVAILABLE = False
    def create_pdf_report(portfolio, rejected, budget, wacc): return None

def generate_text_report(portfolio, budget, wacc):
    txt = f"CAPITALIQ-AI REPORT\n-------------------\nBudget: INR {budget:,.2f}\nDeployed: INR {portfolio['Investment_Capital'].sum():,.2f}\nNPV: INR {portfolio['Dynamic_NPV'].sum():,.2f}\nWACC: {wacc*100:.1f}%\n\nAPPROVED PROJECTS:\n"
    for _, row in portfolio.iterrows(): txt += f"[x] {row['Project_ID']} | ROI: {row['Pred_ROI']:.1f}%\n"
    return txt

def dark_chart(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"), margin=dict(l=20, r=20, t=40, b=20))
    return fig

def render_analysis(text):
    st.markdown(f"<div class='analysis-box'><span class='analysis-title'>💡 EXECUTIVE INSIGHT</span><p class='analysis-text'>{text}</p></div>", unsafe_allow_html=True)

# ----------------------------------------------------
# 4. Callback Functions
# ----------------------------------------------------

def process_data_callback():
    if st.session_state.u_hist is not None and st.session_state.u_prop is not None:
        try:
            df_hist = standardize_columns(pd.read_csv(st.session_state.u_hist))
            df_prop = standardize_columns(pd.read_csv(st.session_state.u_prop))
            rf_roi, feature_imp = train_models(df_hist)
            features = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]
            for f in features: 
                if f not in df_prop.columns: df_prop[f] = 0
            df_prop["Pred_ROI"] = rf_roi.predict(df_prop[features])
            st.session_state['df_prop'] = df_prop
            st.session_state['feature_imp'] = feature_imp
            st.session_state.page_selection = "Capital Allocation Overview"
        except Exception as e: st.error(f"Error processing files: {e}")

def load_demo_callback():
    df_hist, df_prop = get_templates()
    rf_roi, feature_imp = train_models(df_hist)
    features = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]
    df_prop["Pred_ROI"] = rf_roi.predict(df_prop[features])
    st.session_state['df_prop'] = df_prop
    st.session_state['feature_imp'] = feature_imp
    st.session_state.u_hist = "Demo"
    st.session_state.u_prop = "Demo"
    st.session_state.page_selection = "Capital Allocation Overview"

def reset_data_callback():
    st.session_state.clear()
    st.session_state.page_selection = "Data Intake & Model Setup"

# ----------------------------------------------------
# 5. Sidebar & Layout
# ----------------------------------------------------

# Authentication Check
if not check_login():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("""<div class="brand-text">CAPITALIQ-AI</div><div class="brand-sub">Secure Login</div>""", unsafe_allow_html=True)
        pwd = st.text_input("Enter Access Token (Use 'admin'):", type="password")
        if st.button("Authenticate System"):
            if pwd == "admin":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Access Denied.")
    st.stop() 

with st.sidebar:
    st.markdown("""
        <div style="padding-top: 20px;">
            <div class="brand-text">CAPITALIQ-AI</div>
            <div class="brand-sub">Enterprise Edition</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if "page_selection" not in st.session_state:
        st.session_state.page_selection = "Data Intake & Model Setup"

    pages = ["Data Intake & Model Setup", "Capital Allocation Overview", "Model Intelligence Lab", "Risk–Return Simulation Arena", "Final Capital Allocation Sheet", "Strategic Portfolio Topology", "Scenario Stress Testing Hub", "AI Investment Committee Notes"]

    selected_page = st.radio("NAVIGATION", pages, key="page_selection", label_visibility="collapsed")
    
    st.markdown("---")
    st.subheader("Constraints & Sandbox")

    # Professional CAPM Calculator
    with st.expander("WACC Builder (CAPM)"):
        rf = st.number_input("Risk Free Rate (%)", 3.0, 10.0, 4.0) / 100
        beta = st.number_input("Beta", 0.5, 3.0, 1.2)
        mkt_prem = st.number_input("Market Premium (%)", 3.0, 15.0, 6.0) / 100
        cost_equity = rf + beta * mkt_prem
        
        cost_debt = st.number_input("Cost of Debt (%)", 2.0, 15.0, 5.0) / 100
        tax_rate = st.number_input("Tax Rate (%)", 0.0, 40.0, 25.0) / 100
        equity_weight = st.slider("Equity %", 0, 100, 70) / 100
        
        wacc_calc = (equity_weight * cost_equity) + ((1-equity_weight) * cost_debt * (1-tax_rate))
        st.caption(f"Calculated WACC: {wacc_calc*100:.2f}%")
        wacc_input = wacc_calc

    budget_input = st.number_input("Budget (INR)", value=15000000.0, step=500000.0)
    min_dept_spend = st.slider("Min Dept. Allocation (%)", 0, 30, 0) / 100
    
    max_risk = st.slider("Max Portfolio Risk", 1.0, 10.0, 6.5)
    market_shock = st.slider("Market Scenario", -0.20, 0.20, 0.0, 0.01, format="%+.0f%%")
    
    st.markdown("---")
    gemini_key = st.text_input("Gemini API Key (Opt.)", type="password")

    st.button("Reset / Clear All Data", use_container_width=True, on_click=reset_data_callback)

    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; font-size: 10px; color: #64748b;">
            © 2026 ED Technologies.<br>All Rights Reserved.<br>Confidential.
        </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# 6. Main Content
# ----------------------------------------------------

# --- PAGE: DATA INTAKE & MODEL SETUP ---
if selected_page == "Data Intake & Model Setup":
    st.title("Welcome to CapitalIQ-AI")
    if 'df_prop' in st.session_state:
        st.success("Data System Online: Predictive Models Trained & Ready.")
        st.info("Your dataset is currently loaded in memory. Navigate to the Capital Allocation Overview.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Dashboard", type="primary"):
                st.session_state.page_selection = "Capital Allocation Overview"
                st.rerun()
        with col2:
            if st.button("Unload Current Data"):
                reset_data_callback()
                st.rerun()
    else:
        col_intro, col_setup = st.columns([1.5, 1])
        with col_intro:
            st.markdown("### The Enterprise Standard for AI-Driven Capital Allocation")
            st.markdown("""
            System Workflow:
            1. Upload Historical Data: The AI learns from your past project outcomes (ROI, Risk, Success).
            2. Upload New Proposals: The system predicts ROI for new projects and runs linear programming optimization.
            3. Optimization: The Solver maximizes NPV while adhering to your Budget, Risk, and Departmental constraints.
            """)
            h_temp, p_temp = get_templates()
            c1, c2 = st.columns(2)
            c1.download_button("Download Train Template", h_temp.to_csv(index=False), "train_template.csv")
            c2.download_button("Download Predict Template", p_temp.to_csv(index=False), "predict_template.csv")
        with col_setup:
            st.markdown("#### Initialize System")
            with st.container():
                st.file_uploader("1. Training Data (History)", type=["csv"], key="u_hist", on_change=process_data_callback)
                st.file_uploader("2. Proposal Data (New)", type=["csv"], key="u_prop", on_change=process_data_callback)
                st.markdown("---")
                st.button("Load Demo Data", type="primary", on_click=load_demo_callback)

# --- SHARED DATA LOGIC ---
if selected_page != "Data Intake & Model Setup" and 'df_prop' not in st.session_state:
    st.warning("Please load data first.")
    st.stop()

if 'df_prop' in st.session_state:
    df_prop = st.session_state['df_prop'].copy()
    feature_imp = st.session_state['feature_imp']
    df_prop["Pred_ROI"] = df_prop["Pred_ROI"] * (1 + market_shock)
    df_prop["Dynamic_NPV"] = df_prop.apply(lambda row: calculate_dynamic_npv(row, wacc_input), axis=1)
    df_prop["Payback_Years"] = df_prop.apply(calculate_payback, axis=1)
    df_prop["Efficiency"] = df_prop["Pred_ROI"] / df_prop["Risk_Score"]
    df_prop = run_advanced_optimization(df_prop, budget_input, min_dept_spend)
    portfolio = df_prop[df_prop["Selected"] == 1]
    rejected = df_prop[df_prop["Selected"] == 0]

# --- PAGE: CAPITAL ALLOCATION OVERVIEW ---
if selected_page == "Capital Allocation Overview":
    st.title("Executive Dashboard")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Projects Funded", f"{len(portfolio)}", f"Total: {len(df_prop)}")
    kpi2.metric("Capital Deployed", f"₹{portfolio['Investment_Capital'].sum()/1e6:.2f}M", f"Util: {portfolio['Investment_Capital'].sum()/budget_input*100:.1f}%")
    kpi3.metric("Projected NPV", f"₹{portfolio['Dynamic_NPV'].sum()/1e6:.2f}M", delta=f"Payback: {portfolio['Payback_Years'].mean():.1f} Yrs")
    kpi4.metric("Avg Risk Score", f"{portfolio['Risk_Score'].mean():.2f}", f"Max: {max_risk}")
    
    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Capital Allocation by Department")
        if not portfolio.empty:
            fig = px.bar(portfolio, x="Department", y="Investment_Capital", color="Pred_ROI", 
                         title="Budget Distribution (Colored by ROI)", 
                         text_auto='.2s', 
                         color_continuous_scale="viridis")
            fig.update_layout(xaxis_title="Department", yaxis_title="Allocated Budget (INR)")
            fig.update_traces(textfont_size=14, textangle=0, textposition="outside", cliponaxis=False)
            st.plotly_chart(dark_chart(fig), use_container_width=True)
            render_analysis("""This distribution visualizes the capital allocation efficiency across departments. The correlation between bar height (allocation) and color intensity (ROI) reveals the alignment of resources. A divergence here—such as high allocation to low-intensity bars—would signal a need for immediate budget reallocation to higher-yielding functional units to improve the overall Weighted Average Cost of Capital (WACC) spread.""")
        else: st.info("No projects selected. Try increasing the budget.")
    with c2:
        st.subheader("Actionable Reports")
        if PDF_AVAILABLE:
            pdf_data = create_pdf_report(portfolio, rejected, budget_input, wacc_input)
            st.download_button("📄 Download Official PDF Report", data=pdf_data, file_name="CapitalIQ_Report.pdf", mime="application/pdf", use_container_width=True)
        else:
            st.warning("⚠️ PDF generation disabled (missing 'fpdf').")
            txt_report = generate_text_report(portfolio, budget_input, wacc_input)
            st.download_button("Download Text Report", txt_report, "Report.txt", use_container_width=True)
        
        st.markdown("##### 📈 Top ROI Drivers")
        st.dataframe(
            feature_imp.head(5).style.background_gradient(cmap='Greens', subset=['Importance']), 
            use_container_width=True, 
            hide_index=True
        )
        render_analysis("""The model has identified these specific variables as the primary determinants of project success within your dataset. The intensity of the green gradient corresponds to statistical weight. Management should prioritize improving these specific metrics in pre-project planning, as they demonstrate the highest marginal impact on the predicted Return on Investment.""")

# --- PAGE: MODEL INTELLIGENCE LAB ---
elif selected_page == "Model Intelligence Lab":
    st.title("AI & Model Analytics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 1. Financial Sensitivity Bridge")
        if not portfolio.empty:
            fig_torn = generate_professional_waterfall(portfolio)
            st.plotly_chart(fig_torn, use_container_width=True)
            render_analysis("""The bridge highlights the variance between the base theoretical value and the risk-adjusted reality. The red segments quantify the specific impact of WACC and inflation, demonstrating that external economic factors could erode a significant portion of the projected value if not hedged. The blue terminal bar represents the final, conservative Net Present Value after accounting for these inevitable market frictions.""")
        else: st.warning("No portfolio to analyze.")
    with col2:
        st.markdown("##### 2. Strategic Fit vs Benchmark")
        fig_radar = generate_benchmark_radar(df_prop)
        st.plotly_chart(fig_radar, use_container_width=True)
        render_analysis("""This radar analysis contrasts your selected portfolio (Green) against a theoretical ideal benchmark (Dotted White). The goal is to expand the green area to the outer edges. A retraction in the 'Strategic Fit' or 'ROI Potential' axes suggests that while the selected projects may be profitable, they are not fully aligned with the organization's broader long-term strategic directives.""")
    st.markdown("---")
    st.markdown("##### 3. Predictive Signal Strength")
    fig_imp = px.bar(feature_imp, x="Importance", y="Feature", orientation='h', color="Importance", color_continuous_scale="Teal")
    st.plotly_chart(dark_chart(fig_imp), use_container_width=True)
    render_analysis("""This chart ranks the decision-making factors used by the Random Forest algorithm. A longer bar indicates that the model relies more heavily on that specific feature to distinguish between high and low performers. If 'Risk Score' dominates this chart, it implies the portfolio is being optimized primarily for safety; if 'Market Trend' dominates, the portfolio is aggressive and growth-oriented.""")

# --- PAGE: RISK–RETURN SIMULATION ARENA ---
elif selected_page == "Risk–Return Simulation Arena":
    st.title("Efficient Frontier Simulation")
    sim_runs = st.slider("Monte Carlo Iterations", min_value=100, max_value=5000, value=1000, step=100)
    if st.button(f"Run {sim_runs} Simulation Scenarios"):
        st.markdown(f"Simulating {sim_runs} portfolio combinations...")
        progress_bar = st.progress(0)
        results = []
        total_cap = df_prop["Investment_Capital"].sum()
        avg_p = min(0.5, budget_input / (total_cap + 1))
        for i in range(sim_runs):
            mask = np.random.rand(len(df_prop)) < avg_p
            sample = df_prop[mask]
            if not sample.empty and sample["Investment_Capital"].sum() <= budget_input:
                results.append({"Risk": sample["Risk_Score"].mean(), "Return": sample["Pred_ROI"].mean(), "NPV": sample["Dynamic_NPV"].sum()})
            if i % 50 == 0: progress_bar.progress(min(i/sim_runs, 1.0))
        progress_bar.empty()
        sim_df = pd.DataFrame(results)
        if not sim_df.empty:
            fig_ef = px.scatter(sim_df, x="Risk", y="Return", color="NPV", title="Risk vs Return Landscape", color_continuous_scale="turbo", labels={"Risk": "Average Risk Score", "Return": "Average ROI (%)"})
            if not portfolio.empty:
                fig_ef.add_trace(go.Scatter(x=[portfolio["Risk_Score"].mean()], y=[portfolio["Pred_ROI"].mean()], mode='markers+text', marker=dict(color='white', size=20, symbol='star', line=dict(width=2, color='black')), name="Your Portfolio", text=["YOU ARE HERE"], textposition="top center"))
            st.plotly_chart(dark_chart(fig_ef), use_container_width=True)
            render_analysis("""This scatter plot represents the universe of all possible portfolio combinations. The 'Efficient Frontier' is the upper-left boundary of the cloud, representing maximum return for minimum risk. The position of the white star indicates the efficiency of your current selection. If the star sits deep inside the cloud rather than on the edge, the portfolio is sub-optimal and leaving potential value on the table.""")
        else: st.error("Simulation failed to find valid portfolios within constraints.")

# --- PAGE: FINAL CAPITAL ALLOCATION SHEET ---
elif selected_page == "Final Capital Allocation Sheet":
    st.title("Final Investment Schedule")
    tab1, tab2 = st.tabs(["Selected Projects", "Rejected Projects"])
    with tab1:
        st.dataframe(portfolio[["Project_ID", "Department", "Investment_Capital", "Pred_ROI", "Payback_Years", "Efficiency"]].style.format({"Investment_Capital": "₹{:,.0f}", "Pred_ROI": "{:.1f}%", "Payback_Years": "{:.1f} yrs", "Efficiency": "{:.2f}"}).background_gradient(subset=["Efficiency"], cmap="Greens"), use_container_width=True)
        csv = portfolio.to_csv(index=False).encode('utf-8')
        st.download_button("Export Raw Data (CSV)", csv, "Strategic_Portfolio.csv", "text/csv")
        render_analysis("""The projects listed above have successfully cleared the optimization hurdles, offering the highest 'Efficiency' ratio (ROI per unit of Risk). The darker green rows represent the portfolio anchors—investments that provide outsized returns relative to their capital cost. These projects should be prioritized for immediate funding and execution.""")
    with tab2:
        rejected = df_prop[df_prop["Selected"] == 0]
        st.dataframe(rejected[["Project_ID", "Investment_Capital", "Pred_ROI"]], use_container_width=True)
        render_analysis("""These proposals were excluded by the solver algorithm. The rejection is driven by one of three factors: 1) The ROI did not meet the WACC hurdle rate, 2) The Risk Score exceeded the corporate tolerance threshold, or 3) The capital required would have displaced a more efficient project given the hard budget constraint.""")

# --- PAGE: STRATEGIC PORTFOLIO TOPOLOGY ---
elif selected_page == "Strategic Portfolio Topology":
    st.title("Portfolio Topology")
    fig_3d = generate_3d_strategic_triangle(df_prop)
    st.plotly_chart(fig_3d, use_container_width=True)
    render_analysis("""This volumetric analysis plots projects along three critical axes: Strategy, ROI, and Safety. A healthy portfolio should exhibit a cluster of funded projects (bright dots) in the upper-right-back quadrant, indicating high scores across all three dimensions. Outliers—dots that are high in ROI but low in Safety—represent 'Gambles' that may require special risk mitigation plans if approved.""")

# --- PAGE: SCENARIO STRESS TESTING HUB ---
elif selected_page == "Scenario Stress Testing Hub":
    st.title("Scenario Simulation")
    col_save, col_view = st.columns([1, 3])
    with col_save:
        scenario_name = st.text_input("Scenario Name", value="Base Case")
        if st.button("Save Current State"):
            save_scenario_db(scenario_name, budget_input, portfolio['Dynamic_NPV'].sum(), 
                             portfolio['Pred_ROI'].mean(), len(portfolio), wacc_input)
            st.success(f"Saved {scenario_name} to Database!")
            
    with col_view:
        df_scenarios = get_scenarios_db()
        
        if not df_scenarios.empty:
            # 1. Chart - Using the fixed function
            fig_comp = generate_combo_scenario_chart(df_scenarios.to_dict('records'))
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # 2. Table
            st.markdown("##### Scenario Data Registry (SQL)")
            df_display = df_scenarios.copy()
            df_display.columns = [c.lower() for c in df_display.columns]
            
            # Safely calculate millions
            if 'budget' in df_display.columns: df_display['budget'] = df_display['budget'] / 1e6
            if 'npv' in df_display.columns: df_display['npv'] = df_display['npv'] / 1e6

            st.dataframe(
                df_display.style
                .format({
                    'budget': '₹{:.2f}M',
                    'npv': '₹{:.2f}M',
                    'wacc': '{:.1%}',
                    'roi': '{:.2f}%'
                })
                .background_gradient(subset=['npv'], cmap='Blues'),
                use_container_width=True,
                hide_index=True
            )
            render_analysis("""The comparative bar chart above allows for stress-testing different strategic assumptions. A divergence between the Blue Bar (Total NPV Wealth) and the Yellow Line (Efficiency) indicates a trade-off. The table below details the exact input variables (WACC, Budget) used to generate these outcomes, providing a clear audit trail of how financial constraints impact the final project count and aggregate returns.""")
        else: st.info("Adjust WACC/Budget in the sidebar, then click 'Save Current State' to compare scenarios.")

# --- PAGE: AI INVESTMENT COMMITTEE NOTES ---
elif selected_page == "AI Investment Committee Notes":
    st.title("AI Investment Memos")
    if not gemini_key:
        st.warning("⚠️ No API Key detected. Using rule-based generation. Enter Key in Sidebar for LLM Analysis.")
    
    col_app, col_rej = st.columns(2)
    with col_app:
        st.subheader("Top Approvals")
        for i, row in portfolio.sort_values(by="Dynamic_NPV", ascending=False).head(3).iterrows():
            with st.expander(f"APPROVED: {row['Project_ID']} ({row['Department']})", expanded=True):
                # Call Real AI function
                if st.button(f"Generate AI Analysis for {row['Project_ID']}", key=f"btn_app_{i}"):
                    with st.spinner("Consulting AI..."):
                        analysis = generate_ai_memo_text(row, gemini_key)
                        st.markdown(f"**AI Insight:**\n{analysis}")
                
                c1, c2 = st.columns(2)
                c1.metric("NPV Contribution", f"₹{row['Dynamic_NPV']/1e5:.1f} L")
                c2.metric("Strategic Fit", f"{row['Strategic_Alignment']}/10")

    with col_rej:
        st.subheader("Top Rejections")
        rejected = df_prop[df_prop["Selected"] == 0]
        for i, row in rejected.sort_values(by="Pred_ROI", ascending=False).head(3).iterrows():
            with st.expander(f"REJECTED: {row['Project_ID']} ({row['Department']})"):
                st.error(f"Decision: DEFERRED\n\nReason: Failed to beat capital cost hurdle or budget constraint.")
                c1, c2 = st.columns(2)
                c1.metric("Risk Score", f"{row['Risk_Score']}")
                c2.metric("Efficiency", f"{row['Efficiency']:.2f}")
    
    render_analysis("""These automated memos serve as a transparent audit trail for stakeholder communication. By explicitly stating the financial metrics (NPV, Risk Score) and the strategic logic behind every approval and rejection, these documents bridge the gap between complex algorithmic decision-making and executive accountability.""")
