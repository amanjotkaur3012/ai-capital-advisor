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
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ADD THIS: Permissive settings to prevent blocking financial data
safety_config = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
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
    div[data-testid="stMetricValue"] { font-size: 34px !important; color: #ffffff !important; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 15px !important; color: #58a6ff !important; text-transform: uppercase; font-weight: 600; }
    .ai-insight-box {
        background: rgba(88, 166, 255, 0.15); border: 1px solid #58a6ff;
        padding: 25px; border-radius: 10px; color: #f0f6fc; margin: 20px 0;
        font-size: 17px; line-height: 1.6;
    }
    section[data-testid="stSidebar"] { 
        background-color: #010409 !important; 
        border-right: 2px solid #30363d; 
    }

    /* Button text color – visible on all backgrounds */
    .stButton > button {
        color: #58a6ff !important;
        font-weight: 600;
        background-color: #161b22; /* Adds depth to the dark theme */
        border: 1px solid #30363d;
    }

    .stButton > button:hover {
        color: #58a6ff !important;
        border-color: #58a6ff;
        background-color: #0d1117;
    }
    section[data-testid="stSidebar"] { 
        background-color: #010409 !important; 
        border-right: 2px solid #30363d; 
    }

    /* Normal buttons */
    .stButton > button {
        color: #58a6ff !important;
        font-weight: 600;
        background-color: #21262d;
        border: 1px solid #30363d;
    }

    /* Download buttons */
    .stDownloadButton > button {
        color: #58a6ff !important;
        font-weight: 600;
        background-color: #21262d;
        border: 1px solid #30363d;
    }

    /* Hover effects for both */
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        color: #58a6ff !important;
        background-color: #30363d;
        border-color: #8b949e;
    }
    
    /* SIDEBAR TEXT — FORCE WHITE & HIGH CONTRAST */
    section[data-testid="stSidebar"] {
        color: #ffffff !important;
    }
    
    /* SIDEBAR LABELS (White for high visibility) */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
    }

    /* INPUT BOX TEXT (White background, Dark text for legibility) */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {
        color: #0d1117 !important;
        background-color: #ffffff !important;
        border-radius: 4px;
    }

    /* Number input / text input specific targeting */
    section[data-testid="stSidebar"] div[data-testid="stTextInput"] input,
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input {
        color: #0d1117 !important;
        background-color: #ffffff !important;
    }

    /* Selectbox text and internal styling */
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        color: #0d1117 !important;
        background-color: #ffffff !important;
    }

    /* File uploader box - ensuring the drop zone text is visible */
    section[data-testid="stSidebar"] .stFileUploader section {
        background-color: #161b22 !important;
        color: #ffffff !important;
        border: 1px dashed #30363d;
    }

    /* Checkbox & radio labels */
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
    }
    
    /* Radio dot color branding */
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }
    
    /* FILE UPLOADER – DARK THEME FIX */
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px;
        padding: 12px;
    }

    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] label {
        color: #ffffff !important;
        margin-bottom: 10px;
    }

    /* Browse files button inside uploader */
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button {
        background-color: #21262d !important;
        color: #58a6ff !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        font-weight: 600;
        width: 100%; /* Makes button more prominent in sidebar */
    }

    /* Hover state for uploader button */
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button:hover {
        background-color: #30363d !important;
        border-color: #58a6ff !important;
        color: #58a6ff !important;
        transition: 0.3s;
    }

    /* Small text inside uploader (e.g., "Limit 200MB") */
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] small {
        color: #8b949e !important;
    }
    
    /* DEEP TARGETING FOR METRIC LABELS (White Visibility Fix) */
    div[data-testid="stMetricLabel"] > div > p {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        opacity: 1 !important;
    }

    /* Target the fallback if the above doesn't catch it */
    [data-testid="stMetricLabel"] p {
        color: #ffffff !important;
    }

    /* Ensure the large Metric Value (the numbers) stays bright white */
    div[data-testid="stMetricValue"] > div {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    /* Card Styling */
    div[data-testid="stMetric"] {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }
    

    
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

#def check_api_status():
   # """Verifies connection to Gemini API and returns status color."""
   # try:
        # Using the lite model to save quota for the presentation
       # model = genai.GenerativeModel(model_name="models/gemini-2.0-flash-lite")
       # model.generate_content("ping", generation_config={"max_output_tokens": 1})
       # return "#238636", "CONNECTED"  # Institutional Green
   # except Exception:
      #  return "#da3633", "OFFLINE"    # Institutional Red


from fpdf import FPDF

def generate_pdf(selected_df, total_cap, total_val, avg_esg):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(88, 166, 255)
    pdf.cell(0, 20, "STRATOS QUANT | Executive Report", ln=True, align='C')
    
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Portfolio Highlights", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Capital Deployed: ${total_cap:,.0f}", ln=True)
    pdf.cell(0, 8, f"Strategic Value: ${total_val:,.0f}", ln=True)
    pdf.cell(0, 8, f"Avg ESG Score: {avg_esg:.2f}", ln=True)
    pdf.ln(10)
    
    # Simple Table
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(40, 10, "Project ID", 1)
    pdf.cell(60, 10, "Department", 1)
    pdf.cell(40, 10, "Capital", 1)
    pdf.cell(40, 10, "ROI %", 1)
    pdf.ln()
    
    pdf.set_font("Arial", '', 10)
    for i, r in selected_df.iterrows():
        pdf.cell(40, 10, str(r['Project_ID']), 1)
        pdf.cell(60, 10, str(r['Department']), 1)
        pdf.cell(40, 10, f"${r['Investment_Capital']:,.0f}", 1)
        pdf.cell(40, 10, f"{r['Pred_ROI']:.1f}%", 1)
        pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

# --- ADD THIS HERE (Exactly at the left margin, no initial spaces) ---
def check_api_status():
    """Verifies connection to Gemini API and returns status color."""
    try:
        model = genai.GenerativeModel("models/gemini-3-flash-preview")
        # Lightweight ping to check connectivity
        model.generate_content("ping", generation_config={"max_output_tokens": 1})
        return "#238636", "CONNECTED"  # Institutional Green
    except Exception:
        return "#da3633", "OFFLINE"    # Institutional Red
    
    

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
        
        st.markdown("---")
        st.header("EXECUTIVE REPORTING")
        if st.button("PREPARE PDF SUMMARY", key="sidebar_pdf_trigger"):
            t_cap = selected['Investment_Capital'].sum()
            t_val = selected['Strategic_Value'].sum()
            a_esg = selected['ESG_Score'].mean()
            pdf_data = generate_pdf(selected, t_cap, t_val, a_esg)
            st.download_button(
                label="📥 DOWNLOAD EXECUTIVE PDF",
                data=pdf_data,
                file_name="STRATOS_Report.pdf",
                mime="application/pdf",
                key="sidebar_pdf_download"
            )

        # --- MODEL STATUS INDICATOR (ALWAYS VISIBLE) ---
        #st.markdown("---")
        #try:
        #    color, status_text = check_api_status()
        #    st.markdown(f"""
        #        <div style="display: flex; align-items: center; gap: 10px; padding: 10px; 
        #                    background: #161b22; border-radius: 8px; border: 1px solid #30363d;">
        #            <div style="width: 12px; height: 12px; background-color: {color}; 
        #                        border-radius: 50%; box-shadow: 0 0 8px {color};"></div>
        #            <span style="color: #ffffff; font-size: 14px; font-weight: 600; 
        #                         letter-spacing: 0.5px;">API STATUS: {status_text}</span>
        #        </div>
        #    """, unsafe_allow_html=True)
        #except:
        #    st.error("Status check failed.")
            

    # Dataset Setup
    # 1. Dataset Setup
    if up_file:
        df = pd.read_csv(up_file)
        # Fix missing ESG_Score if individual E, S, G columns exist
        if 'ESG_Score' not in df.columns and all(k in df.columns for k in ['E_Score', 'S_Score', 'G_Score']):
            df['ESG_Score'] = (df['E_Score'] + df['S_Score'] + df['G_Score']) / 3
        
        # KEY FIX: If Actual_ROI is missing (common for new uploads), 
        # we provide a dummy value so the model doesn't crash
        if 'Actual_ROI' not in df.columns:
            df['Actual_ROI'] = (df['Strategic_Alignment'] * 1.5) # Fallback logic
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

    # 2. Machine Learning Calculation Layers
    feats = ['Investment_Capital', 'Risk_Score', 'ESG_Score', 'Volatility', 'Strategic_Alignment']
    
    # Verify all features exist before fitting
    missing_cols = [c for c in feats if c not in df.columns]
    if missing_cols:
        st.error(f"Missing mandatory columns in CSV: {missing_cols}")
        st.stop()

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(df[feats], df['Actual_ROI'])
    df['Pred_ROI'] = rf_model.predict(df[feats])
    
    # 3. Financial Engineering Layers
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

        # ============================================================
        # 1. PORTFOLIO AGGREGATE PERFORMANCE
        # ============================================================

        st.markdown('<div class="section-header">PORTFOLIO AGGREGATE PERFORMANCE</div>', unsafe_allow_html=True)

        total_capital = selected['Investment_Capital'].sum()
        total_value = selected['Strategic_Value'].sum()
        avg_esg = selected['ESG_Score'].mean()
        avg_pi = selected['PI'].mean()

        def fmt_money(x):
            if x >= 1e9: return f"${x/1e9:.2f}B"
            if x >= 1e6: return f"${x/1e6:.2f}M"
            if x >= 1e3: return f"${x/1e3:.1f}K"
            return f"${x:.0f}"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAPITAL DEPLOYED", fmt_money(total_capital))
        c2.metric("STRATEGIC VALUE", fmt_money(total_value))
        c3.metric("ESG IMPACT", f"{avg_esg:.1f}/10")
        c4.metric("VALUE INDEX (PI)", f"{avg_pi:.2f}x")

        # ============================================================
        # 2. PORTFOLIO SIGNALS
        # ============================================================

        efficiency = total_value / total_capital
        esg_spread = avg_esg - df['ESG_Score'].mean()

        st.markdown(
            f"""
            **Portfolio Signals**
            - Capital efficiency (Value / Capital): {efficiency:.2f}x
            - ESG premium vs universe: {esg_spread:+.2f}
            """,
            unsafe_allow_html=True
        )

        # ============================================================
        # 3. FUNDING SCHEDULE (DECISION-ORDERED)
        # ============================================================

        st.markdown('<div class="section-header">FUNDING SCHEDULE</div>', unsafe_allow_html=True)

        display_df = selected.sort_values(
            ["PI", "Pred_ROI"], ascending=False
        )

        st.dataframe(
            display_df[
                ['Project_ID', 'Department', 'Investment_Capital', 'Pred_ROI', 'PI', 'ESG_Score']
            ].style.background_gradient(cmap='Greens'),
            use_container_width=True
        )

        # ============================================================
        # 4. CAPITAL CONCENTRATION CHECK
        # ============================================================

        top3_share = display_df.head(3)['Investment_Capital'].sum() / total_capital

        if top3_share > 0.55:
            st.warning(
                f"Capital concentration risk detected: top 3 projects absorb {top3_share:.0%} of deployed capital."
            )

        # ============================================================
        # 5. AI INTERPRETATION
        # ============================================================

        if st.button("Interpret Summary", key="btn_sum"):
            with st.status("AI Analyzing Balance Sheet...", expanded=True) as status:
                try:
                    model = genai.GenerativeModel(model_name="models/gemini-3-flash-preview")
                    prompt = f"Act as a CFO. Portfolio summary: Budget ${total_capital:,.0f}, Strategic Value ${total_value:,.0f}, ESG {avg_esg:.1f}. Analyze the wealth creation quality."
                    
                    # Pass the safety settings here
                    response = model.generate_content(prompt, safety_settings=safety_config)
                    
                    # Check if text exists before calling .text
                    if response.candidates and len(response.candidates[0].content.parts) > 0:
                        st.markdown(f"<div class='ai-insight-box'>{response.text}</div>", unsafe_allow_html=True)
                    else:
                        st.warning("The model blocked this analysis due to safety filters. Try rephrasing your constraints.")
                    
                    status.update(label="Analysis Complete", state="complete")
                except Exception as e:
                    st.error(f"API Error: {str(e)}")

    elif nav == " ML INTELLIGENCE":

        st.markdown('<div class="section-header">PREDICTIVE ROI LOGIC</div>', unsafe_allow_html=True)

        # ============================================================
        # 1. FEATURE IMPORTANCE (MODEL INTERPRETABILITY)
        # ============================================================

        col_l, col_r = st.columns(2)

        with col_l:
            feat_imp = pd.DataFrame({
                'Feature': feats,
                'Importance': rf_model.feature_importances_
            }).sort_values("Importance", ascending=True)

            feat_imp['Contribution_%'] = (feat_imp['Importance'] / feat_imp['Importance'].sum()) * 100

            fig_imp = px.bar(
                feat_imp,
                x="Contribution_%",
                y="Feature",
                orientation="h",
                text=feat_imp["Contribution_%"].round(1).astype(str) + "%",
                title="Drivers of ROI Prediction"
            )

            fig_imp.update_layout(
                xaxis_title="Relative Contribution (%)",
                yaxis_title="",
                title_font_size=18
            )

            st.plotly_chart(fig_imp, use_container_width=True)

            top_driver = feat_imp.iloc[-1]['Feature']

        # ============================================================
        # 2. VALUE vs ESG MAP (MODEL OUTPUT)
        # ============================================================

        with col_r:
            median_esg = df["ESG_Score"].median()
            median_val = df["Strategic_Value"].median()

            fig_eff = px.scatter(
                df,
                x="ESG_Score",
                y="Strategic_Value",
                size="Investment_Capital",
                color="Selected",
                hover_data=["Project_ID", "Pred_ROI", "PI"],
                title="Value vs ESG Trade-off (Model Output)",
                color_discrete_map={1:'#58a6ff', 0:'#30363d'}
            )

            fig_eff.add_hline(y=median_val, line_dash="dot")
            fig_eff.add_vline(x=median_esg, line_dash="dot")

            fig_eff.update_layout(
                xaxis_title="ESG Score",
                yaxis_title="Strategic Value",
                title_font_size=18
            )

            st.plotly_chart(fig_eff, use_container_width=True)

        

        # ============================================================
        # 3. ESG PILLAR BALANCE (SELECTED vs UNIVERSE BENCHMARK)
        # ============================================================

        st.markdown('<div class="section-header">ESG PILLAR BALANCE (RELATIVE QUALITY)</div>', unsafe_allow_html=True)

        # 1. Selected portfolio averages (The projects optimized by Pulp)
        sel_means = selected[['E_Score', 'S_Score', 'G_Score']].mean()
        sel_df = pd.DataFrame({
            "Pillar": sel_means.index,
            "Score": sel_means.values,
            "Group": "Selected Portfolio"
        })

        # 2. Universe averages (All uploaded or demo projects)
        uni_means = df[['E_Score', 'S_Score', 'G_Score']].mean()
        uni_df = pd.DataFrame({
            "Pillar": uni_means.index,
            "Score": uni_means.values,
            "Group": "Universe Average"
        })

        # 3. Combine for plotting
        plot_df = pd.concat([sel_df, uni_df], ignore_index=True)

        fig_esg = px.bar(
            plot_df,
            x="Pillar",
            y="Score",
            color="Group",
            barmode="group",
            text=plot_df["Score"].round(1),
            title="ESG Pillar Comparison: Selected Portfolio vs Universe",
            color_discrete_map={
                "Selected Portfolio": "#58a6ff", # Your primary blue
                "Universe Average": "#30363d"    # Your secondary grey
            }
        )
        
        fig_esg.update_layout(
            yaxis_title="Average Score (0–10)",
            xaxis_title="ESG Pillar",
            title_font_size=18,
            legend_title="Benchmark",
            yaxis=dict(range=[0,11]), # Added breathing room at the top
            template="plotly_dark",   # Ensures alignment with your dark theme
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_esg, use_container_width=True)

        # ===== INTERPRETATION METRICS =====
        gap = sel_means - uni_means
        worst_pillar = gap.idxmin()
        best_pillar = gap.idxmax()

        st.markdown(
            f"""
            <div style="background: rgba(88, 166, 255, 0.05); padding: 20px; border-radius: 8px; border: 1px solid #30363d;">
            <p style='color:#58a6ff; font-weight:700; margin-bottom:10px;'>ESG QUALITY SIGNALS</p>
            <ul style='list-style-type: none; padding-left: 0;'>
               <li> Strongest relative pillar: <b>{best_pillar.replace('_',' ')}</b></li>
               <li> Weakest relative pillar: <b>{worst_pillar.replace('_',' ')}</b></li>
               <li> Net ESG premium vs universe: <b>{gap.mean():+.2f}</b></li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ============================================================
        # 4. MODEL SIGNALS (DATA-DRIVEN)
        # ============================================================

        # --- CALCULATE SIGNALS FOR LOGIC AND AI ---
        # Using sel_means from the bar chart logic above
        avg_esg_val = sel_means.mean()
        imbalance = sel_means.max() - sel_means.min()

        if imbalance > 3:
            model_signal = "ESG imbalance detected across pillars"
        else:
            model_signal = "ESG distribution is structurally balanced"

        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.05); padding: 20px; border-radius: 8px; border: 1px solid #238636; margin: 20px 0;">
            <p style='color:#10b981; font-weight:700; margin-bottom:10px;'>MODEL ANALYTICS</p>
            <ul style='list-style-type: none; padding-left: 0; color: #f0f6fc;'>
                <li> Primary ROI Driver: <b>{top_driver.replace('_',' ')}</b></li>
                <li> Average ESG Score: <b>{avg_esg_val:.2f}</b></li>
                <li> ESG Dispersion: <b>{imbalance:.2f}</b></li>
                <li> Model Status: <b>{model_signal}</b></li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ============================================================
        # 5. AI INTERPRETATION (ML INTELLIGENCE)
        # ============================================================
        if st.button("Interpret ML Intelligence", key="btn_ml"):
            with st.status("AI Decoding Predictive Logic...", expanded=True) as status:
                try:
                    model = genai.GenerativeModel(model_name="models/gemini-3-flash-preview")
                    # Prompt using the variables we just calculated
                    prompt = f"Explain to a Board: The main ROI driver is {top_driver}. The portfolio ESG score is {avg_esg_val:.2f} with a dispersion of {imbalance:.2f}. Is the model picking high-quality projects?"
                    
                    response = model.generate_content(prompt, safety_settings=safety_config)
                    
                    if response.candidates and len(response.candidates[0].content.parts) > 0:
                        st.markdown(f"<div class='ai-insight-box'>{response.text}</div>", unsafe_allow_html=True)
                    else:
                        st.warning("The ML analysis was blocked by safety filters.")
                    
                    status.update(label="ML Insight Ready", state="complete")
                except Exception as e:
                    st.error(f"API Error: {str(e)}")


    elif nav == " SENSITIVITY":
        # ============================================================
        # 1. VALUE SENSITIVITY: BUDGET vs ESG
        # ============================================================

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

        h_df = pd.DataFrame(
            h_map,
            index=[f"${b/1e6:.1f}M" for b in b_range],
            columns=[f"ESG {e:.1f}" for e in e_range]
        )

        fig_hm = px.imshow(
            h_df,
            text_auto=".1f",
            aspect="auto",
            title="Where Portfolio Value Scales or Breaks",
            color_continuous_scale="RdYlGn"
        )

        max_val = h_df.values.max()
        opt_idx = np.unravel_index(np.argmax(h_df.values), h_df.shape)
        opt_budget = h_df.index[opt_idx[0]]
        opt_esg = h_df.columns[opt_idx[1]]

        fig_hm.add_annotation(
            x=opt_esg,
            y=opt_budget,
            text="MAX VALUE ZONE",
            showarrow=True,
            arrowhead=2,
            font=dict(color="white")
        )

        fig_hm.update_layout(
            xaxis_title="ESG Constraint Tightness",
            yaxis_title="Capital Budget Level",
            title_font_size=18
        )

        st.plotly_chart(fig_hm, use_container_width=True)

        st.caption(
            f" **Decision Signal:** Optimal value occurs near **{opt_budget} / {opt_esg}**. "
            "Beyond this ESG level, value erosion accelerates."
        )

        # ============================================================
        # 2. VALUE CLIFF DETECTION
        # ============================================================

        value_drop = np.diff(h_df.values, axis=1).mean()

        if value_drop < -0.15 * max_val:
            st.warning(
                " **Value Cliff Detected:** Increasing ESG strictness causes rapid value loss. "
                "Board-level trade-off discussion recommended."
            )
        else:
            st.success(" ESG tightening shows controlled value impact.")

        # ============================================================
        # 3. 3-YEAR CASH OUTLAY SCHEDULE
        # ============================================================

        st.markdown('<div class="section-header">3-YEAR CASH OUTLAY SCHEDULE</div>', unsafe_allow_html=True)

        selected['Y1'] = selected['Investment_Capital'] * selected['Phase_1_Cap']
        selected['Y2'] = selected['Investment_Capital'] * selected['Phase_2_Cap']
        selected['Y3'] = selected['Investment_Capital'] - (selected['Y1'] + selected['Y2'])

        ph_sum = selected[['Y1', 'Y2', 'Y3']].sum().reset_index()
        ph_sum.columns = ['Year', 'Outlay']

        peak_year = ph_sum.loc[ph_sum["Outlay"].idxmax(), "Year"]
        peak_val = ph_sum["Outlay"].max()

        fig_cash = px.area(
            ph_sum,
            x='Year',
            y='Outlay',
            title="Capital Burn Profile (Liquidity Stress Test)",
            text=ph_sum["Outlay"].round(0)
        )

        fig_cash.add_hline(
            y=ph_sum["Outlay"].mean(),
            line_dash="dot",
            annotation_text="Average Burn"
        )

        fig_cash.add_annotation(
            x=peak_year,
            y=peak_val,
            text="Peak Liquidity Pressure",
            showarrow=True,
            arrowhead=2
        )

        fig_cash.update_layout(
            xaxis_title="Investment Year",
            yaxis_title="Capital Outlay ($)",
            title_font_size=18
        )

        st.plotly_chart(fig_cash, use_container_width=True)

        st.caption(
            f" **Liquidity Insight:** Year **{peak_year}** requires the highest cash commitment. "
            "Ensure financing lines or reserves are aligned to this peak."
        )

        # ============================================================
        # 4. LIQUIDITY STRESS TEST
        # ============================================================

        avg_outlay = ph_sum["Outlay"].mean()
        max_outlay = ph_sum["Outlay"].max()
        stress_ratio = max_outlay / avg_outlay

        if stress_ratio > 1.25:
            st.error(
                " **Liquidity Stress Alert:** Peak capital demand exceeds average burn by >25%. "
                "Staged deployment or external financing recommended."
            )
        elif stress_ratio > 1.1:
            st.warning(
                " **Moderate Liquidity Pressure:** Capital peaks are manageable but require planning."
            )
        else:
            st.success(
                " **Stable Capital Profile:** No abnormal liquidity stress detected."
            )

       # ============================================================
        # 5. SCENARIO NARRATIVES (ROBUST & EXECUTIVE READY)
        # ============================================================

        st.markdown('<div class="section-header">SCENARIO NARRATIVES</div>', unsafe_allow_html=True)

        # 1. Extract data from the Sensitivity Map (h_df)
        scenarios = {
            "Growth First": h_df.iloc[:, 0].mean(),
            "Balanced Strategy": h_df.iloc[:, 2].mean(),
            "ESG Strict": h_df.iloc[:, -1].mean()
        }

        sc_df = pd.DataFrame.from_dict(
            scenarios, orient="index", columns=["Portfolio Value"]
        ).reset_index()

        # 2. Feasibility Logic: Prevents empty bars/crashes in the UI
        sc_df["Feasible"] = sc_df["Portfolio Value"] > 0
        # Replace 0 with NaN so Plotly doesn't try to draw a ghost bar
        sc_df["Display Value"] = sc_df["Portfolio Value"].replace(0, np.nan)

        # 3. Build the Strategic Bar Chart
        fig_sc = px.bar(
            sc_df,
            x="index",
            y="Display Value",
            text=sc_df["Portfolio Value"].apply(lambda x: "INFEASIBLE" if x == 0 else f"${x:,.0f}"),
            title="Portfolio Value Outcomes Under Board Mandates",
            color="Feasible",
            color_discrete_map={
                True: "#58a6ff",  # Your Institutional Blue
                False: "#30363d"  # Your Institutional Grey
            }
        )

        fig_sc.update_layout(
            xaxis_title="Strategy Type",
            yaxis_title="Total Strategic Value ($)",
            title_font_size=18,
            showlegend=False,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            uniformtext_mode='hide', 
            uniformtext_minsize=12,
            yaxis=dict(showgrid=True, gridcolor="#30363d")
        )

        # Labels sitting professionally outside the bars
        fig_sc.update_traces(textposition='outside')

        st.plotly_chart(fig_sc, use_container_width=True)

        # 4. DATA-DRIVEN INTERPRETATION
        # Check specifically if the ESG scenario is possible with current constraints
        esg_possible = sc_df.loc[sc_df["index"] == "ESG Strict", "Feasible"].iloc[0]

        if not esg_possible:
            st.warning(
                " **ESG Strict scenario is infeasible** under current budget. The project pool cannot meet the required {esg_min} ESG average within this capital limit."
            )

        # Determine the best scenario based on math
        best_scenario = sc_df.sort_values("Portfolio Value", ascending=False).iloc[0]["index"]

        st.markdown(
            f"""
            <div style="background: rgba(88, 166, 255, 0.05); padding: 15px; border-radius: 8px; border: 1px solid #30363d; margin-top: 10px;">
                <span style="color:#58a6ff; font-weight:800;">BOARD INSIGHT:</span> 
                The <b style="color:#ffffff;">{best_scenario}</b> mandate maximizes portfolio value under current conditions.
            </div>
            """, 
            unsafe_allow_html=True
        )
        # ============================================================
        # 6. EXECUTIVE RECOMMENDATION
        # ============================================================

        #st.markdown('<div class="section-header">EXECUTIVE SIGNAL</div>', unsafe_allow_html=True)

        #if best_scenario == "Balanced Strategy" and stress_ratio < 1.2:
            #rec = "PROCEED"
            #color = "green"
        #elif best_scenario == "Growth First" and stress_ratio < 1.15:
            #rec = "PROCEED WITH CAUTION"
            #color = "orange"
        #else:
            #rec = "RESTRUCTURE"
            #color = "red"

        #st.markdown(
            #f"<h2 style='color:{color}; font-weight:800;'>Recommendation: {rec}</h2>",
            #unsafe_allow_html=True
        #)

        # ============================================================
        # 7. AI INTERPRETATION (REFINED FOR API STABILITY)
        # ============================================================

        if st.button("Interpret Sensitivity", key="btn_sensitivity"):
            with st.status("AI Analyzing Value Trade-offs...", expanded=True) as status:
                try:
                    model = genai.GenerativeModel(model_name="models/gemini-3-flash-preview")
                    prompt = f"""
                    Act as a Strategic Finance Advisor. Analyze this capital allocation sensitivity:
                    - Optimal Value Zone: Budget {opt_budget} at ESG Hurdle {opt_esg}.
                    - Strategy Scenario: {best_scenario} provides the highest mean value.
                    - Liquidity Pressure: Stress ratio is {stress_ratio:.2f}.
                    - Recommendation: {rec}.
                    Explain where the 'Value Cliff' poses a risk.
                    """
                    response = model.generate_content(prompt)
                    st.markdown(f"<div class='ai-insight-box'>{response.text}</div>", unsafe_allow_html=True)
                    status.update(label="Sensitivity Analysis Complete", state="complete")
                except Exception as e:
                    st.error(f"API Error: {str(e)}")

    elif nav == " RISK MANAGEMENT":

        # ============================================================
        # 1. STRATEGIC RISK–RETURN QUADRANT
        # ============================================================

        st.markdown('<div class="section-header">STRATEGIC RISK–RETURN QUADRANT</div>', unsafe_allow_html=True)

        m_r = df['Risk_Score'].median()
        m_p = df['PI'].median()

        df['Quadrant'] = np.where(
            (df['Risk_Score'] <= m_r) & (df['PI'] >= m_p), 'CORE',
            np.where(
                (df['Risk_Score'] > m_r) & (df['PI'] >= m_p), 'SPECULATIVE',
                np.where(
                    (df['Risk_Score'] <= m_r) & (df['PI'] < m_p), 'DEFENSIVE',
                    'EXIT'
                )
            )
        )

        fig_q = px.scatter(
            df,
            x="Risk_Score",
            y="PI",
            color="Quadrant",
            size="Investment_Capital",
            text="Project_ID",
            hover_data=["Pred_ROI", "Strategic_Value"],
            title="Project Classification by Risk and Return",
            color_discrete_map={
                "CORE": "#238636",
                "SPECULATIVE": "#f59e0b",
                "DEFENSIVE": "#3b82f6",
                "EXIT": "#dc2626"
            }
        )

        fig_q.add_hline(y=m_p, line_dash="dot")
        fig_q.add_vline(x=m_r, line_dash="dot")

        fig_q.update_layout(
            xaxis_title="Risk Score (Lower = Safer)",
            yaxis_title="Profitability Index (Higher = Better)",
            title_font_size=18
        )

        st.plotly_chart(fig_q, use_container_width=True)

        core_pct = (df['Quadrant'] == 'CORE').mean() * 100
        exit_pct = (df['Quadrant'] == 'EXIT').mean() * 100

        st.markdown(
            f"""
            **Portfolio Composition**
            - Strategic core projects: {core_pct:.1f}%
            - Exit or restructure candidates: {exit_pct:.1f}%
            """,
            unsafe_allow_html=True
        )

        # ============================================================
        # 2. DOWNSIDE RISK METRICS (VaR & SHARPE)
        # ============================================================

        st.markdown('<div class="section-header">DOWNSIDE RISK (VaR & SHARPE)</div>', unsafe_allow_html=True)

        mu_p = (selected['Pred_ROI'] / 100).mean()
        sig_p = (selected['Pred_ROI'] / 100).std()

        var95 = norm.ppf(0.05, mu_p, sig_p)
        sharpe_avg = selected['Sharpe_Score'].mean()

        k1, k2 = st.columns(2)
        k1.metric("Value at Risk (95%)", f"{abs(var95):.2%}")
        k2.metric("Average Sharpe Ratio", f"{sharpe_avg:.2f}")

        # Risk quality classification (data-driven)
        if sharpe_avg >= 1.0 and abs(var95) <= 0.20:
            risk_quality = "STRONG"
        elif sharpe_avg >= 0.7:
            risk_quality = "ACCEPTABLE"
        else:
            risk_quality = "WEAK"

        st.markdown(
            f"""
            **Risk Quality Assessment**
            - Portfolio risk-adjusted return quality: {risk_quality}
            - Expected downside loss under normal conditions: {abs(var95):.2%}
            """,
            unsafe_allow_html=True
        )

        # ============================================================
        # 3. EXECUTIVE RISK ACTION SIGNAL
        # ============================================================

        st.markdown('<div class="section-header">EXECUTIVE RISK ACTION SIGNAL</div>', unsafe_allow_html=True)

        if core_pct >= 60 and sharpe_avg >= 0.9:
            action = "MAINTAIN CURRENT ALLOCATION"
        elif exit_pct >= 30:
            action = "REBALANCE AND REDUCE EXPOSURE TO WEAK PROJECTS"
        else:
            action = "REVIEW HEDGING AND CAPITAL STAGING STRATEGY"

        st.markdown(
            f"""
            **Recommended Action**
            {action}
            """,
            unsafe_allow_html=True
        )

        # ============================================================
        # 4. AI INTERPRETATION (PRESERVED, IMPROVED PROMPT)
        # ============================================================

        if st.button("Interpret Risk Exposure", key="btn_risk"):
            with st.status("AI Stress-Testing Portfolio...", expanded=True) as status:
                try:
                    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-lite")
                    prompt = f"Analyze for a CFO: Core Projects {core_pct:.1f}%, Exit Candidates {exit_pct:.1f}%, VaR {abs(var95):.2%}, Sharpe {sharpe_avg:.2f}. Risk Quality is {risk_quality}. Recommendation: {action}."
                    response = model.generate_content(prompt)
                    st.markdown(f"<div class='ai-insight-box'>{response.text}</div>", unsafe_allow_html=True)
                    status.update(label="Risk Assessment Generated", state="complete")
                except Exception as e:
                    st.error(f"API Error: {str(e)}")

    
    elif nav == " INSTITUTIONAL THESIS":

        st.markdown('<div class="section-header">PROJECT INVESTMENT THESIS</div>', unsafe_allow_html=True)

        target = st.selectbox(
            "Select Project for Institutional Review",
            selected['Project_ID']
        )

        r = selected[selected['Project_ID'] == target].iloc[0]

        # ============================================================
        # 1. PROJECT SNAPSHOT (NUMBERS FIRST)
        # ============================================================

        st.markdown("### Project Snapshot")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capital Required", f"${r['Investment_Capital']:,.0f}")
        c2.metric("Predicted ROI", f"{r['Pred_ROI']:.2f}%")
        c3.metric("Profitability Index", f"{r['PI']:.2f}x")
        c4.metric("Risk Score", f"{r['Risk_Score']:.1f}")

        # ============================================================
        # 2. VALUE CREATION LOGIC
        # ============================================================

        st.markdown("### Value Creation Logic")

        value_quality = r['Pred_ROI'] / r['Risk_Score']

        st.markdown(
            f"""
            - Strategic value generated: ${r['Strategic_Value']:,.0f}
            - Risk-adjusted return efficiency: {value_quality:.2f}
            - ESG alignment score: {r['ESG_Score']:.2f}
            """,
            unsafe_allow_html=True
        )

        # ============================================================
        # 3. CAPITAL STAGING LOGIC
        # ============================================================

        st.markdown("### Capital Deployment Structure")

        st.markdown(
            f"""
            - Phase 1 capital: {r['Phase_1_Cap']*100:.0f}% of total
            - Phase 2 capital: {r['Phase_2_Cap']*100:.0f}% of total
            - Final tranche: {(1 - r['Phase_1_Cap'] - r['Phase_2_Cap'])*100:.0f}% of total
            """,
            unsafe_allow_html=True
        )

        # ============================================================
        # 4. DOWNSIDE RISKS (DATA-DRIVEN)
        # ============================================================

        st.markdown("### Key Risks")

        risk_flags = []
        if r['Risk_Score'] > df['Risk_Score'].median():
            risk_flags.append("Risk level above portfolio median")
        if r['ESG_Score'] < df['ESG_Score'].median():
            risk_flags.append("Below-median ESG alignment")
        if r['PI'] < 1.0:
            risk_flags.append("Value creation below invested capital")

        if risk_flags:
            for f in risk_flags:
                st.markdown(f"- {f}")
        else:
            st.markdown("- No material red flags relative to portfolio")

        # ============================================================
        # 5. DECISION SIGNAL (AUTOMATIC)
        # ============================================================

        st.markdown("### Investment Committee Signal")

        if r['PI'] >= 1.2 and r['Risk_Score'] <= df['Risk_Score'].median():
            decision = "APPROVE"
        elif r['PI'] >= 1.0:
            decision = "APPROVE WITH CONDITIONS"
        else:
            decision = "DEFER OR RESTRUCTURE"

        st.markdown(f"**Recommended Decision:** {decision}")

        # ============================================================
        # 6. AI INSTITUTIONAL THESIS (STRUCTURED PROMPT)
        # ============================================================

        if st.button("Generate Institutional Thesis", key="btn_thesis"):
            with st.status("AI Generating Investment Memorandum...", expanded=True) as status:
                try:
                    model = genai.GenerativeModel(model_name="models/gemini-3-flash-preview")
                    prompt = f"Write institutional investment thesis for project {target}: Capital: {r['Investment_Capital']:,.0f}, PI: {r['PI']:.2f}, Risk: {r['Risk_Score']:.1f}, ESG: {r['ESG_Score']:.2f}."
                    
                    # Use safety config
                    response = model.generate_content(prompt, safety_settings=safety_config)
                    
                    # Validate parts exist
                    if response.candidates and len(response.candidates[0].content.parts) > 0:
                        st.markdown(f"<div class='ai-insight-box'>{response.text}</div>", unsafe_allow_html=True)
                    else:
                        st.error("Thesis generation blocked by safety filters. Check project risk scores.")
                        
                    status.update(label="Thesis Generated", state="complete")
                except Exception as e:
                    st.error(f"Thesis API Error: {str(e)}")

if __name__ == "__main__":
    main()
