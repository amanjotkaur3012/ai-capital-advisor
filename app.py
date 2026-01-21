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
GEMINI_API_KEY = "AIzaSyDcT_v6HM6S_MqEcxO5pFixbakP3_43Iu4"
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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

    # Dataset Setup
    if up_file:
        df = pd.read_csv(up_file)
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

    # Calculation Layers
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

        if st.button("Interpret Summary"):
            with st.spinner("AI analyzing portfolio..."):
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    res = model.generate_content(
                        f"""
                        Provide an executive summary of portfolio performance:

                        - Capital deployed: {total_capital:,.0f}
                        - Strategic value created: {total_value:,.0f}
                        - Portfolio efficiency: {efficiency:.2f}x
                        - Average ESG score: {avg_esg:.2f}
                        - Capital concentration (top 3): {top3_share:.0%}

                        Focus on value creation quality, capital discipline, and governance considerations.
                        """
                    )
                    st.markdown(f"<div class='ai-insight-box'>{res.text}</div>", unsafe_allow_html=True)
                except:
                    st.warning("AI temporarily unavailable. Please retry.")

    elif nav == " ML INTELLIGENCE":
        # All code below this must be indented by one level (4 spaces)
        st.markdown('<div class="section-header">PREDICTIVE ROI LOGIC</div>', unsafe_allow_html=True)

        # ============================================================
        # 1. FEATURE IMPORTANCE — INTERPRETABLE
        # ============================================================
        col_l, col_r = st.columns(2)

        with col_l:
            st.write("**WHAT DRIVES ROI (MODEL INTERPRETATION)**")

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
                title="Relative Contribution to ROI Prediction",
            )

            fig_imp.add_vline(
                x=feat_imp["Contribution_%"].mean(),
                line_dash="dash",
                annotation_text="Average Influence",
                opacity=0.4
            )

            fig_imp.update_layout(
                xaxis_title="Contribution to ROI Prediction (%)",
                yaxis_title="",
                title_font_size=16
            )

            st.plotly_chart(fig_imp, use_container_width=True)

            top_driver = feat_imp.iloc[-1]['Feature']
            st.caption(
                f" **Model Insight:** ROI predictions are primarily driven by **{top_driver}**. "
                f"Capital efficiency is more sensitive to this factor than others."
            )

        # ============================================================
        # 2. EFFICIENCY FRONTIER — DECISION MAP
        # ============================================================
        with col_r:
            st.write("**VALUE vs ESG DECISION MAP**")

            median_esg = df["ESG_Score"].median()
            median_val = df["Strategic_Value"].median()

            fig_eff = px.scatter(
                df,
                x="ESG_Score",
                y="Strategic_Value",
                size="Investment_Capital",
                color="Selected",
                hover_data=["Project_ID", "Pred_ROI", "PI"],
                title="Value–Sustainability Trade-off",
                color_discrete_map={1: '#58a6ff', 0: '#30363d'}
            )

            fig_eff.add_hline(
                y=median_val, line_dash="dot",
                annotation_text="High Value Zone"
            )

            fig_eff.add_vline(
                x=median_esg, line_dash="dot",
                annotation_text="High ESG Zone"
            )

            fig_eff.update_layout(
                xaxis_title="ESG Quality (Higher = Better)",
                yaxis_title="Strategic Value ($)",
                title_font_size=16
            )

            st.plotly_chart(fig_eff, use_container_width=True)

            st.caption(
                " **Interpretation:** Top-right quadrant = projects creating strong value while meeting ESG goals."
            )

        # ============================================================
        # 3. ESG RADAR — BALANCE DIAGNOSTIC (Fixed & Completed)
        # ============================================================
        st.markdown('<div class="section-header">ESG BALANCE DIAGNOSTIC</div>', unsafe_allow_html=True)

        p_means = selected[['E_Score', 'S_Score', 'G_Score']].mean().reset_index()
        p_means.columns = ['Pillar', 'Score']

        fig_radar = px.line_polar(
            p_means, 
            r='Score', 
            theta='Pillar', 
            line_close=True,
            title="Portfolio ESG Pillar Distribution"
        )
        fig_radar.update_traces(fill='toself', line_color='#10b981')
        st.plotly_chart(fig_radar, use_container_width=True)


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
        # 5. SCENARIO NARRATIVES
        # ============================================================

        st.markdown('<div class="section-header">SCENARIO NARRATIVES</div>', unsafe_allow_html=True)

        scenarios = {
            "Growth First": h_df.iloc[:, 0].mean(),
            "Balanced Strategy": h_df.iloc[:, 2].mean(),
            "ESG Strict": h_df.iloc[:, -1].mean()
        }

        sc_df = pd.DataFrame.from_dict(
            scenarios, orient="index", columns=["Portfolio Value"]
        ).reset_index()

        fig_sc = px.bar(
            sc_df,
            x="index",
            y="Portfolio Value",
            text="Portfolio Value",
            title="Strategy Outcomes Under Board Mandates"
        )

        fig_sc.update_layout(
            xaxis_title="Strategy Type",
            yaxis_title="Total Strategic Value ($)",
            title_font_size=18
        )

        st.plotly_chart(fig_sc, use_container_width=True)

        best_scenario = sc_df.sort_values("Portfolio Value", ascending=False).iloc[0]["index"]

        st.caption(
            f" **Board Insight:** The **{best_scenario}** mandate maximizes portfolio value under current conditions."
        )

        # ============================================================
        # 6. EXECUTIVE RECOMMENDATION
        # ============================================================

        st.markdown('<div class="section-header">EXECUTIVE SIGNAL</div>', unsafe_allow_html=True)

        if best_scenario == "Balanced Strategy" and stress_ratio < 1.2:
            rec = "PROCEED"
            color = "green"
        elif best_scenario == "Growth First" and stress_ratio < 1.15:
            rec = "PROCEED WITH CAUTION"
            color = "orange"
        else:
            rec = "RESTRUCTURE"
            color = "red"

        st.markdown(
            f"<h2 style='color:{color}; font-weight:800;'>Recommendation: {rec}</h2>",
            unsafe_allow_html=True
        )

        # ============================================================
        # 7. AI INTERPRETATION
        # ============================================================

        if st.button("Interpret Sensitivity"):
            with st.spinner("Analyzing trade-offs..."):
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    res = model.generate_content(
                        f"""
                        Explain this sensitivity analysis to a CFO:
                        - Optimal budget/ESG zone: {opt_budget} / {opt_esg}
                        - Best scenario: {best_scenario}
                        - Liquidity stress ratio: {stress_ratio:.2f}
                        - Executive recommendation: {rec}

                        Focus on capital safety, flexibility, and strategic trade-offs.
                        """
                    )
                    st.markdown(f"<div class='ai-insight-box'>{res.text}</div>", unsafe_allow_html=True)
                except:
                    st.warning("AI cooling down.")

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

        if st.button("Interpret Risk Exposure"):
            with st.spinner("Analyzing portfolio risk..."):
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    res = model.generate_content(
                        f"""
                        Provide a professional risk assessment for a CFO:

                        - Core project share: {core_pct:.1f}%
                        - Exit candidate share: {exit_pct:.1f}%
                        - Value at Risk (95%): {abs(var95):.2%}
                        - Average Sharpe Ratio: {sharpe_avg:.2f}
                        - Risk quality: {risk_quality}
                        - Recommended action: {action}

                        Focus on capital protection, stability, and governance implications.
                        """
                    )
                    st.markdown(f"<div class='ai-insight-box'>{res.text}</div>", unsafe_allow_html=True)
                except:
                    st.warning("AI temporarily unavailable. Please retry.")

    
    elif nav == " INSTITUTIONAL THESIS":
        st.markdown('<div class="section-header">PROJECT DEEP-DIVE THESIS</div>', unsafe_allow_html=True)
        target = st.selectbox("SELECT PROJECT FOR QUANT ANALYSIS", selected['Project_ID'])
        r = selected[selected['Project_ID'] == target].iloc[0]
        
        if st.button("Interpret Project Thesis"):
            with st.spinner("AI Quant at work..."):
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    res = model.generate_content(f"Deep Quant analysis for project {target}. Strategic Value ${r['Strategic_Value']:.2f}, Profitability Index {r['PI']:.2f}, Risk Score {r['Risk_Score']:.1f}. Provide a professional investment thesis.")
                    st.markdown(f"<div class='ai-insight-box'>{res.text}</div>", unsafe_allow_html=True)
                except: st.error("AI rate limit. Wait 60s.")

    st.markdown("---")
    st.download_button(" DOWNLOAD (CSV)", selected.to_csv(index=False), file_name="STRATOS_Approved_Portfolio.csv", mime="text/csv")

if __name__ == "__main__":
    main()
