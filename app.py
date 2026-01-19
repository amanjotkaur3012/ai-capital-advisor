import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import pulp

# =========================================================
# 1. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CapitalAI | Investment Decision System",
    layout="wide",
    page_icon="📊"
)

st.markdown("""
<style>
.stApp { background-color: #0f172a; }
h1,h2,h3 { color:#f8fafc; }
p,span,label { color:#cbd5e1; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. SIDEBAR (FINANCE INPUTS)
# =========================================================
with st.sidebar:
    st.title("CapitalAI")
    st.caption("AI Driven Capital Allocation Advisor")

    budget = st.number_input("Total Budget (₹)", 5_000_000, 50_000_000, 15_000_000, step=500_000)
    wacc = st.slider("WACC (%)", 5.0, 20.0, 10.0) / 100
    max_risk = st.slider("Max Risk Appetite", 1.0, 10.0, 6.0)

    st.markdown("---")
    st.info("AI = ROI Prediction\nFinance = NPV + WACC\nOptimization = Allocation")

# =========================================================
# 3. DATA (DEMO – YOU CAN ADD UPLOAD LATER)
# =========================================================
@st.cache_data
def load_data():
    hist = pd.DataFrame({
        "Investment": np.random.randint(500_000, 5_000_000, 100),
        "Risk": np.random.uniform(1, 10, 100),
        "Strategy": np.random.uniform(1, 10, 100),
        "Market": np.random.uniform(0.8, 1.3, 100),
        "ROI": np.random.uniform(5, 25, 100)
    })

    prop = pd.DataFrame({
        "Project": [f"P{i:02d}" for i in range(1, 15)],
        "Investment": np.random.randint(500_000, 4_000_000, 14),
        "Risk": np.random.uniform(1, 9, 14),
        "Strategy": np.random.uniform(4, 10, 14),
        "Market": np.random.uniform(0.9, 1.2, 14)
    })
    return hist, prop

hist, prop = load_data()

# =========================================================
# 4. AI MODEL (ROI PREDICTION)
# =========================================================
features = ["Investment", "Risk", "Strategy", "Market"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(hist[features], hist["ROI"])

prop["Pred_ROI"] = model.predict(prop[features])

# =========================================================
# 5. FINANCE LOGIC (NPV + FILTERING)
# =========================================================
def calculate_npv(row, wacc):
    annual_return = row["Investment"] * (row["Pred_ROI"] / 100)
    return (annual_return / wacc) - row["Investment"]

prop["NPV"] = prop.apply(lambda r: calculate_npv(r, wacc), axis=1)
prop["Efficiency"] = prop["Pred_ROI"] / prop["Risk"]

# risk screening (real corporate logic)
prop = prop[prop["Risk"] <= max_risk].reset_index(drop=True)

# =========================================================
# 6. OPTIMIZATION (CAPITAL ALLOCATION)
# =========================================================
def optimize(df, budget):
    prob = pulp.LpProblem("Capital_Allocation", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("Select", df.index, cat="Binary")

    prob += pulp.lpSum(df.loc[i, "NPV"] * x[i] for i in df.index)
    prob += pulp.lpSum(df.loc[i, "Investment"] * x[i] for i in df.index) <= budget

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    df["Selected"] = [int(x[i].value()) for i in df.index]
    return df

prop = optimize(prop, budget)
portfolio = prop[prop["Selected"] == 1]

# =========================================================
# 7. DASHBOARD
# =========================================================
st.title("📊 Executive Investment Dashboard")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Projects Selected", len(portfolio))
k2.metric("Capital Used", f"₹{portfolio['Investment'].sum():,.0f}")
k3.metric("Avg ROI", f"{portfolio['Pred_ROI'].mean():.2f}%")
k4.metric("Total NPV", f"₹{portfolio['NPV'].sum():,.0f}")

st.markdown("---")

# =========================================================
# 8. TABLE + CHART
# =========================================================
st.subheader("Optimized Investment Plan")

st.dataframe(
    prop.style.format({
        "Investment": "₹{:,.0f}",
        "Pred_ROI": "{:.2f}%",
        "NPV": "₹{:,.0f}"
    }).background_gradient(subset=["NPV"], cmap="Greens"),
    use_container_width=True
)

fig = px.scatter(
    prop, x="Risk", y="Pred_ROI", size="Investment",
    color="Selected",
    title="Risk vs Return Map",
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 9. EXECUTIVE SUMMARY
# =========================================================
st.subheader("🧠 Investment Committee Summary")

if not portfolio.empty:
    st.success(f"""
    The system recommends investing in **{len(portfolio)} projects**, deploying 
    **₹{portfolio['Investment'].sum():,.0f}** of capital.  
    The portfolio generates an expected **average ROI of {portfolio['Pred_ROI'].mean():.2f}%**, 
    exceeding the cost of capital (WACC = {wacc*100:.1f}%).  

    Projects were selected using **AI-based return forecasting** and **NPV-maximizing optimization**.
    """)
else:
    st.warning("No projects met risk or budget constraints.")

# =========================================================
# 10. DOWNLOAD
# =========================================================
csv = portfolio.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Approved Investment Plan",
    csv,
    "capital_allocation_plan.csv",
    "text/csv"
)
