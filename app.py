import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.title("📱 Banking Lab V6 — Diffusion, Shocks & Who Pays the Bill?")
st.caption("Bass Diffusion + Policy Shocks + Cross-Subsidy Economics")
st.warning("⚠️ Educational use only. Stylized model.")

# ---------------------------------------------------
# SIDEBAR — ECONOMICS
# ---------------------------------------------------
st.sidebar.header("📈 Scale of the System")
users_m = st.sidebar.slider("Active Users (Millions)", 1, 500, 100)
tx_per_user = st.sidebar.slider("Transactions per User / Month", 1, 100, 30)

st.sidebar.header("🏗 Payment System Costs")
infra_cost = st.sidebar.slider("Platform Cost per Tx (₹)", 0.1, 2.0, 0.5)
bank_cost = st.sidebar.slider("Bank Cost per Tx (₹)", 0.1, 2.0, 0.7)

st.sidebar.header("💳 Fee Charged to Banks")
fee = st.sidebar.slider("Fee per Tx (₹)", 0.0, 1.0, 0.2)

st.sidebar.header("🏦 Lending Engine")
conversion = st.sidebar.slider("% Users Converted", 0.0, 20.0, 5.0) / 100
loan_size = st.sidebar.slider("Avg Loan Size (₹)", 1000, 500000, 100000)
nim = st.sidebar.slider("Net Interest Margin (%)", 1.0, 20.0, 5.0) / 100

st.sidebar.header("💰 Float")
avg_balance = st.sidebar.slider("Avg Wallet Balance (₹)", 0, 5000, 1000)
float_return = st.sidebar.slider("Return on Float (%)", 0.0, 10.0, 4.0) / 100

# ---------------------------------------------------
# SIDEBAR — POLICY SHOCKS
# ---------------------------------------------------
st.sidebar.header("⚡ Policy Scenario")

scenario = st.sidebar.selectbox(
    "Select Scenario",
    ["Baseline", "Demonetization", "COVID", "FASTag", "All Combined"]
)

split_point = st.sidebar.slider("Piecewise Split (COVID Period)", 20, 80, 45)

# ---------------------------------------------------
# CORE ECONOMICS
# ---------------------------------------------------
users = users_m * 1_000_000
annual_tx = users * tx_per_user * 12

platform_cost_total = annual_tx * infra_cost
bank_cost_total = annual_tx * bank_cost
platform_fee_revenue = annual_tx * fee

loan_users = users * conversion
loan_book = loan_users * loan_size
lending_profit = loan_book * nim

float_pool = users * avg_balance
float_income = float_pool * float_return

platform_profit = platform_fee_revenue + lending_profit + float_income - platform_cost_total

# ---------------------------------------------------
# BASS MODEL
# ---------------------------------------------------
def bass_model(t, p, q, m):
    exp_term = np.exp(-(p + q) * t)
    return m * (1 - exp_term) / (1 + (q / p) * exp_term)

# ---------------------------------------------------
# SHOCK FUNCTION
# ---------------------------------------------------
def get_shocks(scenario):
    if scenario == "Demonetization":
        return [(5, 0.6, 0.1)]
    elif scenario == "COVID":
        return [(45, -0.4, 0.2)]
    elif scenario == "FASTag":
        return [(40, 0.8, 0.05)]
    elif scenario == "All Combined":
        return [(5, 0.6, 0.1), (40, 0.8, 0.05), (45, -0.4, 0.2)]
    return []

def apply_shocks(t, base, shocks):
    shock_effect = np.zeros_like(t, dtype=float)
    for time, mag, decay in shocks:
        shock_effect += mag * np.exp(-decay * np.maximum(0, t - time))
    return base * (1 + shock_effect)

# ---------------------------------------------------
# PARAMETER ESTIMATION
# ---------------------------------------------------
def estimate_bass(df):
    t = df["Period"].values
    y = df["Volume"].values

    guess = [1e-5, 0.05, max(y) * 1.5]

    params, _ = curve_fit(
        bass_model,
        t,
        y,
        p0=guess,
        bounds=([1e-7, 1e-4, max(y)], [1, 1, max(y)*10]),
        maxfev=20000
    )
    return params

# ---------------------------------------------------
# PIECEWISE ESTIMATION
# ---------------------------------------------------
def piecewise(df, split):
    df1 = df[df["Period"] <= split]
    df2 = df[df["Period"] > split]

    p1, q1, m1 = estimate_bass(df1)
    p2, q2, m2 = estimate_bass(df2)

    return (p1, q1, m1), (p2, q2, m2)

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tabs = st.tabs([
    "🧩 System Economics",
    "📊 Diffusion + Policy",
    "⚖️ Who Pays What",
    "📈 Scale vs Sustainability",
    "🧠 Intuition"
])

# ---------------------------------------------------
# TAB 1 — ECONOMICS
# ---------------------------------------------------
with tabs[0]:
    st.subheader("🧩 System Economics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Users", f"{users_m} million")
    col1.metric("Transactions", f"{annual_tx/1e9:.2f} billion")

    col2.metric("Platform Cost", f"₹{platform_cost_total/1e7:.2f} Cr")
    col2.metric("Bank Cost", f"₹{bank_cost_total/1e7:.2f} Cr")

    col3.metric("Fee Revenue", f"₹{platform_fee_revenue/1e7:.2f} Cr")
    col3.metric("Lending Profit", f"₹{lending_profit/1e7:.2f} Cr")

    st.metric("Float Income", f"₹{float_income/1e7:.2f} Cr")
    st.metric("Platform Profit", f"₹{platform_profit/1e7:.2f} Cr")

# ---------------------------------------------------
# TAB 2 — DIFFUSION + SHOCKS
# ---------------------------------------------------
with tabs[1]:
    st.subheader("📊 Diffusion + Policy Shocks")

    file = st.file_uploader("Upload CSV (Period, Volume)", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        df["Volume"] = df["Volume"].astype(str).str.replace(",", "", regex=False).astype(float)
        df["Period"] = pd.to_numeric(df["Period"])
        df = df.dropna()

        t = df["Period"].values
        y = df["Volume"].values

        p, q, m = estimate_bass(df)

        st.success(f"p={p:.6f}, q={q:.4f}, M={m:.0f}")
        st.metric("q/p Ratio", f"{q/p:.0f}x")

        base = bass_model(t, p, q, m)
        shocked = apply_shocks(t, base, get_shocks(scenario))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y, name="Actual"))
        fig.add_trace(go.Scatter(x=t, y=base, name="Bass"))
        fig.add_trace(go.Scatter(x=t, y=shocked, name=f"{scenario}"))

        st.plotly_chart(fig, use_container_width=True)

        # Piecewise
        st.subheader("📊 Piecewise (Pre/Post COVID)")
        (p1, q1, _), (p2, q2, _) = piecewise(df, split_point)

        col1, col2 = st.columns(2)
        col1.write(f"Pre: p={p1:.6f}, q={q1:.4f}")
        col2.write(f"Post: p={p2:.6f}, q={q2:.4f}")

# ---------------------------------------------------
# TAB 3 — WHO PAYS
# ---------------------------------------------------
with tabs[2]:
    st.subheader("⚖️ Value Flow")

    st.write(f"🏦 Banks pay: ₹{bank_cost_total/1e7:.2f} Cr")
    st.write(f"💰 Platform profit: ₹{platform_profit/1e7:.2f} Cr")

# ---------------------------------------------------
# TAB 4 — SCALE
# ---------------------------------------------------
with tabs[3]:
    st.subheader("📈 Scale Dynamics")

    tx_range = np.linspace(1e6, annual_tx, 100)
    cost_curve = tx_range * infra_cost
    revenue_curve = tx_range * fee

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tx_range, y=cost_curve, name="Cost"))
    fig.add_trace(go.Scatter(x=tx_range, y=revenue_curve, name="Revenue"))

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# TAB 5 — INTUITION
# ---------------------------------------------------
with tabs[4]:
    st.info("""
    UPI is a zero-price system, not zero-cost.

    Banks fund infrastructure.
    Platforms monetize behavior.
    Policy shocks accelerate adoption.

    👉 Imitation (q) dominates innovation (p)
    👉 Shocks reshape diffusion paths
    """)
