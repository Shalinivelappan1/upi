import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# -----------------------------
# HEADER
# -----------------------------
st.title("📱 Banking Lab V5 — UPI, FinTech & Who Pays the Bill?")
st.caption("Payments are free for users, but someone pays — Stylized economics for teaching")

st.warning("⚠️ Educational use only. Stylized model — not regulatory advice.")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("📈 Scale of the System")
users_m = st.sidebar.slider("Active Users (Millions)", 1, 500, 100)
tx_per_user = st.sidebar.slider("Transactions per User / Month", 1, 100, 30)

st.sidebar.header("🏗 Payment System Costs")
infra_cost = st.sidebar.slider("Platform Infra Cost per Tx (₹)", 0.1, 2.0, 0.5)
bank_cost = st.sidebar.slider("Bank Ops / Compliance Cost per Tx (₹)", 0.1, 2.0, 0.7)

st.sidebar.header("💳 Fee Charged to Banks")
fee = st.sidebar.slider("Fee per Tx (₹)", 0.0, 1.0, 0.2)

st.sidebar.header("🏦 Cross-Sell: Lending")
conversion = st.sidebar.slider("% Users Converted", 0.0, 20.0, 5.0) / 100
loan_size = st.sidebar.slider("Avg Loan Size (₹)", 1000, 500000, 100000)
nim = st.sidebar.slider("Net Interest Margin (%)", 1.0, 20.0, 5.0) / 100

st.sidebar.header("💰 Float / Wallet")
avg_balance = st.sidebar.slider("Avg Wallet Balance (₹)", 0, 5000, 1000)
float_return = st.sidebar.slider("Return on Float (%)", 0.0, 10.0, 4.0) / 100

# -----------------------------
# CORE CALCULATIONS
# -----------------------------
users = users_m * 1_000_000
annual_tx = users * tx_per_user * 12

platform_cost_total = annual_tx * infra_cost
bank_cost_total = annual_tx * bank_cost
platform_fee_revenue = annual_tx * fee

# Lending
loan_users = users * conversion
loan_book = loan_users * loan_size
lending_profit = loan_book * nim

# Float
float_pool = users * avg_balance
float_income = float_pool * float_return

platform_profit = platform_fee_revenue + lending_profit + float_income - platform_cost_total

# -----------------------------
# TABS
# -----------------------------
tabs = st.tabs([
    "🧩 System Economics",
    "📊 Diffusion Lab",
    "⚖️ Who Pays What",
    "📈 Scale vs Sustainability",
    "🧠 Intuition Summary"
])

# -----------------------------
# TAB 1: SYSTEM ECONOMICS
# -----------------------------
with tabs[0]:
    st.subheader("🧩 UPI / Payment System at National Scale")

    col1, col2, col3 = st.columns(3)

    col1.metric("Active Users", f"{users_m} million")
    col1.metric("Annual Transactions", f"{annual_tx/1e9:.2f} billion")

    col2.metric("Platform Infra Cost", f"₹{platform_cost_total/1e7:.2f} Cr")
    col2.metric("Bank Cost", f"₹{bank_cost_total/1e7:.2f} Cr")

    col3.metric("Platform Fee Revenue", f"₹{platform_fee_revenue/1e7:.2f} Cr")
    col3.metric("Lending Profit", f"₹{lending_profit/1e7:.2f} Cr")

    st.metric("Float Income", f"₹{float_income/1e7:.2f} Cr")
    st.metric("Platform Profit", f"₹{platform_profit/1e7:.2f} Cr")

    if platform_profit > 0:
        st.success("🙂 The platform is PROFITABLE — driven by cross-subsidy.")
    else:
        st.error("⚠️ The platform is NOT sustainable without subsidy.")

# -----------------------------
# TAB 2: DIFFUSION LAB
# -----------------------------
with tabs[1]:
    st.subheader("📊 Bass Diffusion Model (Fit from Data)")

    uploaded_file = st.file_uploader("Upload CSV (columns: Period, Volume)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        t = df['Period'].values
        y = df['Volume'].values

        def bass_model(t, p, q, m):
            exp_term = np.exp(-(p + q) * t)
            return m * (1 - exp_term) / (1 + (q / p) * exp_term)

        try:
            params, _ = curve_fit(bass_model, t, y, bounds=(0, [1, 1, 1e5]))
            p, q, m = params

            st.success(f"Estimated p={p:.6f}, q={q:.4f}, M={m:.0f}")

            forecast = bass_model(t, p, q, m)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=y, name="Actual"))
            fig.add_trace(go.Scatter(x=t, y=forecast, name="Bass Forecast"))

            st.plotly_chart(fig, use_container_width=True)

            st.metric("q/p Ratio", f"{q/p:.0f}x")

        except:
            st.error("Model fitting failed. Check data format.")

# -----------------------------
# TAB 3: WHO PAYS WHAT
# -----------------------------
with tabs[2]:
    st.subheader("⚖️ Value Flow Analysis")

    st.write(f"💸 Banks bear cost: ₹{bank_cost_total/1e7:.2f} Cr")
    st.write(f"🏗 Platform infra cost: ₹{platform_cost_total/1e7:.2f} Cr")

    st.write(f"💳 Platform earns fees: ₹{platform_fee_revenue/1e7:.2f} Cr")
    st.write(f"📈 Lending profit: ₹{lending_profit/1e7:.2f} Cr")
    st.write(f"💰 Float income: ₹{float_income/1e7:.2f} Cr")

    st.info("👉 Banks subsidize infrastructure. Platforms monetize users.")

# -----------------------------
# TAB 4: SCALE VS SUSTAINABILITY
# -----------------------------
with tabs[3]:
    st.subheader("📈 Scaling Dynamics")

    tx_range = np.linspace(1e6, annual_tx, 50)

    cost_curve = tx_range * infra_cost
    revenue_curve = tx_range * fee

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tx_range, y=cost_curve, name="Cost"))
    fig.add_trace(go.Scatter(x=tx_range, y=revenue_curve, name="Fee Revenue"))

    st.plotly_chart(fig, use_container_width=True)

    st.warning("⚠️ Scale increases costs linearly — revenue must come from cross-subsidy.")

# -----------------------------
# TAB 5: INTUITION
# -----------------------------
with tabs[4]:
    st.subheader("🧠 Core Insight")

    st.info("""
    UPI is not truly free.

    Users pay ₹0.
    Merchants often pay ₹0.

    But:
    - Banks absorb infrastructure and compliance cost
    - Platforms monetize via:
        • Lending
        • Float
        • Data & ecosystem lock-in

    👉 Scale does NOT fix unit economics.
    👉 It amplifies the subsidy burden — unless monetization kicks in.
    """)
