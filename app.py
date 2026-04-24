import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# -------------------------------
# TITLE
# -------------------------------
st.title("📊 Digital Payments Diffusion Simulator (UPI Lab)")

st.markdown("""
Simulate adoption dynamics using the **Bass Diffusion Model** + **Policy Shocks + Economics Layer**
""")

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("🔧 Model Parameters")

p = st.sidebar.slider("Innovation (p)", 0.000001, 0.01, 0.00002, format="%.6f")
q = st.sidebar.slider("Imitation (q)", 0.001, 1.0, 0.06)
M = st.sidebar.slider("Market Size (Millions)", 1, 1000, 100)

time_periods = st.sidebar.slider("Time (months)", 12, 120, 72)

# Shock inputs
st.sidebar.header("⚡ Policy / Shock Events")

shock_time = st.sidebar.slider("Shock Time (month)", 1, time_periods, 36)
shock_strength = st.sidebar.slider("Shock Strength (+ boost / - break)", -0.5, 1.0, 0.3)

shock_type = st.sidebar.selectbox(
    "Shock Type",
    ["None", "Acceleration (Demonetization)", "Break (COVID)", "Mandate (FASTag)"]
)

# Economics inputs
st.sidebar.header("💰 Economics Layer")

cost_per_tx = st.sidebar.slider("Bank Cost per Tx (₹)", 0.1, 5.0, 1.4)
platform_fee = st.sidebar.slider("Platform Fee per Tx (₹)", 0.0, 2.0, 0.2)

conversion_rate = st.sidebar.slider("Credit Conversion (%)", 0, 20, 5)
avg_loan = st.sidebar.slider("Avg Loan Value (₹)", 1000, 100000, 20000)

# -------------------------------
# BASS MODEL FUNCTION
# -------------------------------
def bass_model(p, q, M, T):
    adopters = np.zeros(T)
    cumulative = np.zeros(T)

    for t in range(1, T):
        adoption = (p + q * cumulative[t-1] / M) * (M - cumulative[t-1])
        adopters[t] = adoption
        cumulative[t] = cumulative[t-1] + adoption

    return adopters, cumulative

# -------------------------------
# BASE SIMULATION
# -------------------------------
adopters, cumulative = bass_model(p, q, M, time_periods)

# -------------------------------
# APPLY SHOCK
# -------------------------------
cumulative_shock = cumulative.copy()

if shock_type != "None":
    for t in range(shock_time, time_periods):
        if shock_type == "Acceleration (Demonetization)":
            cumulative_shock[t] *= (1 + shock_strength)

        elif shock_type == "Break (COVID)":
            cumulative_shock[t] *= (1 - abs(shock_strength))

        elif shock_type == "Mandate (FASTag)":
            cumulative_shock[t] += M * shock_strength * 0.2

# -------------------------------
# TRANSACTION VOLUME (proxy)
# -------------------------------
transactions = cumulative_shock * 10  # assume 10 tx per user

# -------------------------------
# ECONOMICS
# -------------------------------
bank_cost = transactions * cost_per_tx
platform_revenue = transactions * platform_fee

# Lending engine
converted_users = cumulative_shock * (conversion_rate / 100)
loan_volume = converted_users * avg_loan
platform_profit = platform_revenue + (loan_volume * 0.05)

# -------------------------------
# PLOTS
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Adoption Curve")

    fig, ax = plt.subplots()
    ax.plot(cumulative, label="Bass Forecast", linestyle="--")
    ax.plot(cumulative_shock, label="Actual (with shocks)", linewidth=2)

    ax.axvline(shock_time, linestyle=":", label="Shock Event")
    ax.set_xlabel("Time")
    ax.set_ylabel("Adoption")
    ax.legend()

    st.pyplot(fig)

with col2:
    st.subheader("💳 Transaction Volume")

    fig, ax = plt.subplots()
    ax.plot(transactions, color="blue")
    ax.set_title("Transaction Volume Over Time")

    st.pyplot(fig)

# -------------------------------
# ECONOMICS VISUAL
# -------------------------------
st.subheader("💰 System Economics")

fig, ax = plt.subplots()

ax.plot(bank_cost, label="Bank Cost", linewidth=2)
ax.plot(platform_profit, label="Platform Profit", linewidth=2)

ax.set_title("Cross-Subsidy Engine")
ax.legend()

st.pyplot(fig)

# -------------------------------
# METRICS
# -------------------------------
st.subheader("📊 Key Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Users (M)", f"{cumulative_shock[-1]:,.2f}")

with col2:
    st.metric("Bank Total Cost (₹)", f"{bank_cost.sum():,.0f}")

with col3:
    st.metric("Platform Profit (₹)", f"{platform_profit.sum():,.0f}")

# -------------------------------
# INTERPRETATION
# -------------------------------
st.markdown("### 🧠 Interpretation")

if q > p * 1000:
    st.success("Imitation dominates → Network effects drive explosive growth (UPI-like)")
elif shock_type == "Break (COVID)":
    st.warning("Structural break → adoption deviates permanently (IMPS-like)")
elif shock_type == "Mandate (FASTag)":
    st.info("Policy-driven adoption → model fails to predict spike")
else:
    st.write("Balanced growth → moderate diffusion pattern")
