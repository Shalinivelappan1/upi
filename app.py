import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(layout="wide")

# -------------------------------
# TITLE
# -------------------------------
st.title("📊 Digital Payments Diffusion Lab (Advanced)")
st.markdown("Bass Diffusion + Policy Shocks + Economic Layer + Data Fitting")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Model Configuration")

mode = st.sidebar.radio("Mode", ["Simulation", "Fit Real Data"])

# -------------------------------
# BASS MODEL
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
# SHOCK FUNCTION
# -------------------------------
def apply_shock(cumulative, shock_time, shock_strength, shock_type, M):
    result = cumulative.copy()

    for t in range(shock_time, len(cumulative)):
        if shock_type == "Acceleration":
            result[t] *= (1 + shock_strength)
        elif shock_type == "Break":
            result[t] *= (1 - abs(shock_strength))
        elif shock_type == "Mandate":
            result[t] += M * shock_strength * 0.2

    return result

# -------------------------------
# PARAM INPUTS
# -------------------------------
st.sidebar.subheader("Bass Parameters")

p = st.sidebar.slider("p (innovation)", 0.000001, 0.01, 0.00002, format="%.6f")
q = st.sidebar.slider("q (imitation)", 0.001, 1.0, 0.06)
M = st.sidebar.slider("Market Size", 1, 500, 100)
T = st.sidebar.slider("Time Periods", 12, 120, 72)

# Shock
st.sidebar.subheader("Shock")
shock_type = st.sidebar.selectbox("Type", ["None", "Acceleration", "Break", "Mandate"])
shock_time = st.sidebar.slider("Shock Time", 1, T, 36)
shock_strength = st.sidebar.slider("Shock Strength", -0.5, 1.0, 0.3)

# Economics
st.sidebar.subheader("Economics")

cost_per_tx = st.sidebar.slider("Bank Cost per Tx", 0.1, 5.0, 1.4)
platform_fee = st.sidebar.slider("Platform Fee", 0.0, 2.0, 0.2)
conversion_rate = st.sidebar.slider("Loan Conversion %", 0, 20, 5)
avg_loan = st.sidebar.slider("Avg Loan", 1000, 100000, 20000)

# -------------------------------
# SIMULATION MODE
# -------------------------------
if mode == "Simulation":

    adopters, cumulative = bass_model(p, q, M, T)

    if shock_type != "None":
        cumulative_shock = apply_shock(cumulative, shock_time, shock_strength, shock_type, M)
    else:
        cumulative_shock = cumulative

    transactions = cumulative_shock * 10

    # Economics
    bank_cost = transactions * cost_per_tx
    platform_revenue = transactions * platform_fee

    converted_users = cumulative_shock * (conversion_rate / 100)
    loan_volume = converted_users * avg_loan
    platform_profit = platform_revenue + (loan_volume * 0.05)

    # -------------------------------
    # PLOTS
    # -------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Adoption Curve")

        fig, ax = plt.subplots()
        ax.plot(cumulative, label="Bass Forecast", linestyle="--")
        ax.plot(cumulative_shock, label="Actual", linewidth=2)
        ax.axvline(shock_time, linestyle=":", label="Shock")
        ax.legend()

        st.pyplot(fig)

    with col2:
        st.subheader("Economics")

        fig, ax = plt.subplots()
        ax.plot(bank_cost, label="Bank Cost")
        ax.plot(platform_profit, label="Platform Profit")
        ax.legend()

        st.pyplot(fig)

    # -------------------------------
    # METRICS
    # -------------------------------
    st.subheader("Key Metrics")

    c1, c2, c3 = st.columns(3)

    c1.metric("Final Users", f"{cumulative_shock[-1]:,.2f}")
    c2.metric("Bank Cost", f"{bank_cost.sum():,.0f}")
    c3.metric("Platform Profit", f"{platform_profit.sum():,.0f}")

    # Interpretation
    st.subheader("Insight")

    if q > p * 1000:
        st.success("Network effects dominate (UPI-like)")
    elif shock_type == "Break":
        st.warning("Structural break detected (IMPS-like)")
    elif shock_type == "Mandate":
        st.info("Policy-driven adoption (FASTag-like)")
    else:
        st.write("Moderate diffusion")

# -------------------------------
# FIT REAL DATA MODE
# -------------------------------
else:
    st.subheader("Upload Data")

    file = st.file_uploader("Upload CSV with 'time' and 'adoption' columns")

    if file:
        df = pd.read_csv(file)

        y = df["adoption"].values
        T = len(y)

        def loss(params):
            p, q, M = params
            _, pred = bass_model(p, q, M, T)
            return np.mean((y - pred) ** 2)

        result = minimize(loss, [0.001, 0.1, max(y)], bounds=[(1e-6,1),(1e-6,1),(max(y), 10*max(y))])

        p_fit, q_fit, M_fit = result.x

        _, pred = bass_model(p_fit, q_fit, M_fit, T)

        st.write("### Estimated Parameters")
        st.write(f"p: {p_fit:.6f}")
        st.write(f"q: {q_fit:.4f}")
        st.write(f"M: {M_fit:.2f}")

        fig, ax = plt.subplots()
        ax.plot(y, label="Actual")
        ax.plot(pred, label="Fitted Bass", linestyle="--")
        ax.legend()

        st.pyplot(fig)

        # Export
        output = pd.DataFrame({"Actual": y, "Fitted": pred})
        st.download_button("Download Results", output.to_csv(index=False), "results.csv")
