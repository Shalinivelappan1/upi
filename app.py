import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

st.set_page_config(page_title="UPI Diffusion Lab", layout="wide")

# =========================================================
# 📂 DATA LOADER
# =========================================================
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df.columns = df.columns.str.strip()

    # Clean volume
    df["Volume"] = df["Volume"].astype(str).str.replace(",", "")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df = df.dropna()

    # Ensure Period exists
    if "Period" not in df.columns:
        df["Period"] = np.arange(1, len(df) + 1)

    return df


# =========================================================
# 📈 BASS MODEL
# =========================================================
def bass_cumulative(t, p, q, M):
    exp_term = np.exp(-(p + q) * t)
    F_t = (1 - exp_term) / (1 + (q / p) * exp_term)
    return M * F_t


# =========================================================
# ⚡ SHOCK MODEL
# =========================================================
def shock_function(t, shocks):
    shock_effect = np.zeros_like(t, dtype=float)

    for shock_time, magnitude, decay in shocks:
        effect = magnitude * np.exp(-decay * np.maximum(0, t - shock_time))
        shock_effect += effect

    return shock_effect


def bass_with_shocks(t, p, q, M, shocks):
    base = bass_cumulative(t, p, q, M)
    shock_effect = shock_function(t, shocks)
    return base * (1 + shock_effect)


# =========================================================
# 🔍 PARAMETER ESTIMATION
# =========================================================
def estimate_bass_parameters(df):
    t = df["Period"].values
    y = df["Volume"].values

    p0, q0, M0 = 0.0001, 0.1, max(y) * 1.5

    try:
        params, _ = curve_fit(
            bass_cumulative,
            t,
            y,
            p0=[p0, q0, M0],
            bounds=(0, [1, 1, 1e8]),
            maxfev=20000
        )
        return params
    except:
        return None


# =========================================================
# 🧠 PIECEWISE ESTIMATION
# =========================================================
def piecewise_estimation(df, split_point):
    df_pre = df[df["Period"] <= split_point]
    df_post = df[df["Period"] > split_point]

    pre = estimate_bass_parameters(df_pre)
    post = estimate_bass_parameters(df_post)

    return pre, post


# =========================================================
# 💰 ECONOMICS
# =========================================================
def compute_economics(users, tx_per_user, cost, fee):
    tx = users * tx_per_user
    cost_total = np.sum(tx * cost)
    revenue = np.sum(tx * fee)

    return {
        "Transactions": np.sum(tx),
        "Cost": cost_total,
        "Revenue": revenue,
        "Profit": revenue - cost_total
    }


# =========================================================
# 🎛 SIDEBAR
# =========================================================
st.sidebar.title("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV/Excel",
    type=["csv", "xlsx"]
)

st.sidebar.subheader("Bass Parameters (Manual)")
p = st.sidebar.slider("p (Innovation)", 0.000001, 0.01, 0.0001)
q = st.sidebar.slider("q (Imitation)", 0.01, 1.0, 0.1)
M = st.sidebar.slider("M (Market)", 1000, 20000, 10000)

T = st.sidebar.slider("Simulation Periods", 10, 120, 60)

# Shocks
st.sidebar.subheader("⚡ Policy Shocks")
scenario = st.sidebar.selectbox(
    "Scenario",
    ["None", "Demonetization", "COVID", "FASTag", "All"]
)

# Piecewise
split_point = st.sidebar.slider("Split (COVID)", 20, 80, 45)

# Economics
st.sidebar.subheader("💰 Economics")
tx_per_user = st.sidebar.slider("Tx/User", 1, 100, 30)
cost = st.sidebar.slider("Cost per Tx", 0.1, 5.0, 1.4)
fee = st.sidebar.slider("Fee per Tx", 0.0, 2.0, 0.2)


# =========================================================
# 🎯 POLICY SHOCK MAPPER
# =========================================================
def get_shocks(scenario):
    if scenario == "Demonetization":
        return [(5, 0.6, 0.1)]
    elif scenario == "COVID":
        return [(45, -0.4, 0.2)]
    elif scenario == "FASTag":
        return [(40, 0.8, 0.05)]
    elif scenario == "All":
        return [(5, 0.6, 0.1), (40, 0.8, 0.05), (45, -0.4, 0.2)]
    return []


# =========================================================
# 🧠 MAIN
# =========================================================
st.title("📊 Digital Payments Diffusion Lab")

if uploaded_file:
    df = load_data(uploaded_file)
    st.success("Data Loaded")

    # Estimate
    params = estimate_bass_parameters(df)

    if params is not None:
        p_est, q_est, M_est = params

        col1, col2, col3 = st.columns(3)
        col1.metric("p", f"{p_est:.6f}")
        col2.metric("q", f"{q_est:.4f}")
        col3.metric("M", f"{M_est:,.0f}")

        ratio = q_est / p_est if p_est > 0 else 0
        st.metric("q/p Ratio", f"{ratio:,.0f}x")

        t = df["Period"].values

        shocks = get_shocks(scenario)

        fitted = bass_with_shocks(t, p_est, q_est, M_est, shocks)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df["Volume"], name="Actual"))
        fig.add_trace(go.Scatter(y=fitted, name="Model", line=dict(dash="dash")))
        st.plotly_chart(fig, use_container_width=True)

        # Piecewise
        pre, post = piecewise_estimation(df, split_point)

        if pre and post:
            st.subheader("📊 Piecewise")
            c1, c2 = st.columns(2)

            c1.write(f"Pre-COVID p={pre[0]:.6f}, q={pre[1]:.4f}")
            c2.write(f"Post-COVID p={post[0]:.6f}, q={post[1]:.4f}")

        users = df["Volume"].values

    else:
        st.error("Estimation Failed")
        users = df["Volume"].values

else:
    st.warning("Using Simulation Mode")

    t = np.arange(1, T + 1)
    shocks = get_shocks(scenario)

    users = bass_with_shocks(t, p, q, M, shocks)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=users, name="Simulated"))
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 💰 ECONOMICS
# =========================================================
st.subheader("💰 Economics")

econ = compute_economics(users, tx_per_user, cost, fee)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{econ['Transactions']:,.0f}")
c2.metric("Cost", f"₹{econ['Cost']/1e9:.2f}B")
c3.metric("Revenue", f"₹{econ['Revenue']/1e9:.2f}B")
c4.metric("Profit", f"₹{econ['Profit']/1e9:.2f}B")

# =========================================================
# 🔥 HEATMAP (q/p LANDSCAPE)
# =========================================================
st.subheader("🔥 Imitation vs Innovation Landscape")

p_vals = np.linspace(0.00001, 0.001, 20)
q_vals = np.linspace(0.01, 0.2, 20)

Z = np.zeros((len(q_vals), len(p_vals)))

for i, qv in enumerate(q_vals):
    for j, pv in enumerate(p_vals):
        Z[i, j] = qv / pv

fig = go.Figure(data=go.Heatmap(
    z=Z,
    x=np.round(p_vals, 6),
    y=np.round(q_vals, 3),
    colorbar_title="q/p"
))

st.plotly_chart(fig, use_container_width=True)
