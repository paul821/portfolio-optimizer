# app.py — Portfolio Optimizer (One-Fund, GMV, Tangency, Frontier) - DEBUGGED
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import streamlit as st
import matplotlib.pyplot as plt

# ------------- math helpers -------------
def _as_col(x):
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 1)

def _check_inputs(mu, Sigma, ridge=1e-10):
    mu = _as_col(mu)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError(f"Covariance must be n×n with n = len(mu). Got Sigma shape {Sigma.shape} but mu length {n}.")
    # Symmetrize + tiny ridge for stability
    if not np.allclose(Sigma, Sigma.T, atol=1e-10):
        Sigma = (Sigma + Sigma.T) / 2
    Sigma = Sigma + ridge * np.eye(n)
    ones = np.ones((n, 1))
    return mu, Sigma, ones, n

def portfolio_stats(w, mu, Sigma):
    w = _as_col(w)
    m = float(w.T @ mu)
    v = float(w.T @ Sigma @ w)
    s = float(np.sqrt(max(v, 0)))
    return m, s, v, (m / s if s > 0 else np.nan)

# ------------- optimizers -------------
def one_fund_single_risky(mu_s, rf, sigma_s, A):
    # w* in risky = (mu_s - rf) / (A sigma_s^2)
    if sigma_s <= 0:
        raise ValueError("Standard deviation must be positive")
    if A <= 0:
        raise ValueError("Risk aversion must be positive")
    w_star = (mu_s - rf) / (A * sigma_s**2)
    mu_p = w_star * mu_s + (1 - w_star) * rf
    sigma_p = abs(w_star) * sigma_s
    return w_star, mu_p, sigma_p

def gmv_unconstrained(Sigma):
    Sigma = np.asarray(Sigma, dtype=float)
    n = Sigma.shape[0]
    ones = np.ones((n, 1))
    w = np.linalg.solve(Sigma, ones)
    w /= float(ones.T @ w)
    return w.flatten()

def gmv_long_only(Sigma, starts=30, maxiter=800):
    Sigma = np.asarray(Sigma, dtype=float)
    n = Sigma.shape[0]
    def obj(w): return float(w @ Sigma @ w)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n

    rng = np.random.default_rng(0)
    seeds = [np.ones(n)/n] + [rng.random(n) for _ in range(starts)]
    best = None
    for z in seeds:
        z = z / z.sum()
        res = minimize(obj, z, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": maxiter, "ftol": 1e-9})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("GMV (long-only) failed to converge.")
    return best.x

def tangency_unconstrained(mu, Sigma, rf):
    mu, Sigma, ones, n = _check_inputs(mu, Sigma)
    excess = mu - rf * ones
    w_unnorm = np.linalg.solve(Sigma, excess)
    w = w_unnorm / float(ones.T @ w_unnorm)
    return w.flatten()

def tangency_long_only(mu, Sigma, rf, starts=30, maxiter=800):
    mu, Sigma, ones, n = _check_inputs(mu, Sigma)
    mu_ex = (mu - rf * ones).flatten()
    def neg_sharpe(w):
        num = float(w @ mu_ex)
        den = float(np.sqrt(max(w @ Sigma @ w, 1e-18)))
        return -num / den
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n

    rng = np.random.default_rng(7)
    seeds = [np.ones(n)/n] + [rng.random(n) for _ in range(starts)]
    best = None
    for z in seeds:
        z = z / z.sum()
        res = minimize(neg_sharpe, z, method="SLSQP",
                       bounds=bounds, constraints=cons, options={"maxiter": maxiter, "ftol": 1e-9})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("Tangency (long-only) failed to converge.")
    return best.x

def min_var_for_target_return_unconstrained(mu, Sigma, target):
    mu, Sigma, ones, n = _check_inputs(mu, Sigma)
    inv = np.linalg.inv(Sigma)
    A = float(ones.T @ inv @ ones)
    B = float(ones.T @ inv @ mu)
    C = float(mu.T   @ inv @ mu)
    D = A*C - B**2
    if abs(D) < 1e-12:
        raise ValueError("Degenerate covariance structure (D ≈ 0)")
    lam = (C - B*target) / D
    gamma = (A*target - B) / D
    w = lam * (inv @ ones) + gamma * (inv @ mu)
    return w.flatten()

def efficient_frontier_unconstrained(mu, Sigma, n_points=60, ret_range=None):
    mu, Sigma, ones, _ = _check_inputs(mu, Sigma)
    mu_flat = mu.flatten()
    lo, hi = (float(mu_flat.min()) - 1e-6, float(mu_flat.max()) + 1e-6) if ret_range is None else ret_range
    targets = np.linspace(lo, hi, n_points)
    W, R, S = [], [], []
    for t in targets:
        try:
            w = min_var_for_target_return_unconstrained(mu, Sigma, t)
            m, s, _, _ = portfolio_stats(w, mu, Sigma)
            W.append(w); R.append(m); S.append(s)
        except Exception:
            continue  # Skip problematic points
    return np.array(W), np.array(R), np.array(S)

def efficient_frontier_long_only(mu, Sigma, n_points=60, theta_max_scale=50.0):
    # Stable approach: for θ in [0, Θ], minimize w'Σw - θ μ'w, s.t. 1'w=1, w>=0.
    mu, Sigma, ones, n = _check_inputs(mu, Sigma)
    mu_flat = mu.flatten()
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    scale = max(1e-3, float(np.abs(mu_flat).mean()))
    thetas = np.linspace(0.0, theta_max_scale * scale, n_points)
    W, R, S = [], [], []
    rng = np.random.default_rng(123)
    w_start = np.ones(n) / n
    for theta in thetas:
        def obj(w, th=theta): return float(w @ Sigma @ w - th * (w @ mu_flat))
        best = None
        # a couple of restarts
        for z in [w_start, rng.random(n)]:
            z = z / z.sum()
            res = minimize(obj, z, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"maxiter": 800, "ftol": 1e-9})
            if res.success and (best is None or res.fun < best.fun):
                best = res
        if best is None:
            w = w_start
        else:
            w = best.x
            w_start = w
        m, s, _, _ = portfolio_stats(w, mu, Sigma)
        W.append(w); R.append(m); S.append(s)
    return np.array(W), np.array(R), np.array(S)

# ------------- UI -------------
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("📈 Portfolio Optimizer")
st.caption("One-Fund (single risky), Global Minimum-Variance, Tangency (Max-Sharpe), and Efficient Frontier")

with st.sidebar:
    st.header("⚙️ Options")
    opt = st.selectbox(
        "Optimization type",
        ["One-Fund (single risky + rf)",
         "Global Minimum-Variance (GMV)",
         "Tangency / Max-Sharpe",
         "Efficient Frontier"]
    )
    long_only = st.checkbox("Enforce no short sales (long-only)", value=True, help="Applies to GMV, Tangency, and Frontier.")
    rf = st.number_input("Risk-free rate (rf)", value=0.03, step=0.005, format="%.4f")
    n_points = st.slider("Frontier points / sweep resolution", 20, 200, 80)

st.subheader("1) Define Assets & Inputs")

# Defaults
default_assets = ["Asset_A","Asset_B","Asset_C","Asset_D"]
default_mu = [0.08, 0.10, 0.06, 0.12]
default_cov = np.array([
    [0.10**2, 0.10*0.15*0.2, 0.10*0.08*0.1, 0.10*0.18*0.25],
    [0.10*0.15*0.2, 0.15**2, 0.15*0.08*0.15, 0.15*0.18*0.3],
    [0.10*0.08*0.1, 0.15*0.08*0.15, 0.08**2, 0.08*0.18*0.05],
    [0.10*0.18*0.25, 0.15*0.18*0.3, 0.08*0.18*0.05, 0.18**2],
])

col_a, col_b = st.columns([1, 1])
with col_a:
    n_assets = st.number_input("Number of assets", min_value=1, max_value=50, value=4, step=1)
    # asset names editor
    if n_assets <= len(default_assets):
        names_list = default_assets[:n_assets]
    else:
        extra = [f"Asset_{chr(65+i)}" for i in range(len(default_assets), n_assets)]
        names_list = default_assets + extra
    
    # Initialize session state for names if not exists
    if 'asset_names' not in st.session_state or len(st.session_state.asset_names) != n_assets:
        st.session_state.asset_names = names_list
    
    names_df = pd.DataFrame({"Asset": st.session_state.asset_names})
    edited_names = st.data_editor(names_df, use_container_width=True, hide_index=True, key="names_editor")
    asset_names = [str(x).strip() for x in edited_names["Asset"].tolist()]
    
    # Validation: check for duplicates
    if len(asset_names) != len(set(asset_names)):
        st.error("⚠️ Asset names must be unique!")
        st.stop()
    
    # Validation: check for empty names
    if any(name == "" for name in asset_names):
        st.error("⚠️ Asset names cannot be empty!")
        st.stop()
    
    st.session_state.asset_names = asset_names

with col_b:
    # expected returns vector editor
    init_mu = default_mu[:n_assets] if n_assets <= len(default_mu) else default_mu + [0.08]*(n_assets-len(default_mu))
    
    # Initialize session state for mu if not exists
    if 'mu_values' not in st.session_state or len(st.session_state.mu_values) != n_assets:
        st.session_state.mu_values = init_mu
    
    mu_df = pd.DataFrame({"ExpectedReturn": st.session_state.mu_values}, index=asset_names)
    edited_mu = st.data_editor(mu_df, use_container_width=True, key="mu_editor")
    st.session_state.mu_values = edited_mu["ExpectedReturn"].tolist()

# covariance matrix editor (grid)
st.markdown("#### Covariance Matrix")
if n_assets <= default_cov.shape[0]:
    init_cov = default_cov[:n_assets, :n_assets]
else:
    # expand with small values on diagonal
    init_cov = np.zeros((n_assets, n_assets))
    old_size = min(default_cov.shape[0], n_assets)
    init_cov[:old_size, :old_size] = default_cov[:old_size, :old_size]
    for i in range(default_cov.shape[0], n_assets):
        init_cov[i, i] = 0.1**2

# Initialize session state for covariance if not exists or wrong size
if 'cov_matrix' not in st.session_state or st.session_state.cov_matrix.shape[0] != n_assets:
    st.session_state.cov_matrix = init_cov

cov_df = pd.DataFrame(st.session_state.cov_matrix, index=asset_names, columns=asset_names)
edited_cov = st.data_editor(cov_df, use_container_width=True, key="cov_editor")

# Update session state
st.session_state.cov_matrix = edited_cov.to_numpy(dtype=float)

# parse inputs
try:
    mu = edited_mu["ExpectedReturn"].to_numpy(dtype=float)
    Sigma = edited_cov.to_numpy(dtype=float)
    
    # Validate dimensions
    if len(mu) != n_assets:
        raise ValueError(f"Expected returns vector has {len(mu)} entries, expected {n_assets}")
    if Sigma.shape != (n_assets, n_assets):
        raise ValueError(f"Covariance matrix has shape {Sigma.shape}, expected ({n_assets}, {n_assets})")
    
    # Check for positive definiteness (approximate)
    eigenvalues = np.linalg.eigvalsh(Sigma)
    if np.any(eigenvalues < -1e-8):
        st.warning("⚠️ Covariance matrix is not positive semi-definite. Adding small ridge for stability.")
    
    # Check inputs with validation function
    _mu, _Sigma, _ones, _ = _check_inputs(mu, Sigma)
    valid_inputs = True
except Exception as e:
    st.error(f"❌ Input validation failed: {e}")
    valid_inputs = False

# extra inputs for One-fund (single risky)
if opt == "One-Fund (single risky + rf)":
    st.subheader("2) One-Fund parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        mu_s = st.number_input("Risky asset expected return (μ_s)", value=0.10, step=0.005, format="%.4f")
    with c2:
        sigma_s = st.number_input("Risky asset stdev (σ_s)", value=0.20, min_value=0.0001, step=0.01, format="%.4f")
    with c3:
        A = st.number_input("Risk aversion (A)", value=3.0, min_value=0.01, step=0.5, format="%.2f")

# ------------- Run -------------
st.subheader("3) Results")
run = st.button("🚀 Compute", type="primary")

if run:
    if opt == "One-Fund (single risky + rf)":
        try:
            w_star, mu_p, sigma_p = one_fund_single_risky(mu_s, rf, sigma_s, A)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Optimal Risky Weight (w*)", f"{w_star:.4f}")
            with col2:
                st.metric("Portfolio E[R]", f"{mu_p:.4f}")
            with col3:
                st.metric("Portfolio σ", f"{sigma_p:.4f}")
            
            # Visual explanation
            investment_breakdown = pd.DataFrame({
                "Asset": ["Risky Asset", "Risk-Free Asset"],
                "Weight": [w_star, 1 - w_star],
                "Allocation %": [w_star * 100, (1 - w_star) * 100]
            })
            st.dataframe(investment_breakdown.style.format({"Weight": "{:.4f}", "Allocation %": "{:.2f}%"}))
            
            st.info(f"💡 **Interpretation:** Invest {w_star*100:.2f}% in the risky asset and {(1-w_star)*100:.2f}% in the risk-free asset.")
        except Exception as e:
            st.error(f"❌ Computation failed: {e}")

    elif opt == "Global Minimum-Variance (GMV)":
        if not valid_inputs: 
            st.error("Please fix input errors above before computing.")
            st.stop()
        try:
            with st.spinner("Computing GMV portfolio..."):
                if long_only:
                    w = gmv_long_only(Sigma)
                else:
                    w = gmv_unconstrained(Sigma)
            
            m, s, _, _ = portfolio_stats(w, _mu, _Sigma)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("GMV E[R]", f"{m:.4f}")
            with col2:
                st.metric("GMV σ", f"{s:.4f}")
            with col3:
                st.metric("Sum of Weights", f"{w.sum():.4f}")
            
            # Show weights table
            out = pd.DataFrame({"Weight": w, "Allocation %": w * 100}, index=asset_names)
            out = out.sort_values("Weight", ascending=False)
            st.write("**GMV Portfolio Weights**")
            st.dataframe(out.style.format({"Weight": "{:.4f}", "Allocation %": "{:.2f}%"}), use_container_width=True)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = plt.cm.Set3(np.linspace(0, 1, len(asset_names)))
            bars = ax.barh(asset_names, w, color=colors)
            ax.set_xlabel("Weight")
            ax.set_title("GMV Portfolio Allocation")
            ax.axvline(0, color='black', linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"❌ GMV optimization failed: {e}")

    elif opt == "Tangency / Max-Sharpe":
        if not valid_inputs: 
            st.error("Please fix input errors above before computing.")
            st.stop()
        try:
            with st.spinner("Computing Tangency portfolio..."):
                if long_only:
                    w = tangency_long_only(mu, Sigma, rf)
                else:
                    w = tangency_unconstrained(mu, Sigma, rf)
            
            m, s, _, _ = portfolio_stats(w, _mu, _Sigma)
            sharpe = (m - rf) / s if s > 0 else np.nan
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("E[R]", f"{m:.4f}")
            with col2:
                st.metric("σ", f"{s:.4f}")
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe:.4f}")
            with col4:
                st.metric("Sum of Weights", f"{w.sum():.4f}")
            
            # Show weights table
            out = pd.DataFrame({"Weight": w, "Allocation %": w * 100}, index=asset_names)
            out = out.sort_values("Weight", ascending=False)
            st.write("**Tangency Portfolio Weights**")
            st.dataframe(out.style.format({"Weight": "{:.4f}", "Allocation %": "{:.2f}%"}), use_container_width=True)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = plt.cm.Set3(np.linspace(0, 1, len(asset_names)))
            bars = ax.barh(asset_names, w, color=colors)
            ax.set_xlabel("Weight")
            ax.set_title(f"Tangency Portfolio Allocation (Sharpe = {sharpe:.4f})")
            ax.axvline(0, color='black', linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"❌ Tangency optimization failed: {e}")

    elif opt == "Efficient Frontier":
        if not valid_inputs: 
            st.error("Please fix input errors above before computing.")
            st.stop()
        try:
            with st.spinner("Computing Efficient Frontier..."):
                if long_only:
                    W, R, S = efficient_frontier_long_only(mu, Sigma, n_points=n_points)
                else:
                    W, R, S = efficient_frontier_unconstrained(mu, Sigma, n_points=n_points)

            if len(R) == 0:
                st.error("❌ Failed to compute any frontier points.")
                st.stop()

            # Show a few points
            tbl = pd.DataFrame({"E[R]": R, "σ": S})
            st.write(f"**Frontier Points Computed:** {len(R)}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**First 10 Points:**")
                st.dataframe(tbl.head(10).style.format({"E[R]":"{:.4f}", "σ":"{:.4f}"}))
            with col2:
                st.write("**Last 10 Points:**")
                st.dataframe(tbl.tail(10).style.format({"E[R]":"{:.4f}", "σ":"{:.4f}"}))

            # Plot frontier
            fig = plt.figure(figsize=(10, 6))
            plt.scatter(S, R, s=20, alpha=0.6, c=R, cmap='viridis')
            plt.colorbar(label='Expected Return')
            plt.xlabel("σ (Standard Deviation)", fontsize=12)
            plt.ylabel("E[R] (Expected Return)", fontsize=12)
            plt.title("Efficient Frontier" + (" — Long-only" if long_only else " — Unconstrained"), fontsize=14)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

            # Also mark GMV and Tangency on the plot
            try:
                with st.spinner("Adding GMV and Tangency portfolios to plot..."):
                    w_gmv = gmv_long_only(Sigma) if long_only else gmv_unconstrained(Sigma)
                    mg, sg, *_ = portfolio_stats(w_gmv, _mu, _Sigma)
                    w_tan = tangency_long_only(mu, Sigma, rf) if long_only else tangency_unconstrained(mu, Sigma, rf)
                    mt, stdev_t, *_ = portfolio_stats(w_tan, _mu, _Sigma)
                    sharpe_tan = (mt - rf) / stdev_t if stdev_t > 0 else np.nan

                    fig2 = plt.figure(figsize=(10, 6))
                    plt.scatter(S, R, s=20, alpha=0.4, c=R, cmap='viridis', label="Efficient Frontier")
                    plt.scatter([sg], [mg], marker="D", s=150, c='red', edgecolors='black', linewidths=2, label=f"GMV (σ={sg:.4f})", zorder=5)
                    plt.scatter([stdev_t], [mt], marker="*", s=300, c='gold', edgecolors='black', linewidths=2, label=f"Tangency (Sharpe={sharpe_tan:.4f})", zorder=5)
                    
                    # Draw capital allocation line
                    if stdev_t > 0:
                        x_cal = np.array([0, stdev_t * 1.5])
                        y_cal = rf + (mt - rf) / stdev_t * x_cal
                        plt.plot(x_cal, y_cal, 'k--', alpha=0.5, label='Capital Allocation Line')
                    
                    plt.xlabel("σ (Standard Deviation)", fontsize=12)
                    plt.ylabel("E[R] (Expected Return)", fontsize=12)
                    plt.legend(loc='best')
                    plt.title("Efficient Frontier with Key Portfolios", fontsize=14)
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    
                    # Summary table
                    summary = pd.DataFrame({
                        "Portfolio": ["GMV", "Tangency"],
                        "E[R]": [mg, mt],
                        "σ": [sg, stdev_t],
                        "Sharpe": [np.nan, sharpe_tan]
                    })
                    st.write("**Key Portfolio Metrics:**")
                    st.dataframe(summary.style.format({"E[R]": "{:.4f}", "σ": "{:.4f}", "Sharpe": "{:.4f}"}))
                    
            except Exception as e:
                st.warning(f"⚠️ Could not compute GMV/Tangency for plot overlay: {e}")

        except Exception as e:
            st.error(f"❌ Frontier computation failed: {e}")

st.divider()
st.caption("Built with Streamlit • Portfolio optimization using Modern Portfolio Theory")
