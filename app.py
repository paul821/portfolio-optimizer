# app.py — Portfolio Optimizer (One-Fund, GMV, Tangency, Frontier) - FINAL VERSION
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
    if not np.allclose(Sigma, Sigma.T, atol=1e-10):
        Sigma = (Sigma + Sigma.T) / 2
    Sigma = Sigma + ridge * np.eye(n)
    ones = np.ones((n, 1))
    return mu, Sigma, ones, n

def portfolio_stats(w, mu, Sigma):
    w = _as_col(w)
    m = (w.T @ mu).item()
    v = (w.T @ Sigma @ w).item()
    s = np.sqrt(max(v, 0))
    return m, s, v, (m / s if s > 0 else np.nan)

# ------------- optimizers -------------
def one_fund_single_risky(mu_s, rf, sigma_s, A):
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
    w /= (ones.T @ w).item()
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
        res = minimize(obj, z, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": maxiter, "ftol": 1e-9})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("GMV (long-only) failed to converge.")
    return best.x

def tangency_unconstrained(mu, Sigma, rf):
    mu, Sigma, ones, n = _check_inputs(mu, Sigma)
    excess = mu - rf * ones
    w_unnorm = np.linalg.solve(Sigma, excess)
    w = w_unnorm / (ones.T @ w_unnorm).item()
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
        res = minimize(neg_sharpe, z, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": maxiter, "ftol": 1e-9})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("Tangency (long-only) failed to converge.")
    return best.x

def min_var_for_target_return_unconstrained(mu, Sigma, target):
    mu, Sigma, ones, n = _check_inputs(mu, Sigma)
    inv = np.linalg.inv(Sigma)
    A = (ones.T @ inv @ ones).item()
    B = (ones.T @ inv @ mu).item()
    C = (mu.T @ inv @ mu).item()
    D = A * C - B**2
    if abs(D) < 1e-12:
        raise ValueError("Degenerate covariance structure (D ≈ 0)")
    lam = (C - B * target) / D
    gamma = (A * target - B) / D
    w = lam * (inv @ ones) + gamma * (inv @ mu)
    return w.flatten()

def efficient_frontier_unconstrained(mu, Sigma, n_points=60, ret_range=None):
    mu, Sigma, ones, _ = _check_inputs(mu, Sigma)
    mu_flat = mu.flatten()
    lo, hi = (mu_flat.min() - 1e-6, mu_flat.max() + 1e-6) if ret_range is None else ret_range
    targets = np.linspace(lo, hi, n_points)
    W, R, S = [], [], []
    for t in targets:
        try:
            w = min_var_for_target_return_unconstrained(mu, Sigma, t)
            m, s, _, _ = portfolio_stats(w, mu, Sigma)
            W.append(w); R.append(m); S.append(s)
        except Exception:
            continue
    return np.array(W), np.array(R), np.array(S)

def efficient_frontier_long_only(mu, Sigma, n_points=60, theta_max_scale=50.0):
    mu, Sigma, ones, n = _check_inputs(mu, Sigma)
    mu_flat = mu.flatten()
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    scale = max(1e-3, np.abs(mu_flat).mean())
    thetas = np.linspace(0.0, theta_max_scale * scale, n_points)
    W, R, S = [], [], []
    rng = np.random.default_rng(123)
    w_start = np.ones(n) / n
    for theta in thetas:
        def obj(w, th=theta): return float(w @ Sigma @ w - th * (w @ mu_flat))
        best = None
        for z in [w_start, rng.random(n)]:
            z = z / z.sum()
            res = minimize(obj, z, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 800, "ftol": 1e-9})
            if res.success and (best is None or res.fun < best.fun):
                best = res
        w = best.x if best else w_start
        w_start = w
        m, s, _, _ = portfolio_stats(w, mu, Sigma)
        W.append(w); R.append(m); S.append(s)
    return np.array(W), np.array(R), np.array(S)

# ------------- UI HELPERS -------------
def delete_assets(assets_to_delete):
    if not assets_to_delete: return
    current_names = st.session_state.asset_names
    indices_to_delete = [current_names.index(name) for name in assets_to_delete]
    for key in ['asset_names', 'mu_values', 'risk_measures']:
        if key in st.session_state:
            st.session_state[key] = [v for i, v in enumerate(st.session_state[key]) if i not in indices_to_delete]
    for key in ['cov_matrix', 'corr_matrix']:
        if key in st.session_state:
            mat = np.delete(st.session_state[key], indices_to_delete, axis=0)
            mat = np.delete(mat, indices_to_delete, axis=1)
            st.session_state[key] = mat
    st.toast(f"Deleted assets: {', '.join(assets_to_delete)}", icon="🗑️")

def sync_asset_data(new_n_assets):
    current_n = len(st.session_state.get('asset_names', []))
    if new_n_assets == current_n: return
    if new_n_assets > current_n:
        for i in range(current_n, new_n_assets):
            st.session_state.asset_names.append(f"Asset_{chr(65+i)}")
            st.session_state.mu_values.append(0.08)
            is_variance = np.mean(st.session_state.get('risk_measures', [0.1])) < 0.1
            st.session_state.risk_measures.append(0.01 if is_variance else 0.1)
            for key, default_val, is_corr in [('cov_matrix', 0.01, False), ('corr_matrix', 1.0, True)]:
                old_mat = st.session_state[key]
                new_mat = np.zeros((i + 1, i + 1)) if not is_corr else np.eye(i + 1)
                new_mat[:i, :i] = old_mat
                if not is_corr: new_mat[i, i] = default_val
                st.session_state[key] = new_mat
    else:
        for key in ['asset_names', 'mu_values', 'risk_measures']:
            st.session_state[key] = st.session_state[key][:new_n_assets]
        for key in ['cov_matrix', 'corr_matrix']:
            st.session_state[key] = st.session_state[key][:new_n_assets, :new_n_assets]
    st.rerun()

# ------------- UI -------------
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("📈 Portfolio Optimizer")
st.caption("A tool for Modern Portfolio Theory analysis, including GMV, Tangency, and Efficient Frontier.")

if 'app_initialized' not in st.session_state:
    st.session_state.asset_names = ["Asset_A", "Asset_B", "Asset_C", "Asset_D"]
    st.session_state.mu_values = [0.08, 0.10, 0.06, 0.12]
    default_cov = np.array([[0.01, 0.003, 0.0008, 0.0045], [0.003, 0.0225, 0.0018, 0.0081], [0.0008, 0.0018, 0.0064, 0.00072], [0.0045, 0.0081, 0.00072, 0.0324]])
    st.session_state.cov_matrix = default_cov
    st.session_state.risk_measures = np.sqrt(np.diag(default_cov)).tolist()
    D_inv = np.diag(1.0 / np.sqrt(np.diag(default_cov)))
    st.session_state.corr_matrix = np.clip(D_inv @ default_cov @ D_inv, -1, 1)
    st.session_state.app_initialized = True

with st.sidebar:
    st.header("⚙️ Options")
    opt = st.selectbox("Optimization type", ["One-Fund (single risky + rf)", "Global Minimum-Variance (GMV)", "Tangency / Max-Sharpe", "Efficient Frontier"])
    long_only = st.checkbox("Enforce no short sales (long-only)", value=True, help="Applies to GMV, Tangency, and Frontier.")
    rf = st.number_input("Risk-free rate (rf)", value=0.03, step=0.005, format="%.4f")
    n_points = st.slider("Frontier points / sweep resolution", 20, 200, 80)

st.subheader("1) Define Assets & Inputs")
col_a, col_b = st.columns([1, 1])
with col_a:
    n_assets_input = st.number_input("Number of assets", min_value=1, max_value=50, value=len(st.session_state.asset_names), step=1, help="Change to add or remove assets.")
    if n_assets_input != len(st.session_state.asset_names):
        sync_asset_data(n_assets_input)
    n_assets = len(st.session_state.asset_names)
    names_df = pd.DataFrame({"Asset": st.session_state.asset_names})
    edited_names = st.data_editor(names_df, use_container_width=True, hide_index=True)
    st.session_state.asset_names = [str(x).strip() for x in edited_names["Asset"].tolist()]
    if any(not name for name in st.session_state.asset_names) or len(st.session_state.asset_names) != len(set(st.session_state.asset_names)):
        st.error("⚠️ Asset names must be unique and non-empty!")
        st.stop()
    with st.expander("🗑️ Delete Specific Assets"):
        assets_to_delete = st.multiselect("Select assets to remove:", options=st.session_state.asset_names)
        if st.button("Delete Selected Assets", disabled=not assets_to_delete):
            delete_assets(assets_to_delete)
            st.rerun()
with col_b:
    mu_df = pd.DataFrame({"ExpectedReturn": st.session_state.mu_values}, index=st.session_state.asset_names)
    mu_edited = st.data_editor(mu_df, use_container_width=True)
    st.session_state.mu_values = mu_edited["ExpectedReturn"].tolist()

st.markdown("#### Risk Input Method")
col_risk1, col_risk2 = st.columns(2)
with col_risk1:
    risk_metric = st.radio("Risk Metric:", ["Standard Deviation (σ)", "Variance (σ²)"])
with col_risk2:
    matrix_type = st.radio("Matrix Type:", ["Correlation Matrix", "Covariance Matrix"])

use_variance = (risk_metric == "Variance (σ²)")
use_correlation = (matrix_type == "Correlation Matrix")

if use_correlation:
    st.markdown(f"#### {'Standard Deviations' if not use_variance else 'Variances'}")
    risk_label = "Variance" if use_variance else "StandardDeviation"
    risk_df = pd.DataFrame({risk_label: st.session_state.risk_measures}, index=st.session_state.asset_names)
    edited_risk = st.data_editor(risk_df, use_container_width=True)
    st.session_state.risk_measures = edited_risk[risk_label].tolist()
    risk_array = np.array(st.session_state.risk_measures)
    if np.any(risk_array <= 0):
        st.error("All risk measures must be positive!")
        st.stop()
    std_array = np.sqrt(risk_array) if use_variance else risk_array
    st.markdown("#### Correlation Matrix")
    corr_display = st.session_state.corr_matrix.copy()
    corr_display[np.triu_indices(n_assets, 1)] = np.nan
    corr_df = pd.DataFrame(corr_display, index=st.session_state.asset_names, columns=st.session_state.asset_names)
    st.info("💡 Only fill the lower triangle; the matrix is symmetric.")
    edited_corr = st.data_editor(corr_df, use_container_width=True)
    corr_matrix_edited = edited_corr.to_numpy(dtype=float, na_value=0)
    corr_matrix_edited = (corr_matrix_edited + corr_matrix_edited.T) - np.diag(np.diag(corr_matrix_edited))
    np.fill_diagonal(corr_matrix_edited, 1.0)
    st.session_state.corr_matrix = corr_matrix_edited
    Sigma = np.diag(std_array) @ corr_matrix_edited @ np.diag(std_array)
else:
    st.markdown("#### Covariance Matrix")
    cov_display = st.session_state.cov_matrix.copy()
    cov_display[np.triu_indices(n_assets, 1)] = np.nan
    matrix_df = pd.DataFrame(cov_display, index=st.session_state.asset_names, columns=st.session_state.asset_names)
    st.info("💡 Only fill the lower triangle; the matrix is symmetric.")
    edited_matrix = st.data_editor(matrix_df, use_container_width=True)
    cov_matrix_edited = edited_matrix.to_numpy(dtype=float, na_value=0)
    cov_matrix_edited = (cov_matrix_edited + cov_matrix_edited.T) - np.diag(np.diag(cov_matrix_edited))
    st.session_state.cov_matrix = cov_matrix_edited
    Sigma = cov_matrix_edited

try:
    mu = np.array(st.session_state.mu_values)
    asset_names = st.session_state.asset_names
    if len(mu) != n_assets or Sigma.shape != (n_assets, n_assets):
        st.warning("Data is out of sync. Please refresh the page.")
        st.stop()
    eigenvalues = np.linalg.eigvalsh(Sigma)
    if np.any(eigenvalues < -1e-8):
        st.warning("⚠️ Covariance matrix is not positive semi-definite. Results may be unstable.")
    with st.expander("📊 View Input Summary"):
        summary_stats = pd.DataFrame({"E[R]": mu, "Std Dev (σ)": np.sqrt(np.diag(Sigma))}, index=asset_names)
        st.dataframe(summary_stats.style.format("{:.4f}"))
        st.write("**Full Covariance Matrix:**")
        st.dataframe(pd.DataFrame(Sigma, index=asset_names, columns=asset_names).style.format("{:.6f}"))
        if use_correlation:
            st.write("**Full Correlation Matrix:**")
            st.dataframe(pd.DataFrame(st.session_state.corr_matrix, index=asset_names, columns=asset_names).style.format("{:.3f}").background_gradient(cmap='RdYlGn', vmin=-1, vmax=1))
    _mu, _Sigma, _, _ = _check_inputs(mu, Sigma)
    valid_inputs = True
except Exception as e:
    st.error(f"❌ Input validation failed: {e}")
    valid_inputs = False

if opt == "One-Fund (single risky + rf)":
    st.subheader("2) One-Fund Parameters")
    c1, c2, c3 = st.columns(3)
    with c1: mu_s = st.number_input("Risky asset E[R] (μₛ)", value=0.10, step=0.005, format="%.4f")
    with c2: sigma_s = st.number_input("Risky asset σₛ", value=0.20, min_value=1e-4, step=0.01, format="%.4f")
    with c3: A = st.number_input("Risk aversion (A)", value=3.0, min_value=0.01, step=0.5, format="%.2f")

st.subheader("3) Results")
if st.button("🚀 Compute", type="primary"):
    if opt == "One-Fund (single risky + rf)":
        try:
            w_star, mu_p, sigma_p = one_fund_single_risky(mu_s, rf, sigma_s, A)
            sharpe_p = (mu_p - rf) / sigma_p if sigma_p > 0 else np.nan
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Optimal Risky Weight (w*)", f"{w_star:.4f}")
            c2.metric("Portfolio E[R]", f"{mu_p:.4f}")
            c3.metric("Portfolio σ", f"{sigma_p:.4f}")
            c4.metric("Sharpe Ratio", f"{sharpe_p:.4f}")
            if w_star > 1: st.info(f"💡 **Interpretation:** You are levered. Borrow {(w_star-1)*100:.2f}% at the risk-free rate to invest a total of {w_star*100:.2f}% in the risky asset.")
            elif w_star < 0: st.info(f"💡 **Interpretation:** You are shorting the risky asset. Short {abs(w_star)*100:.2f}% of the risky asset and invest a total of {(1-w_star)*100:.2f}% in the risk-free asset.")
            else: st.info(f"💡 **Interpretation:** Invest {w_star*100:.2f}% in the risky asset and the remaining {(1-w_star)*100:.2f}% in the risk-free asset.")
        except Exception as e: st.error(f"❌ Computation failed: {e}")
    elif not valid_inputs:
        st.error("Please fix input errors above before computing.")
    elif opt == "Global Minimum-Variance (GMV)":
        try:
            with st.spinner("Computing GMV..."):
                w = gmv_long_only(Sigma) if long_only else gmv_unconstrained(Sigma)
            m, s, _, _ = portfolio_stats(w, _mu, _Sigma)
            sharpe_gmv = (m - rf) / s if s > 0 else np.nan
            c1, c2, c3 = st.columns(3); c1.metric("GMV E[R]", f"{m:.4f}"); c2.metric("GMV σ", f"{s:.4f}"); c3.metric("Sharpe Ratio", f"{sharpe_gmv:.4f}")
            out = pd.DataFrame({"Weight": w}, index=asset_names).sort_values("Weight", ascending=False)
            st.write("**GMV Portfolio Weights**"); st.dataframe(out.style.format({"Weight": "{:.2%}"}), use_container_width=True)
            fig, ax = plt.subplots(figsize=(8, max(4, n_assets * 0.5))); colors = plt.cm.viridis(np.linspace(0, 1, n_assets))
            out.sort_values("Weight").plot(kind='barh', ax=ax, legend=False, color=colors); ax.set_xlabel("Weight"); ax.set_ylabel("Asset"); ax.set_title("GMV Portfolio Allocation"); ax.axvline(0, color='black', linewidth=0.8); plt.tight_layout(); st.pyplot(fig)
        except Exception as e: st.error(f"❌ GMV optimization failed: {e}")
    elif opt == "Tangency / Max-Sharpe":
        try:
            with st.spinner("Computing Tangency..."):
                w = tangency_long_only(mu, Sigma, rf) if long_only else tangency_unconstrained(mu, Sigma, rf)
            m, s, _, sharpe = portfolio_stats(w, _mu, _Sigma); sharpe = (m - rf) / s if s > 0 else np.nan
            c1, c2, c3 = st.columns(3); c1.metric("Tangency E[R]", f"{m:.4f}"); c2.metric("Tangency σ", f"{s:.4f}"); c3.metric("Max Sharpe Ratio", f"{sharpe:.4f}")
            out = pd.DataFrame({"Weight": w}, index=asset_names).sort_values("Weight", ascending=False)
            st.write("**Tangency Portfolio Weights**"); st.dataframe(out.style.format({"Weight": "{:.2%}"}), use_container_width=True)
            fig, ax = plt.subplots(figsize=(8, max(4, n_assets * 0.5))); colors = plt.cm.viridis(np.linspace(0, 1, n_assets))
            out.sort_values("Weight").plot(kind='barh', ax=ax, legend=False, color=colors); ax.set_xlabel("Weight"); ax.set_ylabel("Asset"); ax.set_title(f"Tangency Portfolio Allocation (Sharpe = {sharpe:.3f})"); ax.axvline(0, color='black', linewidth=0.8); plt.tight_layout(); st.pyplot(fig)
        except Exception as e: st.error(f"❌ Tangency optimization failed: {e}")
    elif opt == "Efficient Frontier":
        try:
            with st.spinner("Computing Efficient Frontier..."):
                W, R, S = efficient_frontier_long_only(mu, Sigma, n_points) if long_only else efficient_frontier_unconstrained(mu, Sigma, n_points)
                if len(R) == 0: raise RuntimeError("Failed to compute any frontier points.")
                w_gmv = gmv_long_only(Sigma) if long_only else gmv_unconstrained(Sigma)
                mg, sg, _, _ = portfolio_stats(w_gmv, _mu, _Sigma)
                w_tan = tangency_long_only(mu, Sigma, rf) if long_only else tangency_unconstrained(mu, Sigma, rf)
                mt, s_tan, _, sharpe_tan = portfolio_stats(w_tan, _mu, _Sigma)
                sharpe_tan = (mt - rf) / s_tan if s_tan > 0 else np.nan
            fig, ax = plt.subplots(figsize=(10, 6))
            frontier_sharpe = (R - rf) / S
            sc = ax.scatter(S, R, s=20, c=frontier_sharpe, cmap='viridis', label="Efficient Frontier Points")
            plt.colorbar(sc, label='Sharpe Ratio')
            ax.scatter(sg, mg, marker="D", s=150, c='red', ec='k', lw=1.5, label=f"GMV (σ={sg:.3f})", zorder=5)
            ax.scatter(s_tan, mt, marker="*", s=300, c='gold', ec='k', lw=1.5, label=f"Tangency (Sharpe={sharpe_tan:.3f})", zorder=5)
            ind_s, ind_r = np.sqrt(np.diag(_Sigma)), _mu.flatten()
            ax.scatter(ind_s, ind_r, marker='o', s=100, c='grey', ec='k', label='Individual Assets', zorder=4)
            for i, name in enumerate(asset_names):
                ax.annotate(name, (ind_s[i], ind_r[i]), xytext=(5, -5), textcoords='offset points')
            if s_tan > 0:
                x_cal = np.array([0, s_tan * 1.5]); y_cal = rf + sharpe_tan * x_cal
                ax.plot(x_cal, y_cal, 'k--', alpha=0.7, label='Capital Allocation Line')
            ax.set_xlabel("σ (Standard Deviation)", fontsize=12); ax.set_ylabel("E[R] (Expected Return)", fontsize=12); ax.legend(loc='best'); ax.set_title("Efficient Frontier & Key Portfolios", fontsize=14); ax.grid(alpha=0.4); st.pyplot(fig)
        except Exception as e: st.error(f"❌ Frontier computation failed: {e}")

st.divider()
st.caption("Built with Streamlit • Modern Portfolio Theory")
