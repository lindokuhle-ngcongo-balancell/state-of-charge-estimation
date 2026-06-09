"""
Battery Charge Estimator using Unscented Kalman Filter (UKF)
=============================================================
Fuses V, I, T, and q_transfer measurements to estimate total charge removed (Q_total).
Robust to coulomb counter (q_transfer) errors/drift.

Expected CSV columns (order-insensitive, case-insensitive):
    time        - timestamp or minute index (optional)
    voltage     - terminal voltage (V), instantaneous at end of minute
    current     - current (A), instantaneous at end of minute (positive = discharge)
    temperature - cell temperature (°C), instantaneous at end of minute
    q_transfer  - charge transferred in that minute (Ah or C), from coulomb counter

Usage:
    python battery_charge_estimator.py --csv data.csv [OPTIONS]

Options:
    --csv           Path to input CSV file (required)
    --capacity      Nominal battery capacity in Ah (default: 100)
    --q_noise       Process noise for Q_total (default: 0.05) — increase if battery
                    behaviour is very dynamic
    --cc_noise      Measurement noise for coulomb counter (default: 0.5) — increase
                    to trust the coulomb counter LESS (more robust to CC errors)
    --v_noise       Measurement noise for voltage (default: 0.02)
    --i_noise       Measurement noise for current (default: 0.1)
    --t_noise       Measurement noise for temperature (default: 1.0)
    --output        Path to save results CSV (default: results.csv)
    --plot          Show plots after processing (flag, no value needed)

Tuning guide:
    - If your CC drifts slowly:     increase --cc_noise (e.g. 1.0–5.0)
    - If your CC has random spikes: increase --cc_noise significantly (e.g. 5.0–20.0)
    - If V/I/T are very noisy:      increase --v_noise, --i_noise, --t_noise
    - If state changes fast:        increase --q_noise
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# UKF Implementation (no external dependencies beyond numpy)
# ---------------------------------------------------------------------------

class UnscentedKalmanFilter:
    """
    Minimal UKF for state estimation.

    State vector:  x = [Q_total, I_bias, V_oc_offset]
        Q_total     : total charge removed from battery (Ah)
        I_bias      : slow drift in current sensor (A) — helps absorb CC errors
        V_oc_offset : slow drift in OCV model offset (V) — handles ageing / temp shift

    Measurements: z = [q_cc, V, I, T]  (each step)
    """

    def __init__(self, n_state=3, alpha=1e-3, beta=2.0, kappa=0.0):
        self.n = n_state
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        lam = alpha**2 * (n_state + kappa) - n_state
        self.lam = lam

        # Sigma-point weights
        self.Wm = np.full(2 * n_state + 1, 1.0 / (2 * (n_state + lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam / (n_state + lam)
        self.Wc[0] = lam / (n_state + lam) + (1 - alpha**2 + beta)

        self.sqrt_factor = np.sqrt(n_state + lam)

    def sigma_points(self, x, P):
        n = self.n
        try:
            S = np.linalg.cholesky(P) * self.sqrt_factor
        except np.linalg.LinAlgError:
            # Fallback: perturb P to be positive definite
            P = P + np.eye(n) * 1e-6
            S = np.linalg.cholesky(P) * self.sqrt_factor

        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = x
        for i in range(n):
            sigmas[i + 1]     = x + S[:, i]
            sigmas[n + i + 1] = x - S[:, i]
        return sigmas

    def predict(self, x, P, f, Q):
        sigmas = self.sigma_points(x, P)
        sigmas_f = np.array([f(s) for s in sigmas])

        x_pred = np.dot(self.Wm, sigmas_f)
        P_pred = Q.copy()
        for i, sig in enumerate(sigmas_f):
            d = sig - x_pred
            P_pred += self.Wc[i] * np.outer(d, d)

        return x_pred, P_pred, sigmas_f

    def update(self, x_pred, P_pred, sigmas_f, h, z, R, mask=None):
        """
        mask: boolean array len(z). False = this measurement is an outlier, skip it.
        """
        sigmas_h = np.array([h(s) for s in sigmas_f])
        z_pred = np.dot(self.Wm, sigmas_h)

        S = R.copy()
        Pxz = np.zeros((self.n, len(z)))
        for i, sig in enumerate(sigmas_h):
            dz = sig - z_pred
            S += self.Wc[i] * np.outer(dz, dz)
            dx = sigmas_f[i] - x_pred
            Pxz += self.Wc[i] * np.outer(dx, dz)

        # If mask provided, zero out columns for outlier measurements
        if mask is not None:
            for j, valid in enumerate(mask):
                if not valid:
                    S[:, j] = 0
                    S[j, :] = 0
                    S[j, j] = 1e9   # large variance = ignore
                    Pxz[:, j] = 0

        K = Pxz @ np.linalg.inv(S)
        innov = z - z_pred
        if mask is not None:
            innov[~mask] = 0.0

        x_new = x_pred + K @ innov
        P_new = P_pred - K @ S @ K.T
        # Symmetrise
        P_new = 0.5 * (P_new + P_new.T)

        return x_new, P_new, innov


# ---------------------------------------------------------------------------
# Battery OCV model  (simple Shepherd-like polynomial)
# Returns Open Circuit Voltage given State-of-Charge (0–1)
# ---------------------------------------------------------------------------

def ocv_from_soc(soc: np.ndarray, V_min=3.0, V_max=4.2) -> np.ndarray:
    """
    Generic OCV-SOC curve. Replace with your actual chemistry curve for best results.
    Uses a smooth polynomial approximation valid for Li-ion / LFP.
    """
    soc = np.clip(soc, 0.0, 1.0)
    # Coefficients tuned for a generic Li-ion cell (3.0–4.2 V)
    # Adjust V_min / V_max for your chemistry.
    V_range = V_max - V_min
    ocv = (V_min
           + V_range * (1.82 * soc**5
                        - 4.90 * soc**4
                        + 4.51 * soc**3
                        - 1.72 * soc**2
                        + 0.90 * soc
                        + 0.39))
    return np.clip(ocv, V_min, V_max)


# ---------------------------------------------------------------------------
# Helper: robust outlier detection for the coulomb counter
# ---------------------------------------------------------------------------

def is_cc_outlier(q_cc: float, q_est: float, I: float,
                  threshold_sigma: float = 4.0,
                  cc_sigma: float = 0.5) -> bool:
    """
    Mark a CC reading as outlier if residual > threshold_sigma * cc_sigma.
    q_est is the filter's predicted charge step (I * dt as fallback).
    """
    residual = abs(q_cc - q_est)
    return residual > threshold_sigma * cc_sigma


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

def run_estimator(df: pd.DataFrame,
                  capacity_Ah: float = 100.0,
                  q_process_noise: float = 0.05,
                  cc_meas_noise: float = 0.5,
                  v_meas_noise: float = 0.02,
                  i_meas_noise: float = 0.1,
                  t_meas_noise: float = 1.0,
                  dt_minutes: float = 1.0) -> pd.DataFrame:
    """
    Run UKF over the dataframe. Returns df with extra columns:
        Q_estimated   - filter's best estimate of total charge removed (Ah)
        SOC_estimated - state of charge (0–1)
        I_bias        - estimated current sensor bias (A)
        P_Q           - 1-sigma uncertainty on Q_estimated
        cc_outlier    - True if that CC reading was flagged as suspect
        innovation_q  - residual between CC reading and filter prediction
    """
    dt = dt_minutes / 60.0  # convert to hours (for Ah)

    # -----------------------------------------------------------------------
    # State transition: f(x) — each state evolves over one minute
    # -----------------------------------------------------------------------
    def f(x):
        Q, I_bias, V_off = x
        # Q increases by I*dt (using current + estimated bias)
        # I_bias and V_off are random-walk states (persist slowly)
        I_meas = df_I[step]
        Q_new  = Q + (I_meas + I_bias) * dt
        I_bias_new = I_bias  # slow drift
        V_off_new  = V_off   # slow drift
        return np.array([Q_new, I_bias_new, V_off_new])

    # -----------------------------------------------------------------------
    # Measurement model: h(x) → [q_cc, V_terminal, I, T]
    # -----------------------------------------------------------------------
    def h(x):
        Q, I_bias, V_off = x
        soc  = np.clip(1.0 - Q / capacity_Ah, 0.0, 1.0)
        I_now = df_I[step]
        T_now = df_T[step]

        # Predicted CC reading = charge step from state
        q_cc_pred = (I_now + I_bias) * dt

        # Predicted terminal voltage using OCV - I*R_internal
        # R_int varies with SOC and temperature (simple empirical model)
        R_int = 0.05 * (1 + 0.01 * (25 - T_now)) * (1 + 0.5 * (1 - soc)**2)
        V_pred = ocv_from_soc(soc) + V_off - I_now * R_int

        # Predicted current (we observe I directly, but I_bias shifts the model)
        I_pred = I_now + I_bias

        # Temperature: weakly tied to SOC and discharge rate (simplified)
        T_pred = T_now  # treat as direct observation

        return np.array([q_cc_pred, V_pred, I_pred, T_pred])

    # -----------------------------------------------------------------------
    # Noise matrices
    # -----------------------------------------------------------------------
    # Process noise Q_mat
    Q_mat = np.diag([q_process_noise**2,
                     (q_process_noise * 0.01)**2,   # I_bias drifts slowly
                     (q_process_noise * 0.005)**2])  # V_off drifts very slowly

    # Measurement noise R_mat: [q_cc, V, I, T]
    R_mat = np.diag([cc_meas_noise**2,
                     v_meas_noise**2,
                     i_meas_noise**2,
                     t_meas_noise**2])

    # -----------------------------------------------------------------------
    # Initialise filter
    # -----------------------------------------------------------------------
    ukf = UnscentedKalmanFilter(n_state=3)

    # Initial Q estimate: integrate first few steps of CC as a warm start
    Q0 = float(np.abs(df['q_transfer'].iloc[:5].sum()))
    x = np.array([Q0, 0.0, 0.0])
    P = np.diag([5.0**2, 0.1**2, 0.05**2])

    # Store results
    results = []
    df_I = df['current'].values
    df_T = df['temperature'].values

    for step in range(len(df)):
        row = df.iloc[step]

        # --- Predict ---
        x_pred, P_pred, sigmas_f = ukf.predict(x, P, f, Q_mat)

        # --- Outlier check on CC reading ---
        q_cc_expected = (df_I[step] + x[1]) * dt
        outlier = is_cc_outlier(row['q_transfer'], q_cc_expected,
                                df_I[step], threshold_sigma=4.0,
                                cc_sigma=cc_meas_noise)

        z = np.array([row['q_transfer'],
                      row['voltage'],
                      row['current'],
                      row['temperature']])

        # Mask: if CC outlier, down-weight CC measurement
        mask = np.array([not outlier, True, True, True])

        # --- Update ---
        x, P, innov = ukf.update(x_pred, P_pred, sigmas_f, h, z, R_mat, mask=mask)

        # Clamp Q to physical limits
        x[0] = np.clip(x[0], 0.0, capacity_Ah * 1.05)

        soc = max(0.0, 1.0 - x[0] / capacity_Ah)

        results.append({
            'Q_estimated':   x[0],
            'SOC_estimated': soc,
            'I_bias':        x[1],
            'V_oc_offset':   x[2],
            'P_Q':           np.sqrt(max(P[0, 0], 0.0)),
            'cc_outlier':    outlier,
            'innovation_q':  innov[0],
        })

    return df.assign(**{k: [r[k] for r in results] for k in results[0]})


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(df: pd.DataFrame):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Battery Charge Estimator — UKF Results", fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    time_axis = df['time'] if 'time' in df.columns else np.arange(len(df))

    # 1. Q_estimated vs cumulative CC
    ax1 = fig.add_subplot(gs[0, :])
    q_cumulative = df['q_transfer'].cumsum().abs()
    ax1.fill_between(time_axis,
                     df['Q_estimated'] - df['P_Q'],
                     df['Q_estimated'] + df['P_Q'],
                     alpha=0.25, color='steelblue', label='±1σ uncertainty')
    ax1.plot(time_axis, df['Q_estimated'], color='steelblue', lw=2, label='UKF Q estimate')
    ax1.plot(time_axis, q_cumulative, color='tomato', lw=1.5,
             linestyle='--', label='Cumulative CC (raw)')
    if df['cc_outlier'].any():
        outlier_idx = df['cc_outlier']
        ax1.scatter(time_axis[outlier_idx], q_cumulative[outlier_idx],
                    color='red', zorder=5, s=40, label='CC outlier flagged')
    ax1.set_ylabel("Charge Removed (Ah)")
    ax1.set_title("Total Charge Removed — UKF vs Raw Coulomb Counter")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. SOC
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_axis, df['SOC_estimated'] * 100, color='mediumseagreen', lw=2)
    ax2.set_ylabel("SOC (%)")
    ax2.set_title("Estimated State of Charge")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    # 3. Innovation (residual)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time_axis, df['innovation_q'], color='darkorange', lw=1.5)
    ax3.axhline(0, color='black', lw=0.8, linestyle='--')
    ax3.set_ylabel("Innovation (Ah)")
    ax3.set_title("CC Measurement Innovation (residual)")
    ax3.grid(True, alpha=0.3)

    # 4. Voltage
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(time_axis, df['voltage'], color='purple', lw=1.5)
    ax4.set_ylabel("Voltage (V)")
    ax4.set_title("Terminal Voltage")
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel("Time")

    # 5. Current + I_bias
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(time_axis, df['current'], color='slategray', lw=1.5, label='Measured I')
    ax5.plot(time_axis, df['I_bias'], color='crimson', lw=1.5,
             linestyle='--', label='Estimated I bias')
    ax5.set_ylabel("Current (A)")
    ax5.set_title("Current & Estimated Sensor Bias")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlabel("Time")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CSV loader — flexible column name matching
# ---------------------------------------------------------------------------

COLUMN_ALIASES = {
    'voltage':     ['voltage', 'v', 'volt', 'volts', 'terminal_voltage'],
    'current':     ['current', 'i', 'amps', 'ampere', 'current_a'],
    'temperature': ['temperature', 'temp', 't', 'temp_c', 'temperature_c'],
    'q_transfer':  ['q_transfer', 'charge', 'q', 'q_min', 'charge_transfer',
                    'coulombs', 'delta_q', 'dq', 'q_coulombs'],
    'time':        ['time', 't', 'timestamp', 'minute', 'minutes', 'index', 'step'],
}


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break

    df = df.rename(columns=rename_map)

    required = ['voltage', 'current', 'temperature', 'q_transfer']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n[ERROR] Missing required columns: {missing}")
        print(f"  Found columns: {list(df.columns)}")
        print(f"  Expected one of the following names for each required field:")
        for req in missing:
            print(f"    '{req}' — accepted names: {COLUMN_ALIASES[req]}")
        sys.exit(1)

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Battery charge estimator using UKF — robust to coulomb counter errors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--csv',      required=True,           help='Input CSV file path')
    parser.add_argument('--capacity', type=float, default=100, help='Nominal capacity (Ah)')
    parser.add_argument('--q_noise',  type=float, default=0.05,help='Process noise for Q (Ah)')
    parser.add_argument('--cc_noise', type=float, default=0.5, help='CC measurement noise (Ah) — increase to trust CC less')
    parser.add_argument('--v_noise',  type=float, default=0.02,help='Voltage measurement noise (V)')
    parser.add_argument('--i_noise',  type=float, default=0.1, help='Current measurement noise (A)')
    parser.add_argument('--t_noise',  type=float, default=1.0, help='Temperature measurement noise (°C)')
    parser.add_argument('--output',   default='results.csv',   help='Output CSV path')
    parser.add_argument('--plot',     action='store_true',     help='Show plots')
    args = parser.parse_args()

    # --- Load ---
    print(f"Loading: {args.csv}")
    df = load_csv(args.csv)
    print(f"  Loaded {len(df)} rows.")
    print(f"  Columns: {list(df.columns)}")

    # --- Run ---
    print(f"\nRunning UKF (capacity={args.capacity} Ah, cc_noise={args.cc_noise} Ah) ...")
    result = run_estimator(
        df,
        capacity_Ah=args.capacity,
        q_process_noise=args.q_noise,
        cc_meas_noise=args.cc_noise,
        v_meas_noise=args.v_noise,
        i_meas_noise=args.i_noise,
        t_meas_noise=args.t_noise,
    )

    # --- Summary ---
    n_outliers = result['cc_outlier'].sum()
    final_Q    = result['Q_estimated'].iloc[-1]
    final_soc  = result['SOC_estimated'].iloc[-1] * 100
    raw_Q      = result['q_transfer'].abs().sum()

    print(f"\n--- Results Summary ---")
    print(f"  CC outlier steps flagged : {n_outliers} / {len(result)}")
    print(f"  Raw CC total (cumsum)    : {raw_Q:.3f} Ah")
    print(f"  UKF Q estimate (final)   : {final_Q:.3f} Ah  ±{result['P_Q'].iloc[-1]:.3f} Ah (1σ)")
    print(f"  Estimated final SOC      : {final_soc:.1f}%")
    print(f"  Max |I_bias| detected    : {result['I_bias'].abs().max():.4f} A")

    # --- Save ---
    result.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")

    # --- Plot ---
    if args.plot:
        plot_results(result)


if __name__ == '__main__':
    main()
