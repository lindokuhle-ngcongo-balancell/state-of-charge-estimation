import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots


def load_lut(lut_path: str) -> pl.DataFrame:
    """Loads and prepares the Lookup Table."""
    return (
        pl.scan_parquet(lut_path)
        .with_columns(
            # Calculate Depth of Discharge (DOD)
            (pl.col("q_removed_ah") / 277 * 100).round(0).alias("dod")
        )
        .filter(pl.col("dod").is_between(0, 100, closed="both"))
        .group_by(["soh", "cell_temperatures", "current_average", "dod"])
        .agg(
            (pl.col("cell_voltages_mean").mean()).round(0).alias("cell_voltages_mean")
        )
        .filter(pl.col("soh").is_between(98, 102, closed="both"))
        .group_by(["cell_temperatures", "current_average", "dod"])
        .agg(
            (pl.col("cell_voltages_mean").mean())
            .round(0)
            .alias("cell_voltages_mean")
        )
        .with_columns((100.0 - pl.col("dod")).alias("soc_lut"))
        .with_columns(
            pl.col("dod").count().over(["cell_temperatures","current_average"]).alias("dod_count")
        )
        .filter(pl.col("dod_count")>=90)
        # .sort(["current_average", "cell_temperatures", "soc_lut"])
        # .with_columns(
        #     # Calculate the delta SoC that corresponds to a 2mV change in voltage
            
        # voltage_delta=pl.col("cell_voltages_mean").diff().over(["cell_temperatures", "current_average"]).abs()
        # )
        # .filter(pl.col("voltage_delta") >= 15)  # Ensure we have a valid voltage change
        .filter((pl.col("current_average").is_between(100, 110, closed="both")) )  # Focus on reasonable current ranges
        .collect()
    )


def compute_hybrid_soc(
    df: pl.DataFrame, df_lut: pl.DataFrame, cell_idx: int, cap_ah: float = 412.0
) -> pl.DataFrame:
    """Computes the estimated SoC using LUT resets and Coulomb Counting (delta_soc)."""
    # 1. Prepare bins on the main dataframe to match LUT keys
    df_bins = df.with_columns(
        i_scaled=(pl.col("current") / cap_ah * 277.0) 
    ).with_columns(
        current_bin=(pl.col("i_scaled") // 5 * 5).cast(pl.Float64),
        temp_bin=(pl.col("temperature_battery") // 5 * 5).cast(pl.Float64),
        # Convert V to mV and match the 2mV rounding rule from the LUT
        v_mv_rounded=(
            ((pl.col(f"cell_voltage_{cell_idx}") * 1000.0)) 
        ).cast(pl.Float64),
    )

    # 2. Join the LUT data directly onto our dataframe based on the matching bins
    df_joined = df_bins.join(
        df_lut,
        left_on=["temp_bin", "current_bin", "v_mv_rounded"],
        right_on=["cell_temperatures", "current_average", "cell_voltages_mean"],
        how="left",
    )

    # Extract required arrays for the fast sequential pass
    soc_lut_arr = df_joined["soc_lut"].to_numpy()
    delta_soc_arr = df_joined["delta_soc"].to_numpy()
    soc_reported = df_joined["soc"].to_numpy()

    # Pre-allocate output arrays
    soc_hybrid = np.zeros(len(df_joined))
    soc_cc_only = np.zeros(len(df_joined))

    # Initialize the first step using the reported SoC
    soc_hybrid[0] = soc_reported[0] if np.isnan(soc_lut_arr[0]) else soc_lut_arr[0]
    soc_cc_only[0] = soc_reported[0]

    # 3. Sequential propagation loop
    for i in range(1, len(df_joined)):
        # Pure Coulomb Counting baseline
        soc_cc_only[i] = np.clip(soc_cc_only[i - 1] + delta_soc_arr[i], 0, 100)

        # Hybrid Method: Reset if LUT value exists, otherwise accumulate delta_soc
        if not np.isnan(soc_lut_arr[i]):
            soc_hybrid[i] = soc_lut_arr[i]
        else:
            soc_hybrid[i] = np.clip(soc_hybrid[i - 1] + delta_soc_arr[i], 0, 100)

    # Append results back to the dataframe
    return df_joined.with_columns(
        [
            pl.Series(f"soc_hybrid_{cell_idx}", soc_hybrid),
            pl.Series(f"soc_cc_{cell_idx}", soc_cc_only),
        ]
    )


# ── Processing Data ───────────────────────────────────────────────────────────

# Load main dataframe and calculate standard delta_soc
df = pl.read_csv("data/P24-000104.csv").sort("sequence_number")
df = df.with_columns(
    delta_soc=(pl.col("charge_in").diff().fill_null(0) - pl.col("charge_out").diff().fill_null(0))
    / (412 * 3600)
    * 100, 
    current=(pl.col("charge_in").diff().fill_null(0) - pl.col("charge_out").diff().fill_null(0))/60
)

# Load Look-up Table
df_lut = load_lut(
    r"data\voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet"
)

# For testing / simplicity, let's run it on Cell 1
# You can wrap this in a loop for all 16 cells if desired
cell_to_analyze = 1
df_results = compute_hybrid_soc(df, df_lut, cell_idx=cell_to_analyze)
# Calculate Errors relative to the reported SoC profile
df_metrics = df_results.with_columns(
    [
        (pl.col(f"soc_hybrid_{cell_to_analyze}") - pl.col("soc")).alias("error_hybrid"),
        (pl.col(f"soc_cc_{cell_to_analyze}") - pl.col("soc")).alias("error_cc"),
    ]
)
df_metrics.write_csv("metrics_output.csv")  # Save metrics for external analysis if needed
# Gather quick stats
stats = df_metrics.select(
    [
        pl.col("error_hybrid").abs().mean().alias("mae_hybrid"),
        pl.col("error_cc").abs().mean().alias("mae_cc"),
    ]
).to_dict(as_series=False)

mae_hybrid = stats["mae_hybrid"][0]
mae_cc = stats["mae_cc"][0]

# ── Plotting with Plotly ──────────────────────────────────────────────────────

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=(
        f"SoC Tracking Comparison (Cell {cell_to_analyze})",
        "Estimation Error (Estimated SoC - Reported SoC)",
    ),
)

# Plot 1: Tracking Profiles
fig.add_trace(
    go.Scatter(
        x=df_metrics["sequence_number"],
        y=df_metrics["soc"],
        name="Reported SoC",
        line=dict(color="black", width=2),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_metrics["sequence_number"],
        y=df_metrics[f"soc_hybrid_{cell_to_analyze}"],
        name=f"Hybrid (LUT+CC) [MAE: {mae_hybrid:.2f}%]",
        line=dict(color="orange", dash="dash"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_metrics["sequence_number"],
        y=df_metrics[f"soc_cc_{cell_to_analyze}"],
        name=f"Pure Coulomb Counting [MAE: {mae_cc:.2f}%]",
        line=dict(color="royalblue", dash="dot"),
    ),
    row=1,
    col=1,
)

# Plot 2: Error Analysis
fig.add_trace(
    go.Scatter(
        x=df_metrics["sequence_number"],
        y=df_metrics["error_hybrid"],
        name="Hybrid Error",
        line=dict(color="firebrick"),
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_metrics["sequence_number"],
        y=df_metrics["error_cc"],
        name="Coulomb Counting Error",
        line=dict(color="darkgrey"),
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_metrics["sequence_number"],
        y=df_metrics["current_bin"],
        name="Current",
        line=dict(color="red"),
    ),
    row=3,
    col=1,
)
# Update layout details
fig.update_layout(
    height=800,
    title_text="SoC Estimation Strategy Performance Evaluation",
    xaxis2_title="Sequence Number",
    yaxis_title="State of Charge (%)",
    yaxis2_title="Error Δ (%)",
    template="plotly_white",
    hovermode="x unified",
)

fig.show()


# # def analyze_best_reset_currents(df_lut: pl.DataFrame) -> pl.DataFrame:
# #     """Analyzes the LUT to find which current bins provide the most distinct

# #     voltage-to-SoC changes (highest voltage gradient per % SoC).
# #     """
# #     analysis_df = (
# #         df_lut.sort(["current_average", "cell_temperatures", "soc_lut"])
# #         # Calculate the voltage change between consecutive SoC steps
# #         .with_columns(
# #             pl.col("cell_voltages_mean")
# #             .diff()
# #             .over(["current_average", "cell_temperatures"])
# #             .abs()
# #             .alias("v_delta_per_soc")
# #         )
# #         # Drop nulls from diff()
# #         .drop_nulls("v_delta_per_soc")
# #         # Aggregate performance metrics per current bucket
# #         .group_by("current_average")
# #         .agg(
# #             [
# #                 pl.col("v_delta_per_soc")
# #                 .mean()
# #                 .alias("mean_gradient_mv"),  # Higher is better
# #                 pl.col("v_delta_per_soc")
# #                 .std()
# #                 .alias("gradient_stability"),  # Lower means more consistent
# #                 pl.count("soc_lut").alias("total_mapped_points"),
# #             ]
# #         )
# #         .sort("mean_gradient_mv", descending=True)
# #     )

# #     return analysis_df


# # # Run the analysis
# # df_current_performance = analyze_best_reset_currents(df_lut)
# # print(df_current_performance)
# # import plotly.graph_objects as go
# # from plotly.subplots import make_subplots

# # # Convert to standard types for plotting
# # currents = df_current_performance["current_average"].to_numpy()
# # gradients = df_current_performance["mean_gradient_mv"].to_numpy()
# # stability = df_current_performance["gradient_stability"].to_numpy()

# # # Create a dual-axis plot
# # fig = make_subplots(specs=[[{"secondary_y": True}]])

# # # Add Mean Gradient Bar (Primary Axis)
# # fig.add_trace(
# #     go.Bar(
# #         x=currents,
# #         y=gradients,
# #         name="Mean Voltage Gradient (mV / 1% SoC)",
# #         marker_color="teal",
# #         opacity=0.75,
# #     ),
# #     secondary_y=False,
# # )

# # # Add Stability/Standard Deviation Line (Secondary Axis)
# # fig.add_trace(
# #     go.Scatter(
# #         x=currents,
# #         y=stability,
# #         name="Gradient Variance (Lower = More Stable)",
# #         mode="lines+markers",
# #         line=dict(color="crimson", width=2),
# #     ),
# #     secondary_y=True,
# # )

# # # Formatting layout
# # fig.update_layout(
# #     title="<b>LUT Reset Feasibility Analysis by Current Bucket</b><br><sup>Higher gradients yield cleaner SoC tracking updates</sup>",
# #     xaxis_title="Current Bin (A)",
# #     template="plotly_white",
# #     hovermode="x unified",
# #     legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
# # )

# # fig.update_yaxes(
# #     title_text="<b>Signal Strength:</b> Gradient (mV / %SoC)", secondary_y=False
# # )
# # fig.update_yaxes(
# #     title_text="<b>Noise Risk:</b> Standard Deviation (mV)", secondary_y=True
# # )

# # fig.show()
# import numpy as np
# import plotly.graph_objects as go
# import polars as pl
# from plotly.subplots import make_subplots


# def evaluate_lut_filters(
#     df: pl.DataFrame, df_lut_raw: pl.DataFrame, cell_idx: int
# ) -> pl.DataFrame:
#     """Sweeps multiple LUT filtering and binning configurations to find the

#     optimal balance between hit-frequency (count) and prediction error (MAE).
#     """
#     # Pre-calculate delta_soc and raw voltage in mV for the target cell
#     df_base = df.with_columns(
#         v_mv=pl.col(f"cell_voltage_{cell_idx}") * 1000.0,
#         i_scaled=pl.col("current") / 412.0 * 277.0,
#     )

#     results = []

#     # Define the experiment matrix to find the "best" settings
#     configs = [
#         {"name": "Narrow Positive", "i_min": 100, "i_max": 110, "v_round": 2},
#         {"name": "Wide Positive", "i_min": 90, "i_max": 130, "v_round": 2},
#         {"name": "Narrow Negative", "i_min": -100, "i_max": -70, "v_round": 2},
#         {"name": "Asymmetric Combined", "i_min": -100, "i_max": 110, "v_round": 2},
#         {"name": "Coarse Voltage (4mV)", "i_min": 100, "i_max": 110, "v_round": 4},
#         {"name": "Fine Voltage (1mV)", "i_min": 100, "i_max": 110, "v_round": 1},
#     ]

#     for config in configs:
#         # 1. Process LUT with the specific configuration filters
#         df_lut_configured = (
#             df_lut_raw.with_columns(
#                 (pl.col("q_removed_ah") / 277 * 100).round(0).alias("dod")
#             )
#             .filter(pl.col("dod").is_between(0, 100, closed="both"))
#             .filter(pl.col("soh").is_between(98, 102, closed="both"))
#             .group_by(["cell_temperatures", "current_average", "dod"])
#             .agg(
#                 pl.col("cell_voltages_mean")
#                 .mean()
#                 .round(0)
#                 .alias("cell_voltages_mean")
#             )
#             .with_columns((100.0 - pl.col("dod")).alias("soc_lut"))
#             # Apply variable current range criteria
#             .filter(
#                 pl.col("current_average")
#                 .abs()
#                 .is_between(
#                     abs(config["i_min"]), abs(config["i_max"]), closed="both"
#                 )
#             )
#         )

#         # 2. Adjust data binning resolution based on configuration rules
#         vr = config["v_round"]
#         df_run = df_base.with_columns(
#             current_bin=(pl.col("i_scaled") // 5 * 5).cast(pl.Float64),
#             temp_bin=(pl.col("temperature_battery") // 5 * 5).cast(pl.Float64),
#             v_mv_rounded=((pl.col("v_mv") // vr) * vr).cast(pl.Float64),
#         )

#         # 3. Vectorized mapping join
#         df_mapped = df_run.join(
#             df_lut_configured,
#             left_on=["temp_bin", "current_bin", "v_mv_rounded"],
#             right_on=[
#                 "cell_temperatures",
#                 "current_average",
#                 "cell_voltages_mean",
#             ],
#             how="left",
#         )

#         # 4. Extract arrays for sequential hybrid error evaluation
#         soc_lut_arr = df_mapped["soc_lut"].to_numpy()
#         delta_soc_arr = df_mapped["delta_soc"].to_numpy()
#         soc_reported = df_mapped["soc"].to_numpy()

#         soc_hybrid = np.zeros(len(df_mapped))
#         soc_hybrid[0] = (
#             soc_reported[0] if np.isnan(soc_lut_arr[0]) else soc_lut_arr[0]
#         )

#         total_resets = 0

#         for i in range(1, len(df_mapped)):
#             if not np.isnan(soc_lut_arr[i]):
#                 soc_hybrid[i] = soc_lut_arr[i]
#                 total_resets += 1
#             else:
#                 soc_hybrid[i] = np.clip(
#                     soc_hybrid[i - 1] + delta_soc_arr[i], 0, 100
#                 )

#         # Calculate performance analytics
#         mae = np.mean(np.abs(soc_hybrid - soc_reported))
#         lookup_frequency = (total_resets / len(df_mapped)) * 100

#         results.append(
#             {
#                 "Configuration": config["name"],
#                 "Current_Range": f"{config['i_min']} to {config['i_max']} A",
#                 "Voltage_Rounding": f"{vr} mV",
#                 "Lookup_Frequency_%": lookup_frequency,
#                 "MAE_%": mae,
#             }
#         )

#     return pl.DataFrame(results)


# # Execute tracking optimization assessment matrix
# df_optimization_summary = evaluate_lut_filters(df,  pl.read_parquet(r"data\voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet"), cell_idx=1)
# print(df_optimization_summary)
# # Convert summary metrics for visualization
# labels = df_optimization_summary["Configuration"].to_list()
# mae_vals = df_optimization_summary["MAE_%"].to_numpy()
# freq_vals = df_optimization_summary["Lookup_Frequency_%"].to_numpy()

# fig = go.Figure()

# # Add bubble trace showing trade-offs
# fig.add_trace(
#     go.Scatter(
#         x=freq_vals,
#         y=mae_vals,
#         mode="markers+text",
#         text=labels,
#         textposition="top center",
#         marker=dict(
#             size=18,
#             color=mae_vals,
#             colorscale="Viridis",
#             showscale=True,
#             colorbar=dict(title="MAE Error (%)"),
#             line=dict(width=2, color="DarkSlateGrey"),
#         ),
#         hovertemplate="<b>%{text}</b><br>"
#         + "Lookup Frequency: %{x:.2f}%<br>"
#         + "System MAE: %{y:.3f}%<extra></extra>",
#     )
# )

# # Format axes and annotations
# fig.update_layout(
#     title="<b>LUT Optimization: Error vs. Match Density Trade-off Matrix</b><br><sup>The ideal configuration resides in the bottom-right corner (Low Error, High Reset Frequency)</sup>",
#     xaxis_title="Lookup Hit Frequency (% of Time series Row Count)",
#     yaxis_title="Total Multi-Hour Profile MAE (%)",
#     template="plotly_white",
#     height=600,
# )

# # Invert Y axis because a lower MAE error indicates superior accuracy performance
# fig.update_yaxes(autorange="reversed")

# fig.show()

# import itertools
# import numpy as np
# import plotly.graph_objects as go
# import polars as pl

# # 1. Load data and establish standard CC baseline
# df = pl.read_csv("data/P24-000104.csv").sort("sequence_number")
# df = df.with_columns(
#     delta_soc=(
#         pl.col("charge_in").diff().fill_null(0)
#         - pl.col("charge_out").diff().fill_null(0)
#     )
#     / (412 * 3600)
#     * 100
# )

# # Build CC baseline array sequentially to match exactly what our loop outputs
# delta_soc_arr = df["delta_soc"].to_numpy()
# soc_cc_baseline = np.zeros(len(df))
# soc_cc_baseline[0] = df["soc"][0]  # seed with reported initial value
# for i in range(1, len(df)):
#     soc_cc_baseline[i] = np.clip(soc_cc_baseline[i - 1] + delta_soc_arr[i], 0, 100)

# # Pre-calculate invariant arrays on main dataframe
# v_mv_base = (df[f"cell_voltage_1"] * 1000.0).to_numpy()
# i_scaled_base = (df["current"] / 412.0 * 277.0).to_numpy()
# temp_battery_base = df["temperature_battery"].to_numpy()

# # Load raw LUT structure
# df_lut_raw = (
#     pl.scan_parquet(
#         r"data\voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet"
#     )
#     .with_columns((pl.col("q_removed_ah") / 277 * 100).round(0).alias("dod"))
#     .filter(
#         pl.col("dod").is_between(0, 100, closed="both")
#         & pl.col("soh").is_between(98, 102, closed="both")
#     )
#     .group_by(["cell_temperatures", "current_average", "dod"])
#     .agg(
#         pl.col("cell_voltages_mean").mean().round(0).alias("cell_voltages_mean")
#     )
#     .with_columns((100.0 - pl.col("dod")).alias("soc_lut"))
#     .collect()
# )

# # 2. Define Parametric Ranges for Optimization Sweep
# pos_ranges = [
#     (90, 110),
#     (100, 110),
#     (100, 120),
#     (90, 130),
# ]  # Positive current testing slices
# neg_ranges = [
#     (-100, -70),
#     (-110, -70),
#     (-120, -80),
#     (-130, -90),
# ]  # Negative current testing slices

# optimization_results = []

# # 3. Sweep combination permutations
# for pos_bound, neg_bound in itertools.product(pos_ranges, neg_ranges):
#     p_min, p_max = pos_bound
#     n_min, n_max = neg_bound

#     # Filter the LUT for this specific iteration slice
#     df_lut_slice = df_lut_raw.filter(
#         (
#             pl.col("current_average").is_between(p_min, p_max, closed="both")
#             # | pl.col("current_average").is_between(n_min, n_max, closed="both")
#         )
#     )

#     # Fast mapping setup via Polars Key Join
#     df_bins = pl.DataFrame(
#         {
#             "current_bin": (i_scaled_base // 5 * 5).astype(np.float64),
#             "temp_bin": (temp_battery_base // 5 * 5).astype(np.float64),
#             "v_mv_rounded": ((v_mv_base // 2) * 2).astype(np.float64),
#         }
#     )

#     df_mapped = df_bins.join(
#         df_lut_slice,
#         left_on=["temp_bin", "current_bin", "v_mv_rounded"],
#         right_on=["cell_temperatures", "current_average", "cell_voltages_mean"],
#         how="left",
#     )

#     soc_lut_arr = df_mapped["soc_lut"].to_numpy()

#     # Hybrid simulation tracking loop
#     soc_hybrid = np.zeros(len(df))
#     soc_hybrid[0] = soc_cc_baseline[0]
#     total_resets = 0

#     for i in range(1, len(df)):
#         if not np.isnan(soc_lut_arr[i]):
#             soc_hybrid[i] = soc_lut_arr[i]
#             total_resets += 1
#         else:
#             soc_hybrid[i] = np.clip(soc_hybrid[i - 1] + delta_soc_arr[i], 0, 100)

#     # Metric evaluation against our static Coulomb Counter baseline
#     mae_vs_cc = np.mean(np.abs(soc_hybrid - soc_cc_baseline))
#     lookup_count = total_resets

#     optimization_results.append(
#         {
#             "pos_range": f"[{p_min}, {p_max}]",
#             "neg_range": f"[{n_min}, {n_max}]",
#             "label": f"Pos:{p_min}-{p_max}A | Neg:{n_min}-{n_max}A",
#             "lookup_count": lookup_count,
#             "mae_vs_cc": mae_vs_cc,
#         }
#     )

# df_opt = pl.DataFrame(optimization_results).sort("mae_vs_cc")
# print(df_opt)

# fig = go.Figure()

# fig.add_trace(
#     go.Scatter(
#         x=df_opt["lookup_count"].to_numpy(),
#         y=df_opt["mae_vs_cc"].to_numpy(),
#         mode="markers+text",
#         text=df_opt["label"].to_list(),
#         textposition="top center",
#         marker=dict(
#             size=14,
#             color=df_opt["lookup_count"].to_numpy(),
#             colorscale="Plasma",
#             showscale=True,
#             colorbar=dict(title="Total Hits (Reset Counts)"),
#             line=dict(width=1, color="black"),
#         ),
#         hovertemplate="<b>%{text}</b><br>"
#         + "Total Matched Hits: %{x}<br>"
#         + "MAE Deviation from CC: %{y:.4f}%<extra></extra>",
#     )
# )

# fig.update_layout(
#     title="<b>LUT Optimization Matrix: Hit Density vs. CC Baseline Shift</b><br><sup>Ideal ranges live in the lower right (High hit count, Minimal deviation spike from CC profile)</sup>",
#     xaxis_title="Lookup Match Hit Frequency (Count over Timeline)",
#     yaxis_title="Mean Absolute Error (MAE) vs. Pure Coulomb Counting Baseline",
#     template="plotly_white",
#     height=650,
# )

# fig.show()

# import numpy as np
# import plotly.graph_objects as go
# import polars as pl

# # ── 1. DATA PREPARATION ───────────────────────────────────────────────────────
# df = pl.read_csv("data/P24-000104.csv").sort("sequence_number")
# df = df.with_columns(
#     delta_soc=(
#         pl.col("charge_in").diff().fill_null(0)
#         - pl.col("charge_out").diff().fill_null(0)
#     )
#     / (412 * 3600)
#     * 100
# )

# # Build pure Coulomb Counter baseline array
# delta_soc_arr = df["delta_soc"].to_numpy()
# soc_cc_baseline = np.zeros(len(df))
# soc_cc_baseline[0] = df["soc"][0]
# for i in range(1, len(df)):
#     soc_cc_baseline[i] = np.clip(soc_cc_baseline[i - 1] + delta_soc_arr[i], 0, 100)

# # Extract baseline telemetry arrays
# v_mv_base = (df["cell_voltage_1"] * 1000.0).to_numpy()
# i_scaled_base = (df["current"] / 412.0 * 277.0).to_numpy()
# temp_battery_base = df["temperature_battery"].to_numpy()

# # Load raw base LUT
# df_lut_raw = (
#     pl.scan_parquet(
#         r"data\voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet"
#     )
#     .with_columns((pl.col("q_removed_ah") / 277 * 100).round(0).alias("dod"))
#     .filter(
#         pl.col("dod").is_between(0, 100, closed="both")
#         & pl.col("soh").is_between(98, 102, closed="both")
#     )
#     .group_by(["cell_temperatures", "current_average", "dod"])
#     .agg(
#         pl.col("cell_voltages_mean").mean().round(0).alias("cell_voltages_mean")
#     )
#     .with_columns((100.0 - pl.col("dod")).alias("soc_lut"))
#     .collect()
# )

# # Map main dataframe tracking bins out once
# df_bins = pl.DataFrame(
#     {
#         "current_bin": (i_scaled_base // 5 * 5).astype(np.float64),
#         "temp_bin": (temp_battery_base // 5 * 5).astype(np.float64),
#         "v_mv_rounded": ((v_mv_base // 2) * 2).astype(np.float64),
#     }
# )


# # ── 2. CORE EVALUATION FUNCTION ───────────────────────────────────────────────
# def evaluate_bounds(lut_filter_expr, label_name, category):
#     df_lut_slice = df_lut_raw.filter(lut_filter_expr)

#     df_mapped = df_bins.join(
#         df_lut_slice,
#         left_on=["temp_bin", "current_bin", "v_mv_rounded"],
#         right_on=["cell_temperatures", "current_average", "cell_voltages_mean"],
#         how="left",
#     )

#     soc_lut_arr = df_mapped["soc_lut"].to_numpy()
#     soc_hybrid = np.zeros(len(df))
#     soc_hybrid[0] = soc_cc_baseline[0]
#     total_resets = 0

#     for i in range(1, len(df)):
#         if not np.isnan(soc_lut_arr[i]):
#             soc_hybrid[i] = soc_lut_arr[i]
#             total_resets += 1
#         else:
#             soc_hybrid[i] = np.clip(
#                 soc_hybrid[i - 1] + delta_soc_arr[i], 0, 100
#             )

#     return {
#         "label": label_name,
#         "category": category,
#         "lookup_count": total_resets,
#         "mae_vs_cc": np.mean(np.abs(soc_hybrid - soc_cc_baseline)),
#         "filter_expr": lut_filter_expr,
#     }


# sweep_results = []

# # ── 3. RUN SWEEPS ─────────────────────────────────────────────────────────────
# # Phase 1: Positive 10A bins (0 to 150)
# for low in range(0, 150, 10):
#     high = low + 10
#     expr = pl.col("current_average").is_between(low, high, closed="both")
#     res = evaluate_bounds(expr, f"Pos: {low} to {high}A", "Positive Only")
#     sweep_results.append(res)

# # Phase 2: Negative 10A bins (-150 to 0)
# for low in range(-150, 0, 10):
#     high = low + 10
#     expr = pl.col("current_average").is_between(low, high, closed="both")
#     res = evaluate_bounds(expr, f"Neg: {low} to {high}A", "Negative Only")
#     sweep_results.append(res)

# df_all_bins = pl.DataFrame(sweep_results)

# # ── 4. CHOOSE AND EVALUATE BEST OF BOTH ───────────────────────────────────────
# # Filter out bins that returned 0 lookups to prevent false accuracy readings
# df_valid = df_all_bins.filter(pl.col("lookup_count") > 0)

# # Sort by lowest MAE to find the winners
# best_pos = (
#     df_valid.filter(pl.col("category") == "Positive Only")
#     .sort("mae_vs_cc").head(1)
# )
# best_neg = (
#     df_valid.filter(pl.col("category") == "Negative Only")
#     .sort("mae_vs_cc")
#     .head(1)
# )

# # Combine the expressions of the two winners
# combined_expr = best_pos["filter_expr"][0] | best_neg["filter_expr"][0]
# combined_label = f"Best Combo ({best_pos['label'][0]} + {best_neg['label'][0]})"

# best_of_both_res = evaluate_bounds(combined_expr, combined_label, "Best of Both")
# df_final_matrix = pl.concat([df_all_bins, pl.DataFrame([best_of_both_res])]).sort(
#     "mae_vs_cc"
# )

# # Print out results to inspect
# print(
#     df_final_matrix.select(
#         ["label", "category", "lookup_count", "mae_vs_cc"]
#     ).to_pandas()
# )

# fig = go.Figure()

# # Define color schemes for our categories
# colors = {
#     "Positive Only": "rgba(46, 204, 113, 0.8)",  # Green
#     "Negative Only": "rgba(52, 152, 219, 0.8)",  # Blue
#     "Best of Both": "rgba(231, 76, 60, 1.0)",  # Red
# }

# for cat in ["Positive Only", "Negative Only", "Best of Both"]:
#     df_sub = df_final_matrix.filter(pl.col("category") == cat)
#     if df_sub.is_empty():
#         continue

#     fig.add_trace(
#         go.Scatter(
#             x=df_sub["lookup_count"].to_numpy(),
#             y=df_sub["mae_vs_cc"].to_numpy(),
#             mode="markers+text" if cat == "Best of Both" else "markers",
#             text=df_sub["label"].to_list(),
#             textposition="top center",
#             name=cat,
#             marker=dict(
#                 size=22 if cat == "Best of Both" else 12,
#                 color=colors[cat],
#                 line=dict(
#                     width=2 if cat == "Best of Both" else 0, color="black"
#                 ),
#             ),
#             hovertemplate="<b>%{text}</b><br>"
#             + "Lookup Count: %{x}<br>"
#             + "MAE vs CC Profile: %{y:.4f}%<extra></extra>",
#         )
#     )

# fig.update_layout(
#     title="<b>State-of-Charge Calibration Resolution Sweep</b><br><sup>Comparing 10A slices against the dual optimized 'Best of Both' boundary</sup>",
#     xaxis_title="Lookup Target Matches (Higher = More Frequent Calibrations)",
#     yaxis_title="Mean Absolute Deviation (MAE) vs Pure CC Profile (%)",
#     template="plotly_white",
#     hovermode="closest",
#     height=600,
# )

# fig.show()