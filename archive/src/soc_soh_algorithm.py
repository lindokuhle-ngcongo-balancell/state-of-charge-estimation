import numpy as np
import polars as pl
from scipy.signal import savgol_filter
WINDOW_SIZE = 121
MULTIPLIER_FOR_SOH=7
df_lut = (
    pl.scan_parquet(
        r"data\\voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet",
    )
    .with_columns(
        (pl.col('q_removed_ah') / ( 277) * 100)
        .round(0)
        .alias('dod')
    )
    .group_by(['soh',"cell_temperatures","current_average", 'dod'])
    .agg(
        (pl.col('cell_voltages_mean').mean()).round(0).alias('cell_voltages_mean')
    )
    .sort(['soh',"cell_temperatures","current_average", 'dod'])
    .with_columns(
        pl.col('cell_voltages_mean').diff().over(
            partition_by=['soh',"cell_temperatures","current_average"], order_by="dod"
        ).alias('cell_voltages_diff')
    )
    # .with_columns(
    #     pl.when((pl.col('cell_voltages_diff').abs() >3.5) | (pl.col("dod")==0)).then(pl.col('cell_voltages_mean')).otherwise(-1).alias('cell_voltages_mean')
    # )
    .collect()
)
def vectorised_soc_estimation(df_lut: pl.DataFrame, df: pl.DataFrame, soh: float = 100) -> pl.DataFrame:
    """
    Vectorised version of algorithm_soc_estimation.
    Returns df with added columns: soc_min, soc_max.
    """

    # ── Pull LUT into numpy once ──────────────────────────────────────────────
    lut = df_lut.filter(pl.col("soh") == soh)
    lut_temp = lut["cell_temperatures"].to_numpy()
    lut_curr = lut["current_average"].to_numpy()
    lut_volt = lut["cell_voltages_mean"].to_numpy()
    lut_dod  = lut["dod"].to_numpy().astype(int)

    # ── Pull data into numpy ──────────────────────────────────────────────────
    temps    = df["cell_temperatures"].to_numpy()
    currents = df["current_average"].to_numpy()
    voltages = df["cell_voltages"].to_numpy()
    n        = len(df)

    soc_min_arr = np.full(n, np.nan)
    soc_max_arr = np.full(n, np.nan)

    # ── Iterate over unique (temp, current) bins ──────────────────────────────
    unique_keys = np.unique(np.stack([temps, currents], axis=1), axis=0)

    for temp_bin, curr_bin in unique_keys:
        # Slice of LUT for this bin, sorted by DOD
        lut_mask = (lut_temp == temp_bin) & (lut_curr == curr_bin)
        if not lut_mask.any():
            continue  # no LUT entry → leave as NaN (equivalent to returning -1,-1)

        bin_dod  = lut_dod[lut_mask]   # shape (D,)
        bin_volt = lut_volt[lut_mask]  # shape (D,)

        # Drop invalid voltages (< 0) from LUT slice
        valid = bin_volt >= 0
        bin_dod  = bin_dod[valid]
        bin_volt = bin_volt[valid]

        if len(bin_dod) == 0:
            continue

        # Sort by DOD ascending so indexing is consistent
# Replace the bracketing section inside the bin loop with this:

# Sort LUT by DOD (0→100), voltages follow
        sort_idx = np.argsort(bin_dod)
        bin_dod  = bin_dod[sort_idx]   # [0, 1, 2, ... 100]
        bin_volt = bin_volt[sort_idx]  # voltage at each DOD step

        v_max = bin_volt[0]   # voltage at DOD=0  (fully charged)
        v_min = bin_volt[-1]  # voltage at DOD=100 (fully discharged)

        data_mask = (temps == temp_bin) & (currents == curr_bin)
        v_query   = voltages[data_mask]

        # For each query voltage, find where it sits in the DOD-ordered voltage array.
        # Since voltage generally DECREASES as DOD increases, we flip for searchsorted.
        flipped_volt = bin_volt[::-1]   # now ascending (low DOD = high V → flip)
        flipped_dod  = bin_dod[::-1]

        # searchsorted in the ascending-flipped array
        idx_right = np.searchsorted(flipped_volt, v_query, side='left').clip(0, len(flipped_volt) - 1)
        idx_left  = (idx_right - 1).clip(0, len(flipped_volt) - 1)

        dod_at_left  = flipped_dod[idx_left]
        dod_at_right = flipped_dod[idx_right]

        # dod_min/max are the bracket — always take true min/max in case of plateau ties
        dod_min_v = np.minimum(dod_at_left, dod_at_right).clip(0, 100)
        dod_max_v = np.maximum(dod_at_left, dod_at_right).clip(0, 100)

        # Boundary conditions
        above = v_query > v_max
        below = v_query < v_min

        dod_min_v = np.where(above, 0,   np.where(below, 100, dod_min_v))
        dod_max_v = np.where(above, 0,   np.where(below, 100, dod_max_v))

        # SOC is 100 - DOD, and soc_min comes from dod_max
        soc_min_arr[data_mask] = (100 - dod_max_v).clip(0, 100)
        soc_max_arr[data_mask] = (100 - dod_min_v).clip(0, 100)

    return df.with_columns([
        pl.Series("soc_min", soc_min_arr, dtype=pl.Float64),
        pl.Series("soc_max", soc_max_arr, dtype=pl.Float64),
    ])

# ── Run ───────────────────────────────────────────────────────────────────────
df_test = (
    pl.read_parquet(
        r"data\\BE19090008_cell_5_processed_soc_rounded_2dp_daily_cv_3500.parquet"
    )
    .filter(pl.col("sequence_number") >= 800_000)
    .sort("sequence_number")
    .with_columns(
        ((pl.col("cell_temperatures") // 5 )* 5).alias("cell_temperatures"),
        ((pl.col("current_average")   // 5 )* 5).alias("current_average"),
    )
    .select(
        "sequence_number", "cell_temperatures", "current_average",
        "soc_nominal", "cell_voltages","soh", "soc",
    )
    .with_columns(
        previous_soc  =pl.col("soc_nominal").shift(1),
        previous_soc_5=pl.col("soc_nominal").shift(5),
    )
    .with_columns(
        charge_movement           =(pl.col("soc_nominal") - pl.col("previous_soc"))   / 100,
        rolling_charge_movement_5 =(pl.col("soc_nominal") - pl.col("previous_soc_5")) / 100,
    )
)
# df_test=(
#                         pl.read_csv(
#                         # "data\\Pioneer_Foods_1_April_2026_to_5_May_2026\\pioneer_foods_1_April_2026_to_5_May_2026.csv"
#                         "data/test_data/P23-000141.csv"
#                         )
#                         # ).filter(pl.col("device_serial")=="P23-000156")
#                         .sort(["device_serial", "sequence_number"])
#                         .with_columns(
#                             max_charge=pl.col("charge_capacity").last().over(partition_by="device_serial", order_by="sequence_number")
#                         )
#                         .with_columns(
#                             # Scale current so that it is expressed in the same units as the LUT
#                             current_scaled=(
#                                 pl.col("current") / (pl.col("max_charge")/3600.0 )* 277.0
#                             ),
#                             # Bin temperature and (scaled) current to nearest 5 — matches LUT binning
#                             cell_temperatures=(
#                                 (pl.col("temperature_battery") // 5 * 5).cast(pl.Float64)
#                             ),
#                         )
#                         .with_columns(
#                             current_average=((pl.col("current_scaled") // 5 * 5).cast(pl.Float64))
#                         )
#                         # Rename columns to match what the estimation functions expect
#                         .rename(
#                             {
#                                 "min_cell_voltage": "cell_voltages",
#                                 "soc":              "soc_nominal",
#                             }
#                         )
#                         .with_columns(
#                             cell_voltages=pl.col("cell_voltages")*1000
#                         )
#                         .filter(pl.col("cell_voltages")>0)
#                         .select(
#                             "device_serial",
#                             "sequence_number",
#                             "cell_temperatures",
#                             "current_average",
#                             "cell_voltages",
#                             "soc_nominal",
#                             # "charge_capacity",
#                             "time_stamp"
#                         )
#                         # .collect()
#             )
# print(len(df_test))
# # ── Run ───────────────────────────────────────────────────────────────────────
result = vectorised_soc_estimation(df_lut, df_test, soh=100)
import plotly.express as px
# from scipy.stats import poisson

def make_weights(window_size: int, decay: float = 0.1) -> list[float]:
    """
    Exponential weights — latest sample (last in window) gets weight 1.0,
    older samples decay exponentially.
    decay: higher = faster falloff (0.05 gentle, 0.3 aggressive)
    """
    x = np.arange(window_size)
    weights = np.exp(decay * (x - (window_size - 1)))  # peaks at last index
    return (weights / weights.sum()).tolist()

WEIGHTS = make_weights(61, decay=0.7)

# # .rolling_mean(window_size=WINDOW_SIZE, weights=WEIGHTS, min_samples=1)
# # ── Plot ──────────────────────────────────────────────────────────────────────
# # result_long = (
# #     result
# #     .drop_nulls().drop_nans().with_columns(
# #         rolling_soc_max=pl.col("soc_max").rolling_mean(window_size=WINDOW_SIZE, min_samples=1),
# #         rolling_soc_min=pl.col("soc_min").rolling_mean(window_size=WINDOW_SIZE, min_samples=1),
# #         rolling_soc_mean=((pl.col("soc_max")+pl.col("soc_min"))/2).rolling_mean(window_size=WINDOW_SIZE, weights=WEIGHTS, min_samples=1)
# #     )
# #     .with_columns((pl.col("sequence_number") // 1_440).alias("day"))# limit to first 10 days for better animation
# #     .with_columns(sequence_number=(pl.col("sequence_number")-pl.col("sequence_number").min()).over("day"),
# #                   sequence_number_original=pl.col("sequence_number"),  # keep original for hover
# #                   soh_estimate=(
# #                                 (100 - (pl.col("soc_nominal") - pl.col("rolling_soc_mean")))
# #                                 # .pipe(lambda s: pl.when(s.is_between(98.5, 100, closed="both")).then(None).otherwise(s))
# #                                 .rolling_mean(window_size=WINDOW_SIZE, min_samples=1)
# #                                 )
# #     )
# #     .with_columns(
# #         second_order_soh_estimate=pl.col("soh_estimate").rolling_mean(window_size=WINDOW_SIZE*MULTIPLIER_FOR_SOH*3, min_samples=1)
# #     )
# #     .with_columns(
# #         third_order_soh_estimate=pl.col("second_order_soh_estimate").rolling_mean(window_size=WINDOW_SIZE*MULTIPLIER_FOR_SOH*5, min_samples=1)
# #     )
# #     # .filter(pl.col("soh_estimate").is_not_null())
# #     .with_columns(
# #         # fourth_order_soh_estimate=pl.col("third_order_soh_estimate").
# #         # savgol_filter_soh_estimate=pl.col("soh_estimate").map_batches(lambda x: savgol_filter(x, window_length=WINDOW_SIZE*MULTIPLIER_FOR_SOH*, polyorder=2) if len(x) >= WINDOW_SIZE*MULTIPLIER_FOR_SOH*5+1 else x)
# #         forth_order_soh_estimate=pl.col("third_order_soh_estimate").rolling_mean(window_size=WINDOW_SIZE*MULTIPLIER_FOR_SOH*7, min_samples=1),
# #     )
# #     .with_columns(
# #         fifth_order_soh_estimate=pl.col("forth_order_soh_estimate").rolling_mean( window_size=WINDOW_SIZE*MULTIPLIER_FOR_SOH*11, min_samples=1),
# #     )
# #     .with_columns(
# #         ewm_soh_estimate=pl.col("third_order_soh_estimate").ewm_mean( alpha=0.1, min_samples=1)
# #     )
# #     .sort("day", "sequence_number").filter(pl.col("sequence_number_original").is_between(800_000, 1_400_000 ))  
# #     .unpivot(
# #         on=[ "soh_estimate","soc_nominal", "soc_min", "soc_max", "rolling_soc_min", "rolling_soc_max", "soc","rolling_soc_mean", ],#"second_order_soh_estimate", "third_order_soh_estimate", "forth_order_soh_estimate", "fifth_order_soh_estimate", "ewm_soh_estimate"],   # "soc_nominal", "soc_min", "soc_max", "rolling_soc_min", "rolling_soc_max", "soc",
# #         index=["sequence_number", "sequence_number_original", "cell_temperatures", "current_average", "cell_voltages", "day", "soh"],
# #         variable_name="variable",
# #         value_name="value",
# #     )
# # )

result_wide = (
    result
    .drop_nulls()
    .drop_nans()
    # .filter(pl.col("current_average")!=0)
    .with_columns(
        rolling_soc_mean=((pl.col("soc_max") + pl.col("soc_min")) / 2)
            .rolling_mean(window_size=61, weights=WEIGHTS, min_samples=1),
    )
    # .filter(pl.col("sequence_number").is_between(800_000, 1_500_000))
    # .filter(pl.col("current_average")<=-35)
    # .filter(pl.col("cell_voltages")<=3280)
    .sort("sequence_number")
         .with_columns((pl.col("sequence_number") // 1_440).alias("day"))# limit to first 10 days for better animation
    .with_columns(sequence_number=(pl.col("sequence_number")-pl.col("sequence_number").min()).over("day"),
                  sequence_number_original=pl.col("sequence_number"),  # keep original for hover
    )
    .with_columns(
        soh_estimate_raw= 100 - (pl.col("soc_nominal") - pl.col("rolling_soc_mean"))
    )
    .with_columns(
        soh_estimate_rolling=pl.col("soh_estimate_raw").rolling_mean(window_size=WINDOW_SIZE, min_samples=1),
        # soh_estimate_nulled=pl.when(pl.col("cell_voltages")<3280).then(pl.col("soh_estimate_raw")).otherwise(None)
    )
    .with_columns(
        # soh_estimate_nulled_rolling=pl.col("soh_estimate_nulled").rolling_mean(window_size=WINDOW_SIZE, min_samples=1),
    )
    .with_columns(
        # forward fill — carry last valid SOH reading forward
        soh_estimate_ffill=pl.col("soh_estimate_rolling").forward_fill(),
        # or interpolate linearly between valid readings
        # soh_estimate_interp=pl.col("soh_estimate_nulled_rolling").interpolate(method="linear"),
        
    )

    .unpivot(
        on=["rolling_soc_mean", "soc_min", "soc_max", "soc_nominal", "soc"],#"second_order_soh_estimate", "third_order_soh_estimate", "forth_order_soh_estimate", "fifth_order_soh_estimate", "ewm_soh_estimate"],   # "soc_nominal", "soc_min", "soc_max", "rolling_soc_min", "rolling_soc_max", "soc",
        index=["cell_temperatures", "current_average", "cell_voltages", "sequence_number", "day", "sequence_number_original"],
        variable_name="variable",
        value_name="value",
    )
)

fig = px.line(
    result_wide,
    x="sequence_number",
    y="value",
    color="variable",
    animation_frame="day",
    # color_discrete_map={
    #     "soc_nominal": "blue",
    #     "soc_min":     "red",
    #     "soc_max":     "green",
    #     "rolling_soc_min": "orange",
    #     "rolling_soc_max": "pink",
    #     "soc": "purple",
    #     "soh_estimate": "black",
    #     "rolling_soc_mean": "cyan",
    #     # "second_order_soh_estimate": "gray",
    #     # "soh": "orange",
    # },
    labels={"value": "SOC (%)", "variable": "Series"},
    hover_data=["cell_temperatures", "current_average", "cell_voltages"],
    title="SOC Nominal vs Estimated SOC Min/Max",
)
fig.show()

# # fig=px.line(
# #     result_long.filter(pl.col("variable").is_in(["soh_estimate", "second_order_soh_estimate", "third_order_soh_estimate", "forth_order_soh_estimate", "fifth_order_soh_estimate", "ewm_soh_estimate"])),
# #     x="sequence_number_original",
# #     y="value",
# #     color="variable",
# #     # animation_frame="day",
# #     color_discrete_map={
# #         "soh_estimate": "black",
# #         "second_order_soh_estimate": "blue",
# #         "third_order_soh_estimate": "red",
# #         "ewm_soh_estimate": "green",
# #             "forth_order_soh_estimate": "orange",
# #             "fifth_order_soh_estimate": "purple",
# #             # "soh": "orange",
# #     },
# #     labels={"value": "SOH Estimate (%)", "variable": "Series"},
# #     hover_data=["cell_temperatures", "current_average", "cell_voltages", "soh"],
# #     title="SOH Estimate vs Second Order SOH Estimate",
# # )
# # fig.show()

# # import plotly.express as px
# # import polars as pl

# # variance_df = (
# #     result
# #     .drop_nulls()
# #     .drop_nans()
# #     .with_columns(
# #         rolling_soc_mean=((pl.col("soc_max") + pl.col("soc_min")) / 2)
# #             .rolling_mean(window_size=WINDOW_SIZE, weights=WEIGHTS, min_samples=1),
# #     )
# #     .filter(pl.col("sequence_number").is_between(800_000, 1_400_000))
# #     .sort("sequence_number")
# #     .with_columns(
# #         soc_delta=(pl.col("soc_nominal") - pl.col("rolling_soc_mean")),
# #         # bin voltages to ~10mV buckets for grouping
# #         voltage_bin=((pl.col("cell_voltages") / 10).round(0) * 10).cast(pl.Int32),
# #     )
# #     .filter(pl.col("current_average") < 0)  # discharge only
# #     .group_by("voltage_bin")
# #     .agg(
# #         pl.col("soc_delta").var().alias("soc_delta_variance"),
# #         pl.col("soc_delta").std().alias("soc_delta_std"),
# #         pl.col("soc_delta").mean().alias("soc_delta_mean"),
# #         pl.col("soc_delta").count().alias("n_samples"),
# #     )
# #     .filter(pl.col("n_samples") > 50)  # drop sparse bins
# #     .sort("voltage_bin")
# #     .with_columns(
# #         voltage_v=(pl.col("voltage_bin") / 1000)  # back to volts for axis
# #     )
# # )

# # # ── Plot 1: variance by voltage bin ───────────────────────────────────────
# # fig1 = px.bar(
# #     variance_df,
# #     x="voltage_v",
# #     y="soc_delta_variance",
# #     hover_data=["n_samples", "soc_delta_std", "soc_delta_mean"],
# #     title="soc_delta variance by voltage bin (discharge only) — lower = more trustworthy",
# #     labels={"voltage_v": "Cell voltage (V)", "soc_delta_variance": "Var(soc_delta)"},
# # )
# # fig1.add_vline(x=3.28, line_dash="dash", line_color="red", annotation_text="3.28V threshold")
# # fig1.show()

# # # ── Plot 2: std dev (same units as SOC %) — easier to interpret ───────────
# # fig2 = px.scatter(
# #     variance_df,
# #     x="voltage_v",
# #     y="soc_delta_std",
# #     size="n_samples",
# #     hover_data=["n_samples", "soc_delta_variance"],
# #     title="soc_delta std dev by voltage bin — where does it collapse?",
# #     labels={"voltage_v": "Cell voltage (V)", "soc_delta_std": "Std(soc_delta) (%)"},
# # )
# # fig2.add_vline(x=3.28, line_dash="dash", line_color="red", annotation_text="3.28V threshold")
# # fig2.show()

# # # ── Plot 3: scatter of raw soc_delta coloured by voltage ──────────────────
# # # sample down for plotly performance
# # scatter_df = (
# #     result
# #     .drop_nulls()
# #     .drop_nans()
# #     .with_columns(
# #         rolling_soc_mean=((pl.col("soc_max") + pl.col("soc_min")) / 2)
# #             .rolling_mean(window_size=WINDOW_SIZE, weights=WEIGHTS, min_samples=1),
# #     )
# #     .filter(
# #         pl.col("sequence_number").is_between(800_000, 1_400_000),
# #         pl.col("current_average") < 0,
# #     )
# #     .sort("sequence_number")
# #     .with_columns(
# #         soc_delta=(pl.col("soc_nominal") - pl.col("rolling_soc_mean")),
# #     )
# #     .gather_every(10)  # thin to every 10th sample for plot speed
# # )

# # fig3 = px.scatter(
# #     scatter_df,
# #     x="cell_voltages",
# #     y="soc_delta",
# #     color="cell_temperatures",
# #     opacity=0.3,
# #     title="soc_delta vs cell voltage coloured by temperature",
# #     labels={
# #         "cell_voltages": "Cell voltage (mV)",
# #         "soc_delta": "soc_delta (%)",
# #         "cell_temperatures": "Temp (°C)",
# #     },
# # )
# # fig3.add_vline(x=3280, line_dash="dash", line_color="red", annotation_text="3.28V")
# # fig3.show()
# def process_device(path: str, label: str) -> pl.DataFrame:
#     """Load, preprocess, run SOC estimation, compute SOH estimate."""
#     df = (
#         pl.read_csv(path).drop("charge_capacity")
#         .sort(["device_serial", "sequence_number"])
#         .drop_nulls()
#         .with_columns(
#             pl.col("temperature_battery").cast(float)
#         )
#         .filter(pl.col("temperature_battery")>-30)
#         .with_columns(
#             current_scaled=(pl.col("current") / 277 * 277.0),
#             cell_temperatures=((pl.col("temperature_battery") // 5 * 5).cast(pl.Float64)),
#         )
#         .with_columns(
#             current_average=((pl.col("current_scaled") // 5 * 5).cast(pl.Float64))
#         )
#         .rename({
#             "min_cell_voltage": "cell_voltages",
#             "soc": "soc_nominal",
#         })
#         .with_columns(cell_voltages=pl.col("cell_voltages") * 1000,     soc_nominal=pl.col("soc_nominal").cast(pl.Float64),)
#         .filter(pl.col("cell_voltages") > 0)
#         .select(
#             "device_serial", "sequence_number", "cell_temperatures",
#             "current_average", "cell_voltages", "soc_nominal", "time_stamp",
#         )
#     )

#     result = vectorised_soc_estimation(df_lut, df, soh=100)

#     return (
#         result
#         .drop_nulls()
#         .drop_nans()
#         .with_columns(
#             rolling_soc_mean=((pl.col("soc_max") + pl.col("soc_min")) / 2)
#                 .rolling_mean(window_size=13, weights=WEIGHTS, min_samples=1),
#         )
#         .filter(pl.col("current_average") <=-150)
#         .sort("sequence_number")
#         .with_columns(
#             soh_estimate_raw=100 - (pl.col("soc_nominal") - pl.col("rolling_soc_mean"))
#         )
#         .with_columns(
#             soh_estimate_rolling=pl.col("soh_estimate_raw")
#                 .rolling_mean(window_size=WINDOW_SIZE, min_samples=1),
#             soh_estimate_nulled=pl.when(pl.col("cell_voltages") < 3280)
#                 .then(pl.col("soh_estimate_raw"))
#                 .otherwise(None),
#         )
#         .with_columns(
#             soh_estimate_ffill=pl.col("soh_estimate_rolling").forward_fill(),
#             soh_estimate_interp=pl.col("soh_estimate_nulled")
#                 .rolling_mean(window_size=WINDOW_SIZE, min_samples=1)
#                 .interpolate(method="linear"),
#         )
#         .with_columns(pl.lit(label).alias("source"))  # tag min vs max
#     ).filter(pl.col("soh_estimate_ffill")>=0)


# df_min = process_device("data/test_data/BE19090008_min.csv", "min_cell_voltage")
# df_max = process_device("data/test_data/BE19090008_max.csv", "max_cell_voltage")

# combined = pl.concat([df_min, df_max], how="diagonal_relaxed")
# import plotly.express as px
# # ── Plot ──────────────────────────────────────────────────────────────────────
# plot_df = combined.unpivot(
#     on=[ "soh_estimate_ffill"],
#     index=["time_stamp", "cell_temperatures", "current_average", "cell_voltages", "source"],
#     variable_name="method",
#     value_name="value",
# ).with_columns(
#     # combined color key: "min_cell_voltage — soh_estimate_rolling" etc
#     series=(pl.col("source") + " — " + pl.col("method"))
# )

# fig = px.line(
#     plot_df,
#     x="time_stamp",
#     y="value",
#     color="series",
#     color_discrete_map={
#         # "min_cell_voltage — soh_estimate_rolling": "steelblue",
#         "min_cell_voltage — soh_estimate_ffill":   "blue",
#         # "min_cell_voltage — soh_estimate_interp":  "lightblue",
#         # "max_cell_voltage — soh_estimate_rolling": "crimson",
#         "max_cell_voltage — soh_estimate_ffill":   "red",
#         # "max_cell_voltage — soh_estimate_interp":  "lightsalmon",
#     },
#     hover_data=["cell_temperatures", "current_average", "cell_voltages", "source"],
#     labels={"value": "SOH Estimate (%)", "time_stamp": "Time"},
#     title="SOH Estimate — min vs max cell voltage",
# )
# fig.show()

def process_device(path: str, label: str, scale_capacity:float) -> pl.DataFrame:
    """Load, preprocess, run SOC estimation for all 16 cell voltages."""
    df_raw = (
        pl.read_csv(path, infer_schema_length=10_000).filter(pl.col("sequence_number")>=2_000_000)
        .sort(["device_serial", "sequence_number"])
        .with_columns(
            soc_nominal=pl.col("soc").cast(pl.Float64),
            current=pl.col("current").cast(pl.Float64),
            temperature_battery=pl.col("temperature_battery").cast(pl.Float64),
            **{f"cell_voltage_{i}": pl.col(f"cell_voltage_{i}").cast(pl.Float64) for i in range(1, 17)},
        )
        .with_columns(
            current_scaled=((pl.col("current") /( scale_capacity )) * 277.0),
            cell_temperatures=((pl.col("temperature_battery") // 5 * 5).cast(pl.Float64)),
        )
        .with_columns(
            current_average=((pl.col("current_scaled") // 5 * 5).cast(pl.Float64))
        )
    )

    # Run estimation for each cell voltage independently
    results = []
    for i in range(1, 17):
        cell_col = f"cell_voltage_{i}"
        df_cell = (
            df_raw
            .filter(pl.col(cell_col) > 0)
            .rename({cell_col: "cell_voltages"})
            .with_columns(cell_voltages=pl.col("cell_voltages") * 1000)
            .select(
                "device_serial", "sequence_number", "cell_temperatures",
                "current_average", "cell_voltages", "soc_nominal", "time_stamp",
            )
        )

        est = vectorised_soc_estimation(df_lut, df_cell, soh=100)

        results.append(
            est
            .drop_nulls()
            .drop_nans()
            .with_columns(
                rolling_soc_mean=((pl.col("soc_max") + pl.col("soc_min")) / 2)
                    .rolling_mean(window_size=17, weights=WEIGHTS, min_samples=1),
            )
            .filter(pl.col("current_average") <= -100)
            .sort("sequence_number")
            .with_columns(
                soh_estimate_raw=100 - (pl.col("soc_nominal") - pl.col("rolling_soc_mean"))
            )
            .with_columns(
                soh_estimate_rolling=pl.col("soh_estimate_raw")
                    .rolling_mean(window_size=WINDOW_SIZE, min_samples=1),
                soh_estimate_nulled=pl.when(pl.col("cell_voltages") < 3280)
                    .then(pl.col("soh_estimate_raw"))
                    .otherwise(None),
            )
            .with_columns(
                soh_estimate_ffill=pl.col("soh_estimate_rolling").forward_fill(),
                soh_estimate_interp=pl.col("soh_estimate_nulled")
                    .rolling_mean(window_size=WINDOW_SIZE, min_samples=1)
                    .interpolate(method="linear"),
            )
            .with_columns(
                # pl.lit(label).alias("source"),
                pl.lit(i).alias("cell_number"),
            ).filter(pl.col("soh_estimate_ffill")>0)
        )

    return pl.concat(results, how="diagonal_relaxed")


# ── Run ───────────────────────────────────────────────────────────────────────
# df_all = process_device("data/test_data/BE21000111_cells.csv", "BE21000111")
df_all = process_device("data/test_data/BE19090008_cells.csv", "BE19090008")
# ── Plot: one line per cell, coloured by cell number ─────────────────────────
import plotly.express as px

plot_df = (
    df_all
    .with_columns(pl.col("cell_number").cast(pl.Utf8).alias("cell"))
    # use whichever estimate you want as the y axis
)
# for i in range(0,4_000_000, 500_000):
fig = px.line(
    plot_df.filter(pl.col("sequence_number")>=2_000_000),
    x="time_stamp",
    y="soh_estimate_ffill",
    color="cell",
    color_discrete_sequence=px.colors.qualitative.Dark24,  # 16 distinct colours
    hover_data=["cell_temperatures", "current_average", "cell_voltages", "cell_number"],
    labels={"soh_estimate_ffill": "SOH Estimate (%)", "time_stamp": "Time", "cell": "Cell #"},
    title=f"SOH Estimate per cell",
)
fig.update_traces(opacity=0.8)
fig.show()

# # ── Bonus: facet by cell for side-by-side comparison ─────────────────────────
# fig2 = px.line(
#     plot_df,
#     x="time_stamp",
#     y="soh_estimate_ffill",
#     facet_col="cell",
#     facet_col_wrap=4,           # 4 cells per row → 4 rows of 4
#     color="cell",
#     color_discrete_sequence=px.colors.qualitative.Tab20,
#     labels={"soh_estimate_ffill": "SOH (%)", "time_stamp": "Time"},
#     title="SOH Estimate — all 16 cells (faceted)",
#     height=900,
# )
# fig2.update_traces(opacity=0.8)
# fig2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))  # clean facet labels
# fig2.show()