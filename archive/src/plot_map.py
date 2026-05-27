import numpy as np
import polars as pl
from scipy.signal import savgol_filter
import plotly.express as px

WINDOW_SIZE = 121
MULTIPLIER_FOR_SOH = 7
df_lut = (
    pl.scan_parquet(
        r"data\\voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet",
    )
    .with_columns((pl.col("q_removed_ah") / (277) * 100).round(0).alias("dod"))
    .filter(pl.col("dod").is_between(0, 100, closed="both"))
    .group_by(["soh", "cell_temperatures", "current_average", "dod"])
    .agg((pl.col("cell_voltages_mean").mean()).round(0).alias("cell_voltages_mean"))
    .sort(["soh", "cell_temperatures", "current_average", "dod"])
    .with_columns(
        pl.col("cell_voltages_mean")
        .diff()
        .over(
            partition_by=["soh", "cell_temperatures", "current_average"], order_by="dod"
        )
        .alias("cell_voltages_diff")
    )
    .filter(pl.col("soh").is_between(98, 102, closed="both"))
    .group_by(["cell_temperatures", "current_average", "dod"])
    .agg((pl.col("cell_voltages_mean").mean()).round(0).alias("cell_voltages_mean"))
    .with_columns(
        pl.col("dod")
        .count()
        .over(["cell_temperatures", "current_average"])
        .alias("dod_count")
    )
    .filter(pl.col("dod_count") >= 90)
    # .with_columns(
    #     pl.when((pl.col('cell_voltages_diff').abs() >3.5) | (pl.col("dod")==0)).then(pl.col('cell_voltages_mean')).otherwise(-1).alias('cell_voltages_mean')
    # )
    .with_columns(
        pl.col("cell_voltages_mean")
        .diff()
        .over(
            partition_by=[
                "current_average",
                "cell_temperatures",
            ],
            order_by="dod",
            descending=True,
        )
        .alias("cell_voltages_diff"),
        pl.col("dod")
        .diff()
        .over(
            partition_by=[
                "current_average",
                "cell_temperatures",
            ],
            order_by="dod",
            descending=False,
        )
        .alias("dod_diff"),
    )
    .filter(pl.col("dod_diff").abs() == 1)
    .filter(pl.col("cell_voltages_diff").abs()>=5)
    .collect()
)
df_lut.write_csv("test.csv")

print(df_lut)
fig = px.scatter(
    df_lut.sort("current_average").to_pandas(),
    x="dod",
    y="cell_voltages_mean",
    color="cell_temperatures",
    animation_frame="current_average",
    # facet_col="current_average",
    # facet_row="cell_temperatures",
    title="Voltage vs DOD for different SOH, Temperatures and Currents",
)
fig.show()
