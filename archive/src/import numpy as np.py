import numpy as np
import polars as pl
from scipy.signal import savgol_filter
import plotly.express as px
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

fig=px.scatter_3d(
    df_lut.filter(pl.col('soh') == 100).sort("current_average").to_pandas(),
    x="dod",
    y="cell_voltages_mean",
    color="cell_temperatures",
    animation_frame="current_average",
    # facet_col="current_average",
    # facet_row="cell_temperatures",
    title="Voltage vs DOD for different SOH, Temperatures and Currents",
)
fig.show()