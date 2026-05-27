import polars as pl
import plotly.express as px
df = pl.read_parquet(r"data\voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet")
fig=px.scatter_3d(
    df.sort("current_average"), 
    x='q_removed_ah',
    z='cell_voltages_median',
    y='soh',
    color='cell_temperatures',
    animation_frame='current_average',
)
fig.show()