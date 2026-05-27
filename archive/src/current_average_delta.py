import polars as pl
import plotly.express as px

df_P34_000350=pl.read_csv("data/P34-000350.csv").sort('sequence_number').with_columns(
    current_average=(pl.col("charge_in").diff().fill_null(0)-pl.col("charge_out").diff().fill_null(0))/60
).select('device_serial', 'time_stamp', 'current_average', "soc")
df_P34_000351=pl.read_csv("data/P34-000351.csv").sort('sequence_number').with_columns(
    current_average=(pl.col("charge_in").diff().fill_null(0)-pl.col("charge_out").diff().fill_null(0))/60
).select('device_serial', 'time_stamp', 'current_average', "soc")
df_P34_000352=pl.read_csv("data/P34-000352.csv").sort('sequence_number').with_columns(
    current_average=(pl.col("charge_in").diff().fill_null(0)-pl.col("charge_out").diff().fill_null(0))/60
).select('device_serial', 'time_stamp', 'current_average', "soc")
df_P34_000353=pl.read_csv("data/P34-000353.csv").sort('sequence_number').with_columns(
    current_average=(pl.col("charge_in").diff().fill_null(0)-pl.col("charge_out").diff().fill_null(0))/60
).select('device_serial', 'time_stamp', 'current_average', "soc")
df_merged=pl.concat([df_P34_000350, df_P34_000351, df_P34_000352, df_P34_000353])
df_merged=df_merged.with_columns(
    cumulative_current_average=(pl.col("current_average")/60).cum_sum().over('device_serial', order_by='time_stamp', descending=False)
)
df_unpivoted_current_average=df_merged.unpivot(index=['device_serial', 'time_stamp'], on=['current_average'], variable_name='variable', value_name='value')
df_unpivoted_cumulative_current_average=df_merged.unpivot(index=['device_serial', 'time_stamp'], on=['cumulative_current_average'], variable_name='variable', value_name='value')
df_unpivoted_soc=df_merged.unpivot(index=['device_serial', 'time_stamp'], on=['soc'], variable_name='variable', value_name='value')
fig = px.scatter(df_unpivoted_current_average.to_pandas(), x='time_stamp', y='value', color='device_serial', title='Current Average Over Time')
fig.add_scatter(x=df_unpivoted_soc.to_pandas()['time_stamp'], y=df_unpivoted_soc.to_pandas()['value'], mode='markers', name='SOC')
fig.show()