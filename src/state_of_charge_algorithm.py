import polars as pl
import argparse
import logging
import plotly.express as px
# LOOKUP_CSV = Path(r"C:\Users\adamb\repos\playing_around\P27_368_recalculated_soc_secondly_8_April_cell_voltages.csv")
logging.basicConfig(level=logging.INFO)


def state_of_charge_algorithm(df: pl.DataFrame, column_name_current: str, column_name_voltage: str, lookup_df: pl.DataFrame) -> pl.Series:
    ### The algorithm works like this it finds the closest current  and voltage values in the lookup table and finds it corresponding state of charge however some voltages are considered snetinel values so we're confident in their corresponding state of charge
    ### Therefore we can use that otherwise its a nearest neighbour lookup using binary search to basically find closest voltage and min and max soc values for that current. 
    ### We find closest current bin from lookup and use that for the voltage lookup and then we find the closest voltage bin and use that to find the corresponding soc value.
    
    # Step 1: For each row in input df, find closest current bin in lookup table
    soc_values = []
    
    for row in df.iter_rows(named=True):
        current = row[column_name_current]
        voltage = row[column_name_voltage]
        
        # Find closest current bin in lookup table
        lookup_df_with_current_diff = lookup_df.with_columns(
            current_diff=pl.col("current_bin").sub(current).abs()
        )
        closest_current_row = lookup_df_with_current_diff.filter(
            pl.col("current_diff") == pl.col("current_diff").min()
        )
        
        if closest_current_row.height == 0:
            soc_values.append(None)
            continue
        
        # Filter lookup by this current bin to find closest voltage
        current_bin_value = closest_current_row.select("current_bin").item()
        lookup_filtered_by_current = lookup_df.filter(
            pl.col("current_bin") == current_bin_value
        )
        
        # Find closest voltage to voltage_median
        lookup_with_voltage_diff = lookup_filtered_by_current.with_columns(
            voltage_diff=pl.col("voltage_median").sub(voltage).abs()
        )
        closest_voltage_row = lookup_with_voltage_diff.filter(
            pl.col("voltage_diff") == pl.col("voltage_diff").min()
        )
        
        if closest_voltage_row.height == 0:
            soc_values.append(None)
            continue
        
        # Get the corresponding state of charge value
        soc = closest_voltage_row.select("state_of_charge_recorded").item()
        soc_values.append(soc)
    
    return pl.Series("state_of_charge_estimated", soc_values) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="State of Charge Algorithm")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument(
        "--lookup", type=str, required=True, help="Path to lookup CSV file"
    )
    parser.add_argument(
        "--current_column",
        type=str,
        required=True,
        help="Name of the current column in the input CSV",
    )
    parser.add_argument(
        "--voltage_column",
        type=str,
        required=True,
        help="Name of the voltage column in the input CSV",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output CSV file"
    )
    args = parser.parse_args()
    
    logging.info(f"Reading input data from {args.input}")
    df = pl.read_csv(args.input)
    
    logging.info(f"Reading lookup data from {args.lookup}")
    lookup_df = pl.read_csv(args.lookup)
    
    logging.info("Calculating State of Charge")
    soc_series = state_of_charge_algorithm(
        df, args.current_column, args.voltage_column, lookup_df
    )
    df = df.with_columns(soc_series.alias("state_of_charge_estimated"))
    
    logging.info(f"Writing output data to {args.output}")
    df.write_csv(args.output)
    
    out = df.unpivot(
        [
            "state_of_charge_estimated",
        ],
        index="timestamp",
        variable_name="metric",
        value_name="soc_value",
    )
