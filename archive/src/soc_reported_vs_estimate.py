    """
    soc_reported_vs_estimate.py
    ───────────────────────────
    Fetch the full history of a single battery from Athena, run the per-cell
    SOC estimation and display an animated Plotly chart with:
        • soc_nominal            — one line, reported BMS SOC (same for all cells)
        • rolling_soc_mean_cell_N — one line per cell, LUT-based rolling SOC estimate

    Animation is by day (sequence_number // 1440), x-axis is the within-day
    sequence offset.

    Usage
    -----
    python soc_reported_vs_estimate.py \
        --serial      BE20000157 \
        --capacity    277 \
        --num_cells   16 \
        --lut_path    data/voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet \
        --lut_soh     100 \
        --aws_profile infra_dev \
        --output_csv  soc_comparison.csv   # optional
    """

    import argparse
    import boto3
    import awswrangler as wr
    import polars as pl
    import numpy as np
    import plotly.express as px

    # ── Constants ─────────────────────────────────────────────────────────────────
    WINDOW_SIZE_SOC = 13
    LUT_REFERENCE_CAPACITY = 277.0


    # ── Weights ───────────────────────────────────────────────────────────────────

    def make_weights(window_size: int, decay: float = 0.7) -> list[float]:
        x = np.arange(window_size)
        weights = np.exp(decay * (x - (window_size - 1)))
        return (weights / weights.sum()).tolist()


    WEIGHTS = make_weights(WINDOW_SIZE_SOC, decay=0.7)


    # ── LUT ───────────────────────────────────────────────────────────────────────

    def load_lut(lut_path: str) -> pl.DataFrame:
        return (
        pl.scan_parquet(
            r"data\\voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet",
        )
        .with_columns(
            (pl.col('q_removed_ah') / ( 277) * 100)
            .round(0)
            .alias('dod')
        )
        .filter(pl.col("dod").is_between(0, 100, closed="both"))
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
        .filter(pl.col("soh").is_between(98,102, closed="both"))
        .group_by(["cell_temperatures","current_average", "dod"])
        .agg(
            ((pl.col('cell_voltages_mean').mean()//2)*2).round(0).alias('cell_voltages_mean')
        )
        .with_columns(
            pl.col("dod").count().over(["cell_temperatures","current_average"]).alias("dod_count")
        )
        .filter(pl.col("dod_count")>=90)
        .with_columns(
            # pl.when((pl.col('cell_voltages_diff').abs() >3.5) | (pl.col("dod")==0)).then(pl.col('cell_voltages_mean')).otherwise(-1).alias('cell_voltages_mean')
            pl.col("cell_voltages_mean").diff().over(
                partition_by=["current_average","cell_temperatures"], order_by="dod").alias('cell_voltages_diff')
        )
        .with_columns(
            pl.when(pl.col("cell_voltages_diff").abs()<1.5).then(True).otherwise(False).alias("plateau_mask")
        )
        .drop("dod_count")
        .collect()
    )


    # ── SOC estimation (vectorised) ───────────────────────────────────────────────

    import numpy as np
    import polars as pl


    def vectorised_soc_estimation(
        df_lut: pl.DataFrame,
        df: pl.DataFrame,
        resistance_ohm: float =0.00025,
        soh: float = 100,
        add_debug_cols: bool = False,
    ) -> pl.DataFrame:
        """
        Estimate SOC using nearest LUT temperature/current bins, with current-bin
        voltage compensation using estimated resistance.

        Assumptions:
        - df["cell_voltages"] is in mV
        - df["current_average"] is in A
        - df["cell_temperatures"] is in degC
        - resistance_ohm is in Ohms
        - LUT voltage column "cell_voltages_mean" is in mV
        - LUT DOD is 0 to 100

        Compensation:
            V_adj = V_measured - R_mOhm * (I_actual - I_lut_bin)

        where:
            R_mOhm = resistance_ohm * 1000
        """

        lut = df_lut  # optionally: df_lut.filter(pl.col("soh") == soh)

        # ---------------------------------------------------------------------
        # LUT arrays
        # ---------------------------------------------------------------------
        lut_temp = lut["cell_temperatures"].to_numpy()
        lut_curr = lut["current_average"].to_numpy()
        lut_volt = lut["cell_voltages_mean"].to_numpy()
        lut_dod = lut["dod"].to_numpy()

        # ---------------------------------------------------------------------
        # Data arrays
        # ---------------------------------------------------------------------
        temps_actual = df["cell_temperatures"].to_numpy()
        currents_actual = df["current_average"].to_numpy()
        voltages_actual = df["cell_voltages"].to_numpy()

        n = len(df)

        soc_min_arr = np.full(n, np.nan)
        soc_max_arr = np.full(n, np.nan)
        plateau_mask_arr = np.full(n, False)
        matched_temp_arr = np.full(n, np.nan)
        matched_curr_arr = np.full(n, np.nan)
        voltage_adj_arr = np.full(n, np.nan)

        resistance_mohm = resistance_ohm * 1000.0

        # ---------------------------------------------------------------------
        # Unique LUT temperatures
        # ---------------------------------------------------------------------
        unique_lut_temps = np.unique(lut_temp)

        # ---------------------------------------------------------------------
        # Step 1: Find nearest LUT temperature for every row
        # ---------------------------------------------------------------------
        for i in range(n):
            t_actual = temps_actual[i]

            if np.isnan(t_actual):
                # Fallback if measured temperature is missing
                # Use median available LUT temperature
                matched_temp = np.median(unique_lut_temps)
            else:
                matched_temp = unique_lut_temps[
                    np.argmin(np.abs(unique_lut_temps - t_actual))
                ]

            matched_temp_arr[i] = matched_temp

        # ---------------------------------------------------------------------
        # Step 2: For each matched temperature, find nearest current per row
        # ---------------------------------------------------------------------
        unique_matched_temps = np.unique(matched_temp_arr[~np.isnan(matched_temp_arr)])

        for temp_bin in unique_matched_temps:
            temp_lut_mask = lut_temp == temp_bin

            if not temp_lut_mask.any():
                continue

            available_currents = np.unique(lut_curr[temp_lut_mask])

            data_temp_mask = matched_temp_arr == temp_bin
            data_indices = np.where(data_temp_mask)[0]

            for i in data_indices:
                i_actual = currents_actual[i]

                if np.isnan(i_actual):
                    # Fallback if measured current is missing
                    # Use current bin closest to zero if available
                    curr_bin = available_currents[
                        np.argmin(np.abs(available_currents))
                    ]
                else:
                    curr_bin = available_currents[
                        np.argmin(np.abs(available_currents - i_actual))
                    ]

                matched_curr_arr[i] = curr_bin

                # IR compensation from actual current to LUT current bin
                voltage_adj_arr[i] = (
                    voltages_actual[i]
                    - resistance_mohm * (i_actual - curr_bin)
                )

        # ---------------------------------------------------------------------
        # Step 3: Evaluate SOC for each matched temp/current LUT bin
        # ---------------------------------------------------------------------
        unique_keys = np.unique(
            np.stack([matched_temp_arr, matched_curr_arr], axis=1),
            axis=0,
        )

        for temp_bin, curr_bin in unique_keys:
            if np.isnan(temp_bin) or np.isnan(curr_bin):
                continue

            lut_mask = (
                (lut_temp == temp_bin)
                & (lut_curr == curr_bin)
            )

            if not lut_mask.any():
                continue

            bin_dod = lut_dod[lut_mask]
            bin_volt = lut_volt[lut_mask]

            valid = np.isfinite(bin_volt) & np.isfinite(bin_dod) & (bin_volt >= 0)

            bin_dod = bin_dod[valid]
            bin_volt = bin_volt[valid]

            if len(bin_dod) == 0:
                continue

            # Sort by DOD: usually voltage decreases as DOD increases
            sort_idx = np.argsort(bin_dod)
            bin_dod = bin_dod[sort_idx]
            bin_volt = bin_volt[sort_idx]

            v_max = bin_volt[0]
            v_min = bin_volt[-1]

            data_mask = (
                (matched_temp_arr == temp_bin)
                & (matched_curr_arr == curr_bin)
            )

            v_query = voltage_adj_arr[data_mask]

            # searchsorted wants increasing x, so flip voltage curve
            flipped_volt = bin_volt[::-1]
            flipped_dod = bin_dod[::-1]

            idx_right = np.searchsorted(
                flipped_volt,
                v_query,
                side="left",
            ).clip(0, len(flipped_volt) - 1)

            idx_left = (idx_right - 1).clip(0, len(flipped_volt) - 1)

            dod_at_left = flipped_dod[idx_left]
            dod_at_right = flipped_dod[idx_right]

            dod_min_v = np.minimum(dod_at_left, dod_at_right).clip(0, 100)
            dod_max_v = np.maximum(dod_at_left, dod_at_right).clip(0, 100)

            above = v_query > v_max
            below = v_query < v_min

            # Above highest voltage means near full: DOD = 0, SOC = 100
            dod_min_v = np.where(above, 0, np.where(below, 100, dod_min_v))
            dod_max_v = np.where(above, 0, np.where(below, 100, dod_max_v))

            # Below lowest voltage means empty: DOD = 100, SOC = 0
            soc_min_arr[data_mask] = (100 - dod_max_v).clip(0, 100)
            soc_max_arr[data_mask] = (100 - dod_min_v).clip(0, 100)

        output_cols = [
            pl.Series("soc_min", soc_min_arr, dtype=pl.Float64),
            pl.Series("soc_max", soc_max_arr, dtype=pl.Float64),
        ]

        if add_debug_cols:
            output_cols.extend([
                pl.Series("matched_lut_temperature", matched_temp_arr, dtype=pl.Float64),
                pl.Series("matched_lut_current", matched_curr_arr, dtype=pl.Float64),
                pl.Series("cell_voltage_compensated", voltage_adj_arr, dtype=pl.Float64),
            ])

        return df.with_columns(output_cols)


    # ── Athena fetch ──────────────────────────────────────────────────────────────

    def build_query(serial: str, num_cells: int, capacity_ah: float) -> str:
        capacity_coulombs = capacity_ah * 3600.0
        cell_voltage_cols = "\n".join(
            f"    cell_data.cell_voltage[{i}] AS cell_voltage_{i},"
            for i in range(1, num_cells + 1)
        )
        return f"""
    SELECT DISTINCT
        device_serial,
        sequence_number,
        cast(time_stamp as timestamp(3)) as time_stamp,
        current,
        charge_in, 
        charge_out, 
    {cell_voltage_cols}
        ( (({capacity_coulombs} - (charge_top - (charge_in - charge_out))) / {capacity_coulombs}) * 100) as soc,
        temperature_battery
    FROM analytics.batteries_v3
    WHERE device_serial = '{serial}'
    AND cell_quantity = {num_cells}
    AND time_stamp >= timestamp '2026-04-01'
    ORDER BY sequence_number ASC
    """


    def fetch_data(serial: str, num_cells: int, capacity_ah: float, aws_profile: str) -> pl.DataFrame:
        boto3.setup_default_session(profile_name=aws_profile)
        sql = build_query(serial, num_cells, capacity_ah)
        print(f"Fetching data for {serial} from Athena …")
        pd_df = wr.athena.read_sql_query(
            sql=sql, database="scratch", workgroup="data_engineering"
        )
        return pl.from_pandas(pd_df)


    # ── Prepare base DataFrame ────────────────────────────────────────────────────

    def prepare_base(df_raw: pl.DataFrame, capacity_ah: float, num_cells: int) -> pl.DataFrame:
        return (
            df_raw
            .sort(["device_serial", "sequence_number"])
            .with_columns(
                soc_nominal=pl.col("soc").cast(pl.Float64),
                # current=pl.col("current").cast(pl.Float64),
                current_average=(
                    (pl.col("charge_in").diff() - pl.col("charge_out").diff()) / 60.0
                )
                .over("device_serial", order_by='sequence_number'),
                temperature_battery=pl.col("temperature_battery").cast(pl.Float64),
                **{
                    f"cell_voltage_{i}": pl.col(f"cell_voltage_{i}").cast(pl.Float64)
                    for i in range(1, num_cells + 1)
                },
            )
            .with_columns(
                current_scaled=(pl.col("current") / capacity_ah * LUT_REFERENCE_CAPACITY),
                cell_temperatures=((pl.col("temperature_battery") // 5 * 5).cast(pl.Float64)),
            )
            .with_columns(
                current_average=((pl.col("current_scaled") // 5 * 5).cast(pl.Float64))
            )
            # .filter((pl.col("current_average")<=-5))
        )


    # ── Build wide result: one rolling_soc_mean column per cell ──────────────────

    def build_wide_result(
        df_base: pl.DataFrame,
        df_lut: pl.DataFrame,
        num_cells: int,
        lut_soh: float,
    ) -> pl.DataFrame:
        """
        Returns a wide DataFrame indexed by sequence_number with columns:
            sequence_number, sequence_number_original, day,
            soc_nominal,
            soc_cell_1, soc_cell_2, … soc_cell_N
        """
        # Start from the base sequences (use soc_nominal from any cell — it's the same)
        # We'll join each cell's rolling estimate in as a new column.
        seq_base = (
            df_base
            .select("sequence_number", "soc_nominal", "current_average")
            .unique("sequence_number")
            .sort("sequence_number")
        )

        for i in range(1, num_cells + 1):
            cell_col = f"cell_voltage_{i}"

            df_cell = (
                df_base
                .filter(pl.col(cell_col) > 0)
                .rename({cell_col: "cell_voltages"})
                .with_columns(cell_voltages=pl.col("cell_voltages") * 1000)
                .select(
                    "sequence_number", "cell_temperatures", "current_average",
                    "cell_voltages", "soc_nominal",
                )
            )

            est = vectorised_soc_estimation(df_lut, df_cell, soh=lut_soh)

            cell_estimate = (
                est
                .drop_nulls()
                .drop_nans()
                .sort("sequence_number")
                .with_columns(
                    rolling_soc_mean=(
                        ((pl.col("soc_max") + pl.col("soc_min")) / 2)
                        .rolling_mean(window_size=WINDOW_SIZE_SOC,min_samples=WINDOW_SIZE_SOC-1)
                    ),
                )
                # .with_columns(
                #     soc_delta=pl.col("rolling_soc_mean") - pl.col("soc_nominal")
                # )
                .select(
                    "sequence_number",
                    # "current_average",
                    pl.col("rolling_soc_mean").alias(f"soc_cell_{i}"),

                    # pl.col("soc_delta").alias(f"soc_delta_{i}")
                )
            )

            seq_base = seq_base.join(cell_estimate, on="sequence_number", how="left")
            print(f"  Cell {i}/{num_cells} done", end="\r", flush=True)

        # print()

        # Add day + within-day offset
        return (
            seq_base
            .sort("sequence_number")
            .with_columns(
                day=(pl.col("sequence_number") //( 1_440*7)),
            )
            .with_columns(
                sequence_number_original=pl.col("sequence_number"),
                sequence_number=(
                    pl.col("sequence_number") - pl.col("sequence_number").min().over("day")
                ),
            )
        )


    # ── Main ──────────────────────────────────────────────────────────────────────

    def main():
        parser = argparse.ArgumentParser(
            description="Animated day-by-day chart: reported SOC vs per-cell rolling LUT SOC estimate."
        )
        parser.add_argument("--serial",      required=True,  help="Device serial, e.g. BE20000157")
        parser.add_argument("--capacity",    required=True,  type=float, help="Battery capacity in Ah")
        parser.add_argument("--num_cells",   required=True,  type=int,   help="Number of cells")
        parser.add_argument("--lut_path",    required=True,  help="Path to the LUT parquet file")
        parser.add_argument("--lut_soh",     default=100.0,  type=float, help="LUT SOH slice to use (default: 100)")
        parser.add_argument("--aws_profile", default="infra_dev", help="AWS profile name (default: infra_dev)")
        parser.add_argument("--output_csv",  default=None,   help="Optional path to save wide-format CSV")
        args = parser.parse_args()

        print(f"Loading LUT from {args.lut_path} …")
        df_lut = load_lut(args.lut_path)

        df_raw = fetch_data(
            serial=args.serial,
            num_cells=args.num_cells,
            capacity_ah=args.capacity,
            aws_profile=args.aws_profile,
        )
        print(f"Fetched {len(df_raw):,} rows for {args.serial}")

        df_base = prepare_base(df_raw, args.capacity, args.num_cells)

        print("Running SOC estimation per cell …")
        df_wide = build_wide_result(df_base, df_lut, args.num_cells, args.lut_soh)
        print(f"Wide result: {len(df_wide):,} rows")

        if args.output_csv:
            df_wide.write_csv(args.output_csv)
            print(f"Saved to {args.output_csv}")

        # Unpivot for px.line: soc_nominal + one column per cell → long format
        cell_cols = [f"soc_cell_{i}" for i in range(1, args.num_cells + 1)]
        result_long = df_wide.unpivot(
            on=["soc_nominal"] + cell_cols,
            index=["sequence_number", "sequence_number_original", "day", "current_average"],
            variable_name="variable",
            value_name="value",
        )

        fig = px.line(
            result_long.to_pandas(),
            x="sequence_number",
            y="value",
            color="variable",
            animation_frame="day",
            hover_data=["sequence_number_original", "current_average"],
            labels={
                "value": "SOC (%)",
                "variable": "Series",
                "sequence_number": "Sequence (within day)",
                "current_average": "Current (A)"
            },
            title=(
                f"SOC Reported vs Per-Cell LUT Estimate — {args.serial} "
                f"({args.num_cells} cells · {args.capacity} Ah · LUT SOH={args.lut_soh}%)"
            ),
            color_discrete_map={"soc_nominal": "black"},
        )
        fig.update_traces(
            selector=dict(name="soc_nominal"),
            line=dict(width=2.5),
            opacity=1.0,
        )
        fig.update_traces(
            selector=lambda t: t.name != "soc_nominal",
            opacity=0.7,
            line=dict(width=1),
        )
        fig.update_yaxes(range=[-5, 105])
        fig.show()
        import plotly.graph_objects as go

        df_day = result_long.to_pandas()

        # Build figure from px.line first, then add secondary axis trace
        fig = px.line(
            df_day,
            x="sequence_number",
            y="value",
            color="variable",
            animation_frame="day",
            hover_data=["sequence_number_original"],
            labels={
                "value": "SOC (%)",
                "variable": "Series",
                "current_average": "Current (A)",
                "sequence_number": "Sequence (within day)",
            },
            title=(
                f"SOC Reported vs Per-Cell LUT Estimate — {args.serial} "
                f"({args.num_cells} cells · {args.capacity} Ah · LUT SOH={args.lut_soh}%)"
            ),
            color_discrete_map={"soc_nominal": "black"},
        )

        fig.update_traces(
            selector=dict(name="soc_nominal"),
            line=dict(width=2.5),
            opacity=1.0,
        )
        fig.update_traces(
            selector=lambda t: t.name != "soc_nominal",
            opacity=0.7,
            line=dict(width=1),
        )

        # Add current trace on secondary axis for each animation frame
        days = df_day["day"].unique()
        first_day = sorted(days)[0]

        # Add the current trace (visible on first frame)
        for i, day in enumerate(sorted(days)):
            day_df = result_long.filter(pl.col("day") == day).to_pandas()
            day_df = day_df.drop_duplicates("sequence_number")

            fig.add_trace(
                go.Scatter(
                    x=day_df["sequence_number"],
                    y=day_df["current_average"],
                    name="Current (A)",
                    yaxis="y2",
                    line=dict(color="royalblue", width=1, dash="dot"),
                    opacity=0.6,
                    visible=(i == 0),  # only first frame visible initially
                )
            )

        # Wire up current trace visibility to animation frames
        current_trace_indices = list(range(len(fig.data) - len(days), len(fig.data)))
        soc_trace_count = len(fig.data) - len(days)

        for i, frame in enumerate(fig.frames):
            frame.traces = list(frame.traces) + [current_trace_indices[i]]
            day_df = result_long.filter(pl.col("day") == sorted(days)[i]).to_pandas()
            day_df = day_df.drop_duplicates("sequence_number")

            frame.data = list(frame.data) + [
                go.Scatter(
                    x=day_df["sequence_number"],
                    y=day_df["current_average"],
                    name="Current (A)",
                    yaxis="y2",
                    line=dict(color="royalblue", width=1, dash="dot"),
                    opacity=0.6,
                )
            ]

        fig.update_layout(
            yaxis=dict(range=[-5, 105], title="SOC (%)"),
            yaxis2=dict(
                title="Current (A)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(x=1.08),
        )

        fig.show()


    if __name__ == "__main__":
        main()
