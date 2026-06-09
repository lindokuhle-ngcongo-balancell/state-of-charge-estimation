import polars as pl

results = pl.read_csv("results/P24-000104_soc_ocv.csv")
nominal = pl.read_csv("data/P24-000104.csv").select(["sequence_number", "soc"]).rename({"soc": "soc_nominal_raw"})

print("=== Mode distribution ===")
modes = results.group_by("mode").agg(pl.len().alias("count")).sort("count", descending=True)
for row in modes.iter_rows(named=True):
    pct = row["count"] / results.height * 100
    print(f"  {row['mode']:<8} {row['count']:>8,}  ({pct:.1f}%)")

r1 = results.filter(pl.col("cell_idx") == 1).join(nominal, on="sequence_number", how="left")
lut = r1.filter(pl.col("mode") == "LUT").with_columns([
    pl.col("soc_hybrid").cast(pl.Float64),
    pl.col("soc_nominal").cast(pl.Float64),
    pl.col("soc_nominal_raw").cast(pl.Float64),
])
diff = (lut["soc_hybrid"] - lut["soc_nominal_raw"]).abs()
print()
print("=== LUT vs BMS SoC error (cell 1) ===")
print(f"  mean   |error|: {diff.mean():.2f}%")
print(f"  median |error|: {diff.median():.2f}%")
print(f"  max    |error|: {diff.max():.2f}%")
print()
print("=== Sample LUT rows (cell 1, first 8) ===")
sample = lut.select(["sequence_number", "time_stamp", "soc_hybrid", "soc_nominal_raw"]).head(8)
for row in sample.iter_rows(named=True):
    ts = str(row["time_stamp"])[:16]
    print(f"  seq={row['sequence_number']}  {ts}  OCV={row['soc_hybrid']:.1f}%  BMS={row['soc_nominal_raw']:.1f}%")
