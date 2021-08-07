import pandas as pd

from tyty import pipeline
import daniel.processing as ps

pd.set_option("display.max_rows", None)

result = pipeline.run_pipeline()

features, target = ps.split_target(result, ["Gender", "EarSide"])

# print(features.head())
# print(len(result))

PressureMatching = result[["Subject", "AdultAbsorbanceData", "Pressure", "EarSide"]]

PressureMatching["abs"] = (
    PressureMatching["AdultAbsorbanceData"] - PressureMatching["Pressure"]
).abs()

# print(PressureMatching.head())

x = PressureMatching.groupby(["Subject", "EarSide"]).min()

print(x)
