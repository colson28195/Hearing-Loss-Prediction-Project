import pandas as pd

import daniel.processing as ps

pd.set_option("display.max_rows", None)

result = ps.run_pipeline()

features, target = ps.split_target(result, ["Gender", "EarSide"])

# print(features.head())
# print(len(result))

PressureMatching = result[["Subject", "AdultAbsorbanceData", "Pressure", "EarSide"]]

PressureMatching["abs"] = (
    PressureMatching["AdultAbsorbanceData"] - PressureMatching["Pressure"]
).abs()

# print(PressureMatching.head())

# x = PressureMatching.groupby(["Subject", "EarSide"]).min()

# print(x)

# def Matching(data):
# return "match" if data["AdultAbsorbanceData"] == data["Pressure"] else "mismatch"

# result["is_pressure_match"] = result.apply(Matching,axis = 1)

# print(result)

# select_row = result.loc[result["is_pressure_match"] == "match"]

# print(select_row)

x = PressureMatching[
    PressureMatching["AdultAbsorbanceData"]
    .sub(PressureMatching["Pressure"])
    .groupby(PressureMatching["Subject"])
    .transform(lambda x: x == abs(x).min())
]

print(x)
