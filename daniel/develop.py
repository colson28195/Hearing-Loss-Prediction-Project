import pandas as pd

from tyty import pipeline
import daniel.processing as ps

pd.set_option("display.max_rows", None)

result = pipeline.run_pipeline()

features, target = ps.split_target(result, ["Gender", "EarSide"])

# print(features.head())
# print(len(result))


# PressureMatching = result[["Subject", "AdultAbsorbanceData", "Pressure", "EarSide"]]

# PressureMatching["abs"] = abs(
#     PressureMatching["AdultAbsorbanceData"] - PressureMatching["Pressure"]
# )

# x = PressureMatching[
#     PressureMatching["AdultAbsorbanceData"]
#     .sub(PressureMatching["Pressure"])
#     .groupby(PressureMatching["Subject"])
#     .transform(lambda x: x == abs(x).min())
# ]


# print(test.head())
# print(t2.head())
# print(t3.head())


def match_pressures(data):
    """
    Extract the rows where the Pressure matches the AdultAbsorbanceData as closely as possible
    """
    cols = [
        col
        for col in data.columns
        if col not in ["Subject", "AdultAbsorbanceData", "Pressure", "EarSide"]
    ]
    smaller = data.drop(columns=cols)
    smaller["abs"] = abs(smaller["AdultAbsorbanceData"] - smaller["Pressure"])
    grouped = (
        smaller.loc[smaller.groupby(["Subject", "EarSide"])["abs"].idxmin()]
        .reset_index(drop=True)
        .drop(columns="abs")
    )
    return grouped.merge(
        data, on=["Subject", "EarSide", "AdultAbsorbanceData", "Pressure"], how="inner"
    )


small = match_pressures(result)
print(result.shape)
print(small.shape)
