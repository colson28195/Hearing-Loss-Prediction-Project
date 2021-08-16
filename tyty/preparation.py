def match_pressures(data):
    """
    Extract the rows where the Pressure matches the AdultAbsorbanceData as closely as possible

    Contributors:
    - Tom
    - Daniel
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
    ).reset_index(drop=True)


def split_target(data, feature_columns=[]):
    """
    Splits the target and the relevant features out

    Contributors:
    - Daniel
    """
    target = data["OverallPoF"]
    features = data[[col for col in data.columns if "f(" in col] + feature_columns]
    return features, target
