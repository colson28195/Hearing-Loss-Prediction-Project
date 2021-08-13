def split_target(data, feature_columns=[]):
    """
    Splits the target and the relevant features out

    Contributors:
    - Daniel
    """
    target = data["OverallPoF"]
    features = data[[col for col in data.columns if "f(" in col] + feature_columns]
    return features, target
