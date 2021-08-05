from tyty import processing


def run_pipeline():
    """
    Runs the processing pipeline

    Contributors:
    - Daniel
    """
    data = processing.import_data("data")
    demo = processing.import_data("demographics")

    processing.clean_empty(data)
    processing.clean_empty(demo)

    processing.remove_columns(demo)
    processing.remove_columns(data, ["Gender"])

    combined = processing.combine_data(data, demo)

    return combined


def split_target(data, feature_columns=[]):
    """
    Splits the target and the relevant features out

    Contributors:
    - Daniel
    """
    target = data["OverallPoF"]
    features = data[[col for col in data.columns if "f(" in col] + feature_columns]
    return features, target
