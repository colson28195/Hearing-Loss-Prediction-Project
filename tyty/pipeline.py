from tyty import processing, preparation


def processing_pipeline():
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


def prep_pipeline(data, feature_columns=[], pressure_match=False):
    """
    Runs the preparation pipeline

    Contributors:
    - Daniel
    """
    if pressure_match:
        data = preparation.match_pressures(data)

    features, target = preparation.split_target(data, feature_columns)

    return features, target
