from sklearn.model_selection import train_test_split
from tyty import processing, preparation, modelling


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


def prep_pipeline(data, feature_columns=[], pressure_match=False, test_size=0.2):
    """
    Runs the preparation pipeline

    Contributors:
    - Daniel
    """
    if pressure_match:
        data = preparation.match_pressures(data)

    features, target = preparation.split_target(data, feature_columns)

    train_data, test_data, train_labels, test_labels = train_test_split(
        features, target, test_size=test_size, random_state=24
    )

    return train_data, test_data, train_labels, test_labels


def full_pipeline(feature_columns=[], pressure_match=False, test_size=0.2):
    """
    Runs the processing and preparation pipelines to return data ready for modelling

    Contributors:
    - Daniel
    """
    result = processing_pipeline()
    train_data, test_data, train_labels, test_labels = prep_pipeline(
        result,
        feature_columns=feature_columns,
        pressure_match=pressure_match,
        test_size=test_size,
    )
    return train_data, test_data, train_labels, test_labels


def modelling_pipeline(model, train_data, train_labels, test_data):
    """
    Trains the model and performs prediction on the unseen test set
    """
    trained_model = modelling.training(model, train_data, train_labels)
    train_predictions = modelling.predicting(trained_model, train_data)
    test_predictions = modelling.predicting(trained_model, test_data)
    return train_predictions, test_predictions
