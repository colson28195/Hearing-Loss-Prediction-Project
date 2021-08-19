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


def prep_pipeline(data, feature_columns=[], pressure_match=False, test_percent=20):
    """
    Runs the preparation pipeline

    Contributors:
    - Daniel
    """
    if pressure_match:
        data = preparation.match_pressures(data)

    train, test = preparation.subject_train_test_split(data, test_percent=test_percent)
    train_data, train_labels = preparation.split_target(
        train, feature_columns=feature_columns
    )
    test_data, test_labels = preparation.split_target(
        test, feature_columns=feature_columns
    )

    return train_data, test_data, train_labels, test_labels


def full_pipeline(feature_columns=[], pressure_match=False, test_percent=20):
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
        test_percent=test_percent,
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
