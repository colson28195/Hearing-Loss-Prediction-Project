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


def prep_pipeline(
    data,
    feature_columns=[],
    all_freq=True,
    pressure_match=False,
    test_percent=20,
    scaling=None,
    pca=False,
):
    """
    Runs the preparation pipeline

    Contributors:
    - Daniel
    - Di
    """
    if pressure_match:
        data = preparation.match_pressures(data)

    train, test = preparation.subject_train_test_split(data, test_percent=test_percent)
    train_data, train_labels = preparation.split_target(
        train, feature_columns=feature_columns, all_freq=all_freq
    )
    test_data, test_labels = preparation.split_target(
        test, feature_columns=feature_columns, all_freq=all_freq
    )

    if scaling == "min_max":
        train_data, test_data = preparation.min_max_scaling(train_data, test_data)
    elif scaling == "std_scale":
        train_data, test_data = preparation.standard_scaling(train_data, test_data)

    if pca:
        train_data, test_data = preparation.pca(train_data, test_data)

    return (train_data, test_data, train_labels, test_labels)


def full_pipeline(
    feature_columns=[],
    all_freq=True,
    pressure_match=False,
    test_percent=20,
    scaling=None,
):
    """
    Runs the processing and preparation pipelines to return data ready for modelling

    Contributors:
    - Daniel
    - Di
    """
    result = processing_pipeline()
    (train_data, test_data, train_labels, test_labels) = prep_pipeline(
        result,
        feature_columns=feature_columns,
        all_freq=all_freq,
        pressure_match=pressure_match,
        test_percent=test_percent,
        scaling=scaling,
    )
    return (train_data, test_data, train_labels, test_labels)


def modelling_pipeline(model, train_data, train_labels, test_data):
    """
    Trains the model and performs prediction on the unseen test set

    Contributors:
    - Daniel
    """
    trained_model = modelling.training(model, train_data, train_labels)
    train_predictions = modelling.predicting(trained_model, train_data)
    test_predictions = modelling.predicting(trained_model, test_data)
    return train_predictions, test_predictions, trained_model
