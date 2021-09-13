def training(model, train_data, train_labels):
    """
    Performs training with the given model

    Contributors:
    - Daniel
    """
    model.fit(train_data, train_labels)
    return model


def predicting(trained_model, data):
    """
    Performs predictions on the test set with the trained model

    Contributors:
    - Daniel
    """
    return trained_model.predict(data)


def show_importances(model, train_data, threshold=0.0):
    """
    Returns the features that have an importance greater than the threshold

    Contributors:
    - Daniel
    """
    importances = [
        (feature, round(importance, 4))
        for feature, importance in zip(train_data.columns, model.feature_importances_)
        if importance > threshold
    ]
    importances.sort(key=lambda x: x[1], reverse=True)
    return [imp[0] for imp in importances]
