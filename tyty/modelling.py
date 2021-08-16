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
