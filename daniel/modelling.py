def training(model, train_data, train_labels):
    """
    Performs training with the given model
    """
    model.fit(train_data, train_labels)
    return model


def predicting(trained_model, test_data):
    """
    Performs predictions on the test set with the trained model
    """
    return trained_model.predict(test_data)
