import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
import beluga
from imblearn.over_sampling import SMOTE
from tyty import pipeline


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


def show_importances(model, features, threshold=0.0):
    """ Returns the features that have an importance greater than the threshold """
    importances = [
        (feature, round(importance, 4))
        for feature, importance in zip(features, model.feature_importances_)
        if importance > threshold
    ]
    importances.sort(key=lambda x: x[1], reverse=True)
    return [imp for imp in importances]


def save_tree(trained_model, features, path):
    """ Saves the Decision Tree image """
    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(
        trained_model, feature_names=features, class_names=["0", "1"], filled=True
    )
    fig.savefig(path)


def run_decision_tree(
    features, all_freq, depth, splits, leaves, pressure, smote, ada, path
):
    """ Runs the Decision Tree and returns the model"""
    train_data, test_data, train_labels, test_labels, features = pipeline.full_pipeline(
        feature_columns=features, all_freq=all_freq, pressure_match=pressure
    )

    if smote:
        oversample = SMOTE()
        train_data, train_labels = oversample.fit_resample(train_data, train_labels)

    model = DecisionTreeClassifier(
        random_state=24,
        max_depth=depth,
        min_samples_split=splits,
        min_samples_leaf=leaves,
    )
    if ada:
        model = AdaBoostClassifier(model)

    scores = cross_validate(
        model, train_data, train_labels, cv=10, return_train_score=True
    )
    print("Cross-Validated Train: ", np.mean(scores["train_score"]))
    print("Cross-Validated Test: ", np.mean(scores["test_score"]))

    train_pred, test_pred, trained_model = pipeline.modelling_pipeline(
        model, train_data, train_labels, test_data
    )
    print()
    print("--- TRAIN ---")
    beluga.metrics.summary(train_labels, train_pred, conditions=True)
    print("--- TEST ---")
    beluga.metrics.summary(test_labels, test_pred, conditions=True)

    if not ada:
        save_tree(trained_model, train_data.columns, path=path)

    new_features = show_importances(trained_model, features)

    print("--- IMPORTANCES ---")
    print(new_features)

    return trained_model, [n[0] for n in new_features]
