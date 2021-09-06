import pandas as pd
from tyty import pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import beluga


def run_svm_clf():

    data = pipeline.processing_pipeline()
    train_data, test_data, train_labels, test_labels = pipeline.prep_pipeline(
        data, scaling="std_scale"
    )

    model = SVC()

    train_pred, test_pred, trained_model = pipeline.modelling_pipeline(
        model, train_data, train_labels, test_data
    )

    print("--- TRAIN ---")
    beluga.metrics.summary(train_labels, train_pred, conditions=True)
    print("--- TEST ---")
    beluga.metrics.summary(test_labels, test_pred, conditions=True)

    return trained_model


def run_grid_search_cv():

    data = pipeline.processing_pipeline()
    train_data, test_data, train_labels, test_labels = pipeline.prep_pipeline(
        data, scaling="std_scale"
    )

    parameters = {
        "kernel": ("linear", "rbf"),
        "C": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "gamma": [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.2, 0.3, 0.4, 0.5],
    }

    classifier = SVC()

    grid = GridSearchCV(
        estimator=classifier,
        param_grid=parameters,
        scoring="accuracy",
        cv=10,
        n_jobs=-1,
    )

    grid_search = grid.fit(train_data, train_labels)

    print("---ACCURACY---")
    print(grid_search.best_score_)
    print("---BEST PARAMS---")
    grid_search.best_params_

    model = SVC(C=10, kernel="rbf", gamma=0.3)

    train_pred, test_pred, trained_model = pipeline.modelling_pipeline(
        model, train_data, train_labels, test_data
    )

    print("--- TRAIN ---")
    beluga.metrics.summary(train_labels, train_pred, conditions=True)
    print("--- TEST ---")
    beluga.metrics.summary(test_labels, test_pred, conditions=True)

    return trained_model
