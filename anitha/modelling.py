from beluga.metrics import true_negative
import pandas as pd
from tyty import pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import beluga


def run_svm_clf():

    data = pipeline.processing_pipeline()
    train_data, test_data, train_labels, test_labels = pipeline.prep_pipeline(
        data, scaling="std_scale", pca=True
    )

    model = SVC(C=10, kernel="rbf", gamma=0.01)

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
        data,
        feature_columns=["Gender", "EarSide", "Age"],
        scaling="std_scale",
        pca=False,
    )

    parameters = {
        "C": [0.1, 1, 10, 100, 1000],
        "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
        "kernel": ["linear"],
    }

    grid = GridSearchCV(SVC(), parameters, refit=True, verbose=3)

    train_pred, test_pred, trained_model = pipeline.modelling_pipeline(
        grid, train_data, train_labels, test_data
    )

    print("---BEST SCORE---")
    print(trained_model.best_score_)
    print("---BEST ESTIMATOR---")
    print(grid.best_estimator_)

    print("--- TRAIN ---")
    beluga.metrics.summary(train_labels, train_pred, conditions=True)
    print("--- TEST ---")
    beluga.metrics.summary(test_labels, test_pred, conditions=True)

    return trained_model
