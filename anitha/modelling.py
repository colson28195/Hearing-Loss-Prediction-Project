import pandas as pd
from tyty import pipeline
from sklearn.svm import SVC
import beluga


def run_svm_unscaled():
    data


def run_svm_clf():

    data = pipeline.processing_pipeline()
    train_data, test_data, train_labels, test_labels = pipeline.prep_pipeline(
        data, scaling="std_scale"
    )

    model = SVC(kernel="rbf", gamma=5, C=0.001)

    train_pred, test_pred, trained_model = pipeline.modelling_pipeline(
        model, train_data, train_labels, test_data
    )

    print("--- TRAIN ---")
    beluga.metrics.summary(train_labels, train_pred, conditions=True)
    print("--- TEST ---")
    beluga.metrics.summary(test_labels, test_pred, conditions=True)

    return trained_model
