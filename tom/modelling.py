import pandas as pd
import numpy as np
import sklearn
from sklearn.svm import SVC
import beluga
from sklearn.model_selection import GridSearchCV

from tyty import pipeline


def run_svm_clf():

    # pd.set_option("display.max_rows", None)
    # np.random.seed(24)

    train_data, test_data, train_labels, test_labels = pipeline.full_pipeline(
        feature_columns=["Gender", "EarSide", "Age"],
        pressure_match=True,
        scaling="std_scale",
    )
    print(type(train_data))
    model = SVC(C=5, gamma=0.005)

    train_pred, test_pred, trained_model = pipeline.modelling_pipeline(
        model, train_data, train_labels, test_data
    )
    print("--- TRAIN ---")
    beluga.metrics.summary(train_labels, train_pred, conditions=True)
    print("--- TEST ---")
    beluga.metrics.summary(test_labels, test_pred, conditions=True)

    # return trained_model
    # param_grid = {'C': [0.1,0.6, 1, 5, 8, 10], 'gamma': [0.1,0.05,0.01,0.005,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    # grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
    # grid.fit(train_data,train_labels)
    # #SVC(C=5, gamma=0.005)
    # print(grid.best_estimator_)
