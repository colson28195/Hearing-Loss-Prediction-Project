import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import beluga
from tyty import pipeline
import tyty.modelling

np.random.seed(24)


def run_rf(
    features,
    all_freq,
    random_state,
    n_estimators,
    n_jobs,
    max_leaf_nodes,
    bootstrap,
    min_samples_split,
    criterion,
    max_features,
    max_depth,
):
    train_data, test_data, train_labels, test_labels, features = pipeline.full_pipeline(
        feature_columns=features, all_freq=all_freq
    )
    model = RandomForestClassifier(
        random_state=random_state,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        max_leaf_nodes=max_leaf_nodes,
        bootstrap=bootstrap,
        min_samples_split=min_samples_split,
        criterion=criterion,
        max_features=max_features,
        max_depth=max_depth,
    )
    # added in cross validation
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

    new_features = tyty.modelling.show_importances(
        trained_model, features, threshold=0.09
    )

    print("--- IMPORTANCES ---")
    print(new_features)

    return trained_model, new_features
