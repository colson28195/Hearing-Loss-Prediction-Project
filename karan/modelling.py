import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import beluga
from tyty import pipeline


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
    train_data, test_data, train_labels, test_labels = pipeline.full_pipeline(
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
    train_pred, test_pred, trained_model = pipeline.modelling_pipeline(
        model, train_data, train_labels, test_data
    )
    print()
    print("--- TRAIN ---")
    beluga.metrics.summary(train_labels, train_pred, conditions=True)
    print("--- TEST ---")
    beluga.metrics.summary(test_labels, test_pred, conditions=True)

    importances = [
        (feature, round(importance, 4))
        for feature, importance in zip(
            train_data.columns, trained_model.feature_importances_
        )
        if importance > 0.09
    ]
    print("--- IMPORTANCES ---")
    print(sorted(importances, key=lambda x: x[1], reverse=True))
    new_features = [imp[0] for imp in importances]

    return trained_model, new_features
