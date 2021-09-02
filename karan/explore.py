import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tyty import processing, preparation, modelling, pipeline
from karan import modelling
import beluga

train_data, test_data, train_labels, test_labels = pipeline.full_pipeline(
    feature_columns=["Gender", "EarSide", "Age"]
)

# models = {
#     "RandomForestClassifier": RandomForestClassifier(
# random_state=47, n_estimators=350, n_jobs=-1, max_leaf_nodes=20,bootstrap=True,min_samples_split=100,criterion="gini"
# ,max_features=6,max_depth=15
#        random_state=47,
#        n_estimators=350,
#        n_jobs=-1,
#        max_leaf_nodes=25,
#        bootstrap=True,
#        min_samples_split=100,
#        criterion="gini",
#        max_features=6,
#        max_depth=15,
#    ),
# }
# for name, model in models.items():
#    train_pred, test_pred, trained_model = pipeline.modelling_pipeline(
#        model, train_data, train_labels, test_data
#    )
#    print(name + "Model Trained.")
#    beluga.metrics.summary(train_labels, train_pred, conditions=True)
#    beluga.metrics.summary(test_labels, test_pred, conditions=True)
#    print()

trained_model, new_features = modelling.run_rf(
    ["Gender"],
    all_freq=True,
    random_state=47,
    n_estimators=350,
    n_jobs=-1,
    max_leaf_nodes=25,
    bootstrap=True,
    min_samples_split=100,
    criterion="gini",
    max_features=1,
    max_depth=15,
)
for i in range(2):
    trained_model, new_features = modelling.run_rf(
        new_features,
        all_freq=False,
        random_state=47,
        n_estimators=350,
        n_jobs=-1,
        max_leaf_nodes=25,
        bootstrap=True,
        min_samples_split=100,
        criterion="gini",
        max_features=1,
        max_depth=15,
    )
