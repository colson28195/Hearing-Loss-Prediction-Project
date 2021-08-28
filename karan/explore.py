import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from tyty import processing, preparation, modelling, pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import beluga

train_data, test_data, train_labels, test_labels = pipeline.full_pipeline(
    feature_columns=["Gender", "EarSide", "Age"]
)

models = {
    "KNeighborsClassifier": KNeighborsClassifier(),
    "RandomForestClassifier": RandomForestClassifier(
        random_state=42, n_estimators=200, n_jobs=-1, max_leaf_nodes=4
    ),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=47, max_leaf_nodes=4),
}

for name, model in models.items():
    train_pred, test_pred = pipeline.modelling_pipeline(
        model, train_data, train_labels, test_data
    )
    print(name + "Model Trained.")
    beluga.metrics.summary(train_labels, train_pred, conditions=True)
    beluga.metrics.summary(test_labels, test_pred, conditions=True)
    print()
