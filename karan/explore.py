import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from tyty import processing, preparation, modelling, pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

train_data, test_data, train_labels, test_labels = pipeline.full_pipeline(
    feature_columns=["Gender", "EarSide", "Age"]
)

models = {
    "KNeighborsClassifier": KNeighborsClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
}

for name, model in models.items():
    model.fit(train_data, train_labels)
    print(name + "Model Trained.")

print()

for name, model in models.items():
    print(
        name + "Training set"
        ": {:.2f}%".format(model.score(train_data, train_labels) * 100)
    )
    print(
        name + "Testing set"
        ": {:.2f}%".format(model.score(test_data, test_labels) * 100)
    )
    print()
