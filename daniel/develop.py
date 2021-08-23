import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import beluga

from tyty import pipeline
import daniel

pd.set_option("display.max_rows", None)
np.random.seed(24)

train_data, test_data, train_labels, test_labels = pipeline.full_pipeline(
    feature_columns=["Gender", "EarSide", "Age"]
)

model = DecisionTreeClassifier(random_state=24)

train_pred, test_pred = pipeline.modelling_pipeline(
    model, train_data, train_labels, test_data
)

beluga.metrics.summary(train_labels, train_pred, conditions=True)
beluga.metrics.summary(test_labels, test_pred, conditions=True)
