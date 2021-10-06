import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import beluga
from tyty import pipeline


train_data, test_data, train_labels, test_labels, features = pipeline.full_pipeline(
    feature_columns=["Gender", "EarSide", "Age"]
)

# train_data = train_data.drop(["f(250)", "f(400)", "f(2500)"], axis=1)
# test_data = test_data.drop(["f(250)", "f(400)", "f(2500)"], axis=1)


model = LogisticRegression(penalty="l1", solver="liblinear")

train_pred, test_pred, trained_model = pipeline.modelling_pipeline(
    model, train_data, train_labels, test_data
)

beluga.metrics.summary(train_labels, train_pred, conditions=True)
beluga.metrics.summary(test_labels, test_pred, conditions=True)

importance = model.coef_[0]
feature_importance = abs(model.coef_[0])
feature_importance = 100.0 * (feature_importance / feature_importance.max())


print(feature_importance)
