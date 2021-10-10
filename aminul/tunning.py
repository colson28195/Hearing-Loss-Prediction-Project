import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from tyty import pipeline


train_data, test_data, train_labels, test_labels, features = pipeline.full_pipeline(
    feature_columns=["Gender", "EarSide", "Age"]
)

model = LogisticRegression()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

space = dict()
space["solver"] = ["newton-cg", "lbfgs", "liblinear"]
space["penalty"] = ["none", "l1", "l2", "elasticnet"]
space["C"] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

# define search
search = GridSearchCV(model, space, scoring="accuracy", n_jobs=-1, cv=cv)
# execute search
result = search.fit(train_data, train_labels)
# summarize result
print("Best Score: %s" % result.best_score_)
print("Best Hyperparameters: %s" % result.best_params_)
