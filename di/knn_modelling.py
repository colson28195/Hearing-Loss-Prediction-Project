import pandas as pd
import numpy as np

from tyty import pipeline
from sklearn.neighbors import KNeighborsClassifier
import beluga

import matplotlib.pyplot as plt

np.random.seed(24)

train_data, test_data, train_labels, test_labels = pipeline.full_pipeline(
    feature_columns=[], scaling="std_scale"
)

knn_model = KNeighborsClassifier(n_neighbors=116, weights="uniform", metric="minkowski")
train_pred, test_pred, model = pipeline.modelling_pipeline(
    knn_model, train_data, train_labels, test_data
)

print("--- TRAIN ---")
beluga.metrics.summary(train_labels, train_pred, conditions=True)
print("--- TEST ---")
beluga.metrics.summary(test_labels, test_pred, conditions=True)


# Plotting error curve to finding optimal value for k
error1 = []
error2 = []
for k in range(80, 120):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    train_pred, test_pred, model = pipeline.modelling_pipeline(
        knn_model, train_data, train_labels, test_data
    )
    error1.append(np.mean(train_labels != train_pred))
    error2.append(np.mean(test_labels != test_pred))

plt.plot(range(80, 120), error1, label="train")
plt.plot(range(80, 120), error2, label="test")
plt.xlabel("k Value")
plt.ylabel("Error")
plt.legend()
# plt.show()
