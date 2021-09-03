import pandas as pd
import numpy as np

from tyty import pipeline
from sklearn.neighbors import KNeighborsClassifier
import beluga

np.random.seed(24)

train_data, test_data, train_labels, test_labels = pipeline.full_pipeline(
    feature_columns=[], scaling="std_scale"
)

# print(train_data, test_data)

knn_model = KNeighborsClassifier(n_neighbors=120)
train_pred, test_pred, model = pipeline.modelling_pipeline(
    knn_model, train_data, train_labels, test_data
)

print("--- TRAIN ---")
beluga.metrics.summary(train_labels, train_pred, conditions=True)
print("--- TEST ---")
beluga.metrics.summary(test_labels, test_pred, conditions=True)


""" graphing
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
# Compute the (weighted) graph of k-neighburs for points in X
# A = model.kneighbors_graph(train_data)
# print(A.toarray())

plot_decision_regions(train_data, train_labels, clf=model, legend=2)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Knn with K='+ 120)
plt.show()
"""
