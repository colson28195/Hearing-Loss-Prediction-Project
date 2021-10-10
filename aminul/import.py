import pandas as pd

from tyty import pipeline


train_data, test_data, train_labels, test_labels, features = pipeline.full_pipeline(
    feature_columns=["Gender", "EarSide", "Age"]
)
print(train_data)

# Data typedescription:
# data_type = features.describe()
# print(data_type)
