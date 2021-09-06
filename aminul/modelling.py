import pandas as pd

from sklearn.model_selection import train_test_split

from tyty import pipeline

result = pipeline.processing_pipeline()

features, target = pipeline.split_target(result, ["Gender", "EarSide", "Age"])

# print(features)
# Data typedescription:

print(features)
