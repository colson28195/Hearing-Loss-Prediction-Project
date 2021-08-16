import pandas as pd

from tyty import pipeline

result = pipeline.run_pipeline()

features, target = pipeline.split_target(result, ["Gender", "EarSide", "Age"])

# print(features)
# Data typedescription:
data_type = features.describe()
print(data_type)
