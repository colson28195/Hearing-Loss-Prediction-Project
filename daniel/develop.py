import pandas as pd

from tyty import pipeline

# import daniel.processing as ps

pd.set_option("display.max_rows", None)

result = pipeline.processing_pipeline()

features, target = pipeline.prep_pipeline(result, ["Gender", "EarSide"])

print(features.head())
