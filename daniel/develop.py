import pandas as pd

from tyty import processing

pd.set_option("display.max_rows", None)

data = processing.clean_empty(processing.import_data("data"))
demo = processing.clean_empty(processing.import_data("demographics"))

# print(data.head())
# print(demo.head())
# print(data.dtypes)
# print(demo.dtypes)
# print(demo['PQMedDig'])
# print(data.isna().any())
