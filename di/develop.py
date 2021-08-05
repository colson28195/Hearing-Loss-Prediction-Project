from data_processing import clean_nan
from data_processing import import_data


data = clean_nan(import_data("data"))
demo = clean_nan(import_data("demographics"))

print(data.head())
print(demo.head())
print(data.columns)
print(demo.columns)
