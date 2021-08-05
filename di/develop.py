from tyty import processing


data = processing.clean_nan(processing.import_data("data"))
demo = processing.clean_nan(processing.import_data("demographics"))

print(data.head())
print(demo.head())
print(data.columns)
print(demo.columns)
