from tyty import processing


data = processing.clean_empty(processing.import_data("data"))
demo = processing.clean_empty(processing.import_data("demographics"))

print(data.head())
print(demo.head())
print(data.columns)
print(demo.columns)
