import pandas as pd

from tyty import processing

x = processing.import_data("data")
# print(x.head())

newdf = x.loc[(x.AdultAbsorbanceData == x.Pressure)]
list = sorted(x)
print(list)
print(newdf)
