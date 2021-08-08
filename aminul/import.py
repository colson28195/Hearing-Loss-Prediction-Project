import pandas as pd

from tyty import processing

x = processing.import_data("data")
# participant id can be deleted, although it required to select row for correct pressure.
x.drop("Participant ID", inplace=True, axis=1)
# protocol colunm has no vlaue. so it can be deleted
x.drop("Protocol", inplace=True, axis=1)
# Created column is insignificant
x.drop("Created", inplace=True, axis=1)
# data column have similar value in each raw. so it is not significant.
x.drop("Data", inplace=True, axis=1)
# Pressure column also have similar value.
x.drop("Pressure", inplace=True, axis=1)
# Adult absorbance and pressure is required for appropriate raw selection . but as a column , they are not significant.
x.drop("AdultAbsorbanceData", inplace=True, axis=1)
# drop pressure
# x.drop('pressure', inplace=True, axis=1)
print(x.head())
