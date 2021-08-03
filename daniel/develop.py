import pandas as pd

from tyty import processing


def import_data(dataset):
    """
    Imports the provided data as pandas dataframes from the data folder
    """
    return pd.read_csv("data/" + dataset + ".csv")


def clean_nan(dataset):
    """
    Removes the empty columns from the provided data
    """
    dataset.dropna(how="all", axis=1, inplace=True)
    return dataset


data = processing.clean_nan(processing.import_data("data"))
demo = processing.clean_nan(processing.import_data("demographics"))

print(data.head())
print(demo.head())
