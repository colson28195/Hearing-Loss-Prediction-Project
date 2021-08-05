import pandas as pd


def import_data(dataset):
    """
    Imports the provided data as pandas dataframes from the data folder
    """
    return pd.read_csv("data/" + dataset + ".csv")


def clean_empty(dataset):
    """
    Removes the empty columns from the provided data
    """
    dataset.dropna(how="all", axis=1, inplace=True)
    return dataset
