import pandas as pd


def import_data(dataset):
    """
    Imports the provided data as pandas dataframes from the data folder
    
    Contributors:
    - Daniel
    """
    return pd.read_csv("data/" + dataset + ".csv")


def clean_nan(dataset):
    """
    Removes the empty columns from the provided data
    
    Contributors:
    - Daniel
    """
    dataset.dropna(how="all", axis=1, inplace=True)
    return dataset
