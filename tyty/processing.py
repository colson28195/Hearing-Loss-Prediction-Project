import pandas as pd


def import_data(dataset):
    """
    Imports the provided data as pandas dataframes from the data folder

    Contributors:
    - Daniel
    """
    return pd.read_csv("data/" + dataset + ".csv")


def clean_empty(dataset):
    """
    Removes the empty columns from the provided data

    Contributors:
    - Daniel
    """
    dataset.dropna(how="all", axis=1, inplace=True)
    return True


def remove_columns(dataset, columns=[]):
    """
    Removes unnecessary columns from the data

    Contributors:
    - Daniel
    """
    removals = [col for col in dataset.columns if 'Unnamed' in col] + columns
    dataset.drop(columns=removals, inplace=True)
    return True


def combine_data(data, demo):
    """
    Fixes columns and then combines them

    Contributors:
    - Daniel
    """
    demo['EarSide'] = ([0] * 120) + ([1] * 119)
    data['EarSide'] = data['EarSide'].map({"Left": 0, "Right": 1})
    data.rename(columns={"Participant ID": "Subject"}, inplace=True)
    return demo.merge(data, on=['Subject', 'EarSide'], how='inner')
