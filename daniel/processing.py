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
    return True


def remove_columns(dataset, columns=[]):
    """
    Removes unnecessary columns from the data
    """
    removals = [col for col in dataset.columns if 'Unnamed' in col] + columns
    dataset.drop(columns=removals, inplace=True)
    return True


def combine_data(data, demo):
    """
    Fixes columns and then combines them
    """
    demo['EarSide'] = ([0] * 120) + ([1] * 119)
    data['EarSide'] = data['EarSide'].map({"Left": 0, "Right": 1})
    data.rename(columns={"Participant ID": "Subject"}, inplace=True)
    return demo.merge(data, on=['Subject', 'EarSide'], how='inner')


def run_pipeline():
    """
    Runs the processing pipeline
    """
    data = import_data("data")
    demo = import_data("demographics")

    clean_empty(data)
    clean_empty(demo)

    remove_columns(demo)
    remove_columns(data, ['Gender'])

    combined = combine_data(data, demo)

    return combined


def split_target(data, feature_columns=[]):
    """
    Splits the target and the relevant features out
    """
    target = data['OverallPoF']
    features = data[[col for col in data.columns if 'f(' in col] + feature_columns]
    return features, target
