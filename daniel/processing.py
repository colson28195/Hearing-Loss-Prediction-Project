import pandas as pd
from sklearn.model_selection import train_test_split
from tyty import processing, preparation


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
    removals = [col for col in dataset.columns if "Unnamed" in col] + columns
    dataset.drop(columns=removals, inplace=True)
    return True


def combine_data(data, demo):
    """
    Fixes columns and then combines them
    """
    demo["EarSide"] = ([0] * 120) + ([1] * 119)
    data["EarSide"] = data["EarSide"].map({"Left": 0, "Right": 1})
    data.rename(columns={"Participant ID": "Subject"}, inplace=True)
    return demo.merge(data, on=["Subject", "EarSide"], how="inner").reset_index(
        drop=True
    )


def processing_pipeline():
    """
    Runs the processing pipeline
    """
    data = processing.import_data("data")
    demo = processing.import_data("demographics")

    processing.clean_empty(data)
    processing.clean_empty(demo)

    processing.remove_columns(demo)
    processing.remove_columns(data, ["Gender"])

    combined = processing.combine_data(data, demo)

    return combined


def split_target(data, feature_columns=[]):
    """
    Splits the target and the relevant features out
    """
    target = data["OverallPoF"]
    features = data[[col for col in data.columns if "f(" in col] + feature_columns]
    return features, target


def match_pressures(data):
    """
    Extract the rows where the Pressure matches the AdultAbsorbanceData as closely as possible
    """
    cols = [
        col
        for col in data.columns
        if col not in ["Subject", "AdultAbsorbanceData", "Pressure", "EarSide"]
    ]
    smaller = data.drop(columns=cols)
    smaller["abs"] = abs(smaller["AdultAbsorbanceData"] - smaller["Pressure"])
    grouped = (
        smaller.loc[smaller.groupby(["Subject", "EarSide"])["abs"].idxmin()]
        .reset_index(drop=True)
        .drop(columns="abs")
    )
    return grouped.merge(
        data, on=["Subject", "EarSide", "AdultAbsorbanceData", "Pressure"], how="inner"
    ).reset_index(drop=True)


def prep_pipeline(data, feature_columns=[], pressure_match=False, test_size=0.2):
    """
    Runs the preparation pipeline
    """
    if pressure_match:
        data = preparation.match_pressures(data)

    features, target = preparation.split_target(data, feature_columns)

    train_data, test_data, train_labels, test_labels = train_test_split(
        features, target, test_size=test_size, random_state=24
    )

    return train_data, test_data, train_labels, test_labels


def full_pipeline(feature_columns=[], pressure_match=False, test_size=0.2):
    """
    Runs the processing and preparation pipelines to return data ready for modelling
    """
    result = processing_pipeline()
    train_data, test_data, train_labels, test_labels = prep_pipeline(
        result,
        feature_columns=feature_columns,
        pressure_match=pressure_match,
        test_size=test_size,
    )
    return train_data, test_data, train_labels, test_labels
