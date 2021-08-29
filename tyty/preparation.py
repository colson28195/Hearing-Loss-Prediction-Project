import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

np.random.seed(24)


def match_pressures(data):
    """
    Extract the rows where the Pressure matches the AdultAbsorbanceData as closely as possible

    Contributors:
    - Tom
    - Daniel
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


def subject_train_test_split(data, test_percent=20):
    """
    Splits the data so each subject is in either the train or the test set but not both

    Contributors:
    - Daniel
    """
    num_subs = np.unique(data["Subject"])
    train_num = len(num_subs) * test_percent // 100
    train_subs = np.random.choice(num_subs, train_num)
    train_data = data[~data["Subject"].isin(train_subs)]
    test_data = data[data["Subject"].isin(train_subs)]
    return train_data, test_data


def split_target(data, feature_columns=[], all_freq=True):
    """
    Splits the target and the relevant features out

    Contributors:
    - Daniel
    """
    target = data["OverallPoF"]
    if all_freq:
        features = data[[col for col in data.columns if "f(" in col] + feature_columns]
    else:
        features = data[feature_columns]
    return features, target


def standard_scaling(train_data, test_data):
    """
    Scale train and test data using standard scaling

    Contributors:
    - Karan
    - Anitha
    - Di
    """
    std_scaler = StandardScaler()
    train_transformed = std_scaler.fit_transform(train_data)
    test_transformed = std_scaler.transform(test_data)
    return train_transformed, test_transformed


def min_max_scaling(train_data, test_data):
    """
    Scale train and test data using MinMax scaling

    Contributors:
    - Karan
    - Anitha
    - Di
    """
    min_max_scaler = MinMaxScaler()
    train_transformed = min_max_scaler.fit_transform(train_data)
    test_transformed = min_max_scaler.transform(test_data)
    return train_transformed, test_transformed
