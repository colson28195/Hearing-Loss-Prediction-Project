import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from ipywidgets import interact

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
    test_num = len(num_subs) * test_percent // 100
    test_subs = np.random.choice(num_subs, test_num)
    train_data = data[~data["Subject"].isin(test_subs)]
    test_data = data[data["Subject"].isin(test_subs)]
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


def pca(train_data, train_labels, test_data):
    """
    Find first two principal components using PCA

    Contributors:
    - Anitha
    """
    pca = PCA(n_components=3)
    train_transformed = pca.fit_transform(train_data)
    feat_var = np.var(train_transformed, axis=0)
    feat_var_rat = feat_var / (np.sum(feat_var))
    print("Variance Ratio of 4 PCs: ", feat_var_rat)
    test_transformed = pca.transform(test_data)

    Xax = train_transformed[:, 0]
    Yax = train_transformed[:, 1]
    labels = train_labels
    cdict = {0: "red", 1: "green"}
    labl = {0: "0", 1: "1"}
    marker = {0: "*", 1: "o"}
    alpha = {0: 0.3, 1: 0.5}

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    for l in np.unique(labels):
        ix = np.where(labels == l)
        ax.scatter(
            Xax[ix],
            Yax[ix],
            c=cdict[l],
            s=40,
            label=labl[l],
            marker=marker[l],
            alpha=alpha[l],
        )

    plt.xlabel("First Principal Component", fontsize=14)
    plt.ylabel("Second Principal Component", fontsize=14)
    plt.legend()
    plt.show()

    return train_transformed, test_transformed
