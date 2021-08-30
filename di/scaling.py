import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def standard_scaling(x_train, x_test):
    std_scaler = StandardScaler()
    x_train_transformed = std_scaler.fit_transform(x_train)
    x_test_transformed = std_scaler.transform(x_test)
    return x_train_transformed, x_test_transformed


def min_max_scaling(x_train, x_test):
    min_max_scaler = MinMaxScaler()
    x_train_transformed = min_max_scaler.fit_transform(x_train)
    x_test_transformed = min_max_scaler.transform(x_test)
    return x_train_transformed, x_test_transformed
