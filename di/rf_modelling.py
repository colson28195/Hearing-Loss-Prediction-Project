## rf modelling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tyty import processing, preparation, modelling, pipeline
from di import rf_function
import beluga

np.random.seed(24)

trained_model, new_features = rf_function.run_rf(
    ["Gender", "Age", "EarSide", "Pressure"],
    all_freq=True,
    random_state=47,
    n_estimators=350,
    n_jobs=-1,
    max_leaf_nodes=25,
    bootstrap=True,
    min_samples_split=100,
    criterion="gini",
    max_features=1,
    max_depth=15,
)

# for i in range(2):
#    trained_model, new_features = rf_function.run_rf(
#        new_features,
#       all_freq=False,
#        random_state=47,
#        n_estimators=350,
#        n_jobs=-1,
#        max_leaf_nodes=25,
#        bootstrap=True,
#        min_samples_split=100,
#        criterion="gini",
#        max_features=1,
#        max_depth=15,
#    )
