## rf modelling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tyty import processing, preparation, modelling, pipeline
from karan import modelling
import beluga

np.random.seed(24)

trained_model, new_features = modelling.run_rf(
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
