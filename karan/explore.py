import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tyty import processing, preparation, modelling, pipeline
from karan import modelling
import beluga

np.random.seed(23)

trained_model, new_features = modelling.run_rf(
    ["Age", "f(8000)", "f(2500)", "f(5000)"],
    # best
    # ['Age', 'f(8000)', 'f(4000)', 'f(2500)'],
    all_freq=False,
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
