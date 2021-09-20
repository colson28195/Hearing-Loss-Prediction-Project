import pandas as pd
import numpy as np

from tyty import modelling

pd.set_option("display.max_rows", None)
np.random.seed(10)

trained_model, new_features = modelling.run_decision_tree(
    ["Pressure"],
    all_freq=True,
    depth=5,
    splits=0.05,
    leaves=0.05,
    pressure=True,
    path="tyty/images/decision_tree.png",
)
