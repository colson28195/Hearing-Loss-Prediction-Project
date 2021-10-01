import pandas as pd
import numpy as np

from daniel import modelling

pd.set_option("display.max_rows", None)
np.random.seed(10)

trained_model, new_features = modelling.run_decision_tree(
    ["Pressure"],
    all_freq=True,
    depth=5,
    splits=0.05,
    leaves=0.05,
    pressure=True,
    smote=True,
    ada=False,
    path="daniel/images/decision_tree_X.png",
)
