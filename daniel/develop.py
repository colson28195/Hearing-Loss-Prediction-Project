import pandas as pd
import numpy as np
from daniel import modelling

pd.set_option("display.max_rows", None)
np.random.seed(24)

trained_model, new_features = modelling.run_decision_tree(
    ["Gender", "Age", "EarSide", "Pressure"],
    all_freq=True,
    depth=5,
    splits=0.05,
    leaves=0.05,
    pressure=True,
    smote=False,
    ada=False,
    path="daniel/images/decision_tree_X.png",
)
# for i in range(10):
#     trained_model, new_features = modelling.run_decision_tree(
#         new_features,
#         all_freq=False,
#         depth=5,
#         splits=0.01,
#         leaves=0.01,
#         pressure=False,
#         smote=False,
#         ada=False,
#         path="daniel/images/decision_tree_X" + str(i) + ".png",
#     )
