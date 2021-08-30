import pandas as pd
import numpy as np
import beluga

from tyty import pipeline
from daniel import modelling

pd.set_option("display.max_rows", None)
np.random.seed(24)

trained_model, new_features = modelling.run_decision_tree(
    ["Gender"],
    all_freq=True,
    depth=4,
    splits=0.01,
    leaves=0.01,
    pressure=False,
    smote=False,
    path="daniel/decision_treeSMOTE.png",
)
for i in range(3):
    trained_model, new_features = modelling.run_decision_tree(
        new_features,
        all_freq=False,
        depth=4,
        splits=0.01,
        leaves=0.01,
        pressure=False,
        smote=False,
        path="daniel/decision_treeSMOTE" + str(i) + ".png",
    )
