# activate venv before running the code
from tom import modelling

trained_model, features = modelling.run_svm_clf()

features_names = features
# svm = svm.SVC(kernel='linear')
# svm.fit(X, Y)
modelling.f_importances(features, trained_model.coef_[0])

#                  Machine Learning Model Summary
# =================================================================
# | Label    Accuracy     Precision    Sensitivity   Specificity  |
# |   0       0.8808        1.0000        0.8736        1.0000    |
# |   1       0.8808        0.3235        1.0000        0.8736    |
# -----------------------------------------------------------------
# =================================================================

# --- TEST ---

#                  Machine Learning Model Summary
# =================================================================
# | Label    Accuracy     Precision    Sensitivity   Specificity  |
# |   0       0.9048        1.0000        0.9048        0.0000    |
# -----------------------------------------------------------------
# |                Total Model Accuracy =  0.9048                 |
# =================================================================
