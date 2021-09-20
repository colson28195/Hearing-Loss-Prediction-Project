# activate venv before running the code
from tom import modelling

trained_model, features = modelling.run_svm_clf()

features_names = features
# svm = svm.SVC(kernel='linear')
# svm.fit(X, Y)
modelling.f_importances(features, trained_model.coef_[0])
