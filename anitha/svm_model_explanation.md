The main motive of SVM is to create best decision boundry that seperates the two classes. A line or hyperplane is fitted to WBT data, to classify people with conductive hearing disorders from normal patients.

The tolerance of algorithm is controlled using Soft Margins and Kernal Tricks, to handle misclassifications.GridSearch CV method is employed to find best parameters for the SVM algorithm, due to limited knowledge of the dataset. RBF kernal is chosen as optimal kernal, with degree of tolerance (C) set to 10 and a very low gamma value indicating that every data point has influence on the decision boundry.The test model accuracy is 0.896 and train model accuracy is 0.893

![SVM results](images/svm_results.jpg)
