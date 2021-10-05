Points for me to consider :

About the model

Advantages

    Better Accuracy


Disadvantages

    Interpretability


Random Forest is an ensemble model that incorporates many decision trees and averages out the results of each individual tree.   As this incorporates multiple trees, it inherently takes longer than a single tree, but a wisdom-of-the-crowd approach results in a decrease in variance of the results.

A baseline RF model provided results that were good fig 1 but there was room for improvement. On performing some hyperparameter tuning, results were found to change for the better with an accuracy of fig 2. On performing feature importance in addition to more intense hyper parameter tuning, output gained was fig 3 with important features being provide them. On using 10 fold cross validation to further check the performance of the model the results were found to be fig 4.

Coming to decide which model to go with, RF was one of the top models to perform but since the overall forest loses interpretability due to the nature of averaging across a large number of trees that would make it extremely difficult for the client and clinicians using the model to understand how the black box works. To maintain our goal of providing a model with the balance of interpretability and good results, we did not further consider the use of the RF model.
