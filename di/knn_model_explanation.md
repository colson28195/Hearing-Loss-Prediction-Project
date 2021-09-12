K-Nearest Neighbors (KNN) algorithm assumes that similar data points are close to each other, this means that data points representing patients with conductive hearing conditions are closer to each other, while data points representing normal patients are closer to each other. The algorithm is based on calculating the distance between data points with pre-defined number of neighbors (k).

Choosing the optimal number for k is a tricky task in KNN algorithm. When test error stabilizes and is low, this is the optimal value for k.  To find the optimal value of k, an error curve was plotted, and it has demonstrated that the optimal value of k is 116 for the WBT dataset, which results in a test model accuracy of 0.8949 and train model accuracy of 0.872.

![error_curve](di\error_curve_with_different_k.png)

There is no _feature_importance attribution for KNN algorithm, the importance of features could be computed by running regression model and rank the coefficients for the features.  However, the ranking was not very clear.

Disadvantage of KNN:
-	Choosing the optimal number for k can be tricky, a low value of k could lead to overfitting problems while a high value of k could lead to underfitting.
-	The computing process will become much slower when more data is fed in. This might be an issue in the future if more patient’s data is collected and modelled.

Despite that KNN algorithm is a simple classification model and it has a good accuracy score, it has its disadvantages. Comparing to other models, it does not have the highest accuracy. Hence, I don’t think it is the best model for the WBT dataset.
