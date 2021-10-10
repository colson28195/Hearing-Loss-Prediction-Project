Explain the reasons for selecting the approaches/methods:
Support vector machines (SVMs) are supervised machine learning algorithms, which use labelled dataset for classification and regression analysis. SVMs are capable of finding a decision boundary, also called hyperplane, that separates the data points into two classes. Here the support vectors refer to the data points that are closest to the hyperplane, these factors could affect the position of the hyperplane, which will eventually decide the class of each data point. In the context of our project, we need to analyse WBT data, a high-dimensional dataset that includes demographic features and 16 frequencies, to classify a patient’s ears as normal or with conductive conditions. With a set of algorithms called “kernel methods”, SVMs can do pattern analysis by mapping inputs in higher dimensional space, which makes is suitable for our task.


figure 1: Support vectors-the closest data point to the hyperplane

figure 2: margins: distance between the hyperplane and the closest point from both data sets

To sum up, here are the reasons why we find SVMs suitable for this project:
1.	Provide accurate results
2.	Do well for small and clean datasets
3.	Can implement both linearly separable and non-linearly separable data
4.	Effective in high dimensional space
 
For each method, explain the approach (with references) and present experimental results
