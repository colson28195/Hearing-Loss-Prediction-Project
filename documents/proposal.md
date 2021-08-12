## Project Title
The Use of Machine Learning in Wideband Tympanometry Data

## Aim
The aim of the project is to apply machine learning methods in understanding, interpreting and classifying wideband tympanometry data, to automatically diagnose ears as normal or with conductive conditions. In addition, the performance of machine learning tools will be compared to the current method, receiver characteristics analysis, to examine whether machine learning tools are better suited to wideband tympanometry.

## Background
Wideband Tympanometry technology (WBT) is a useful tool used to assess the middle ear function at different frequencies and pressure levels. The middle ear function plays a critical role for hearing, where it transfers the vibration of eardrum into waves in the fluid and membranes of the inner ear. Thus, it is important to accurately detect whether patients’ have conductive conditions, and now WBT is helping to test the middle ear function. WBT provides resonant frequency of the middle ear and can test frequencies from variance frequency range and pressure, which improves from the conventional tympanometry using only low-single frequency. Despite WBT has been available for almost two decades, clinics do not have wide use of it due to the limited interpretation of results. Traditionally, receiver characteristics analysis is used to interpret WBT data and specificity is preferred in analysing the ROC curve. However, the interpretation of ROC curve can be tedious and difficult to interpret, which is why many clinics have not been able to incorporate WBT. Thus, machine learning tools are proposed to analyse WBT with better accuracy and produce results that are easily interpretable.

This proposal addresses the limitation of current method used on WBT and proposes machine learning tools. The WBT data includes tests on frequencies from 250Hz to 8000Hz, which is currently analysed and modelled using ROC curve. The ROC curve is tedious to analyse and involves lots of manual labelling and comparisons, which is not the best tool in a busy clinical environment. Therefore, machine learning classifiers will help to resolve this problem by automatically detecting abnormal ear conditions with improved accuracy. This can help clinics to better incorporate WBT in diagnose patients and reduce time in detecting abnormal conditions. Furthermore, machine learning tools are robust with pre-processing of WBT data and feature extraction, which will be able to take more data in the future and improve clinical efficiency.

## Value proposition
Machine learning is a powerful tool in analysing patterns, learning and making predictions. With a well-trained machine learning algorithm, it can improve the analysis of WBT data and better detect ears with conductive conditions. Here are the key benefits:
1. Improve the accuracy in detecting ears with conductive conditions. The highest accuracy rate from the current method, receiver characteristics analysis, is only 74%. Machine learning classifiers will be trained and compared to find the most accurate one.

2. ncrease reproducibility of the machine learning algorithm and interpretability of the results. Once the machine learning classifier is trained, future data can be fed in easily and results will be produced instantly.  This will save time and human capital in clinics as well as help to transform raw data to meaningful insights about patients’ ear condition.

3. Machine learning tools will be able to help clinic practitioners interpret patients test results quickly and more accurately, which improves efficiency and reduces cost. In addition, machine learning tools can be applied to other data easily, such as WBA data.


## Deliverables
The deliverables for the project include:
1. A machine learning tool that can use WBT data to diagnose ears as normal or with conductive conditions. The key features of the tools will be
- Highly interactive user inferface which allows selection of a ML methods from a curated set of algorithms to evaluate the ear data
- Better visualization of the output metrics for improveed interpretability
- Comparing the result of differet ML methods and choosing the ideal method to diagnose the ear

![mock_tool_ui](https://raw.githubusercontent.com/danielchegwidden/tytonidae-tympanometry/dev-ar/anitha/images/tyty_web_app_ui.JPG?token=AT422D7YNYJPINUD7YFL2M3BDWW2I)

2. Code repository hosted on Github, which can be utilised for future development purposes

3. Final Report that summaries the data characteristics and methods used in the tool

### Project timeline for the client
- GANTT chart for the client includes progress of the project, client meetings and milestones.
(attached as seperate file - Gantt_Charts.xlsx)

## Methods
There is a number of machine learning algorithms can be used to analyse WBT data and comparisons will be made among these methods and against the ROC:

1. Support Vector Machines - SVM is a supervised learning model which performs classification by creating an augmented line to classify data. The aim is to map training data in such a way that the augmented line separating the different calsses is as wide as possible so that new data can be predicted by detecing which side of the line they would fall in. SVM is robust and has a range of functions to choose from and more often than not provides results with a higher accuracy however this comes at the cost of it being compuationally expensive.

2. Decisison Tree - This is a predictive modelling approach that performs classification that explicitly represents decisions and decision making. These decisions are made on the basis of how well the features or a subset of the given features can help classify data. The importance of each feature used to distinguish data is very clear as the model will be represented in the structure of a tree which starts with the most distinguishing feature which gradually decreases as we ravel through the branches. Due to the nature of figuring out which feature will help make the best decisions, this model has a higher time complexity but at the same time we can expect a good accuracy.

3. Random Forrest - Random Forrest is a model constructed by combining multiple decision trees(hence the word forrest) and eventually averaging the results from said decision trees which gives us a better prediciton. More the number of decision trees, better the accuracy. Even though the prediction is better, due to the fact that decision trees have a higher time complexity, time taken by random forrest model to execute will also worsen.

4. Logistic Regression - Logisitic regression is a statistical model that can compute the odds of a record lying in a class after the model has been trained. If the odds of the record corresponding to a class is high, it will assign it to that particular class and if the odds are low, it shall assign that record to the other class. This model is simple to implement but lack accuracy and hence provides relatively bad predictions.


After all models are trained, comparisons will be made to evaluate each model's performance, using accuracy rate, specificity and sensitivity. The models will also be compared against the ROC curve to provide insights as to which model is the optimum model to diagnose ear condidtions.


## Project Management Timeline
- GANTT chart needed for the team includes progress of the project, team meetings and task and deadlines.
(attached as seperate file - Gantt_Charts.xlsx)



## Appendix
To be completed by individuals.
