## Project Title
The Use of Machine Learning in Wideband Tympanometry Data

## Aim
The aim of the project is to apply machine learning methods in understanding, interpreting and classifying wideband tympanometry data, to automatically diagnose ears as normal or with conductive conditions. At the same time, a comparison will be drawn between the performance of machine learning tools and the current method, receiver characteristics analysis, to examine whether machine learning tools are better suited to wideband tympanometry.

## Background
The middle ear function plays a critical role for hearing, where it transfers the vibration of eardrum into waves in the fluid and membranes of the inner ear. Thus, it is important to accurately detect whether patients have conductive conditions. Wideband Tympanometry technology (WBT) is a useful tool used to assess the middle ear function at different frequencies and pressure levels. WBT provides resonant frequency of the middle ear and can test frequencies from variance frequency ranges and pressure, which is an imporvement from the conventional tympanometry using only low-single frequency. Despite WBT being available for almost two decades, clinics do not have wide use of it due to the limited interpretation of results. Traditionally, receiver characteristics analysis is used to interpret WBT data and specificity is preferred in analysing the ROC curve. However, the interpretation of ROC curve can be tedious and difficult, which is why many clinics have not been able to incorporate WBT. Thus, machine learning tools are being proposed to analyse WBT to produce results that are more accurate and easily interpretable.

This proposal addresses the limitation of current method used on WBT and proposes machine learning tools. The WBT data includes tests on frequencies from 250Hz to 8000Hz, which is currently analysed and modelled using ROC curve. The method is tedious and involves plenty of manual labelling and comparisons, which is not ideal in a busy clinical environment. Therefore, machine learning classifiers will help to resolve this problem by automatically detecting abnormal ear conditions with improved accuracy. This can help clinics to better incorporate WBT in diagnosing patients and reduce time in detecting abnormal conditions. Furthermore, machine learning tools are robust with pre-processing of WBT data and feature extraction, which will be able to take more data in the future and improve clinical efficiency.

## Value proposition
Machine learning is useful in analysing patterns and making predictions. With a well-trained machine learning algorithm, analysis of WBT data can be improved significantly which in turn helps detect conductive conditions in ears with much more ease.

Here are the key benefits:
1. Improve the accuracy in detecting ears with conductive conditions. The highest accuracy rate from the current method, receiver characteristics analysis, is only 74%. Machine learning classifiers will be trained and compared to find the most accurate one.

2. Increase reproducibility of the machine learning algorithm and interpretability of the results. Once the machine learning classifier is trained, future data can be fed in easily and results will be produced instantly. This will save time and human capital in clinics as well as help transform raw data into meaningful insights about patients’ ear condition.

3. Machine learning tools will be able to help clinic practitioners interpret patients' test results faster and more accurately, which improves efficiency and reduces cost. In addition, machine learning tools can be applied to other data easily, such as WBA data.


## METHODS
Assumptions that we are making.
What sort of preprocessing we are doing and why.

Few of the Machine Learning algorithms that will be used in to analyse and predict outcomes mentioned below -

1. Support Vector Machines - SVM is a supervised learning model which performs classification by creating an augmented line to classify data. The aim is to map training data in such a way that the augmented line separating the different calsses is as wide as possible so that new data can be predicted by detecing which side of the line they would fall in. SVM is robust and has a range of functions to choose from and more often than not provides results with a higher accuracy however this comes at the cost of it being compuationally expensive.

2. Decisison Tree - This is a predictive modelling approach that performs classification that explicitly represents decisions and decision making. These decisions are made on the basis of how well the features or a subset of the given features can help classify data. The importance of each feature used to distinguish data is very clear as the model will be represented in the structure of a tree which starts with the most distinguishing feature which gradually decreases as we ravel through the branches. Due to the nature of figuring out which feature will help make the best decisions, this model has a higher time complexity but at the same time we can expect a good accuracy.

3. Random Forrest - Random Forrest is a model constructed by combining multiple decision trees(hence the word forrest) and eventually averaging the results from said decision trees which gives us a better prediciton. More the number of decision trees, better the accuracy. Even though the prediction is better, due to the fact that decision trees have a higher time complexity, time taken by random forrest model to execute will also worsen.

4. Logistic Regression - Logisitic regression is a statistical model that can compute the odds of a record lying in a class after the model has been trained. If the odds of the record corresponding to a class is high, it will assign it to that particular class and if the odds are low, it shall assign that record to the other class. This model is simple to implement but lack accuracy and hence provides relatively bad predictions.


Once we have run the above models, we shall analyse and compare results of them all to the ROC curve and provide insights as to which model is the optimum model to diagnose ear condidtions.





BELOW CONTINUE DRAFT

UP FOR DEBATE IF WE WANT TO ADD A LINE THAT SUGGESTS IF WE HAVE A LOT OF TIME LEFT, WE CAN USE MORE ALGORITHMS.

DELIVERABLES & TIMELINE
Get more insights by people who attended the meeting and what was agreed upon

-	Discuss conventional tympanometry and WBA to provide more context to the project (research is required)
-	Discuss the current method in classifying WBA data
-	How ML tools are an improvement from ROC
-	Challenges may be faced

An automated machine learning tool that can use WBA data to diagnose ears as normal or with conductive conditions
Features and advantages about the ML tool

PROJECT MANAGEMENT TIMELINE & COSTS
Need to discuss how to articulate

INDIVIDUAL REFLECTION & PLAN
Need to be done by everybody




Run different models and compare results, choose the model with the highest accuracy to diagnose ears.


## Project Management Timeline & Costs
GANTT chart needed for the team – need to decide whether we will use Excel or other automated GANTT chart platform
The GANTT organises the progress of the project, including team meetings, client meetings, tasks and deadlines.
Costs – estimation of manpower and salaries to maintain and update the ML tool.

## Appendix
To be completed by individuals.
