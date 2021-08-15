## Project Title
Tytonidae Tympanometry - Applying Machine Learning to predict hearing loss using Wideband Absorbance data.

## Aim
This project aims to apply machine learning techniques to wideband absorbance (WBA) data in order to classify ears as either normal, or with conductive conditions. These techniques seek to replace the receiver operating characteristic (ROC) analysis that is currently performed, by both increasing the performance and improving the interpretability of the results.

## Background
Our ears are important, we need them working in order to hear. The middle ear functions support our hearing by acting as a matching device between the different pressures exerted by the air and our cochlear fluids. The conventional method for testing the effectiveness of the middle ear is through using a single tone at 226Hz to measure the acoustic changes in the ear. More recently, wideband tympanometry has been used to generate data from 250Hz to 8000Hz to give a more accurate picture of what is present in the inner ear, namely looking for conductive conditions.

The wideband data that has been collected across 16 different frequencies in this range measures the absorbance of acoustic energy by the ear at tympanometric peak pressure (WBT). Coupled with being measured at more than 60 different pressures, there is a large amount of data to analyse to understand what is happening in the middle ear. This methodology has been available for almost two decades, but has seen low uptake due to the limited ability of clinicians to interpret the result. Traditionally, the receiver operating characteristic (ROC) was used to interpret WBT results, with the measure of success being the specificity metric.

This proposal seeks to not only apply machine learning to improve the effectiveness of the analysis by increasing specificity, but also to target the interpretability of the results. Because if clinicians can neither trust nor understand the results, then their effectiveness is reduced to zero. By taking in the raw WBT data, a tool can automatically process the data and use the top performing model to predict the occurrence of conductive conditions or not. Coupled with simple visualisations and an easy-to-understand output, this would allow clinicians to focus more on the patient, and delivering the care that they require.

A simple to use and easy to understand tool, that accurately predicts the occurrence of conductive conditions in the middle ear is the goal.

## Value Proposition
The use of machine learning in analysing the WBA data has the following key benefits:

1. Improving on the performance of existing methods. The current best area under the ROC results are 77\%, which is the target to achieve for the machine learning models. We will also use accuracy as a good classification measure, as well as sensitivity and specificity to fine-tune the models' performance.

2. Increasing the interpretability of the results. The current ROC analysis is often difficult for clinicians to interpret and apply its outputs to their patients. The machine learning models seek to output a clear and simple result to empower their application for the benefit of patients. There is a large amount of data collected, and using feature selection to identify valuable and unnecessary data points would give clinicians greater understanding of what is important or not.

3. Enable fast and reproduceable analysis. By creating a trained and deployed model, machine learning can quickly analyse WBA data from new patients and provide clinicians with an instantaneous classification of their patients' ears for conductive conditions. Coupled with an automated processing pipeline, raw data collected from patients can be directly input into a tool that can generate the interpretable results that clinicians are after.

## Deliverables
The deliverables for Tytonidae Tympanometry are as follows:

1. A machine learning tool that takes in the WBA data, and any relevant demographic data, to classify a patient's ears as either normal or with conductive conditions. The output of this tool will not only be the classification, but also a visual analysis of how the prediction was made.

2. A code repository on GitHub which explores different machine learning methods to identify the best performing one, as well as processing steps, that can be used for further or improved analysis. This code repository and any outputs will use Python to generate the results.

3. A report that summarises the data, the processing steps that occurred, the models that were explored, and the results that were achieved by each.

## Methods
For this project, Python has been selected as the programming language of choice, with Pandas and Scikit-Learn being the key Python libraries for data manipulation and machine learning. These were chosen for their versatility and simplicity, to support the ability to easily interpret the results.

Looking at previous work done with wideband absorbance data and machine learning by \citet{grais2021analysing} and \citet{sundgaard2021deep}, there appears to be a preference for more complicated black-box models such as deep learning and convolutional neural networks (CNN). These may provide better results, but they conflict with the requirement to be explainable to clinicians. With this in mind (and still aiming for the bets performance), the following machine learning techniques have been selected for the initial analysis:

1. Logistic Regression - LR is a statistical technique that looks at the probability of a record (a patient in this situation) belonging to either the positive class (with a conductive condition) or the negative class (without a conductive condition - a normal middle ear). As logistic regression is a simple model compared to the others, it provides a baseline result upon which to compare the other results with data that has gone through the same processing steps.

2. Support Vector Machine - SVM is a supervised learning technique that performs classification by creating a street through the data, in n-dimensional space where n is the number of features (where n is 2, this is a line, where n is 3, this becomes a plane, and for higher values of n, this cannot be visualised). This street seeks to be as wide as possible whilst separating the positive and negative classes as best as possible. SVM is effective in high dimensional (n) space, which is useful for WBT data as it is collected across 16 frequencies, plus the additional demographic features that are added in.

3. Decision Tree - DT is one of the most explainable machine learning techniques, where the model is made up of branches and leaf nodes. Each branch represents a test, and each leaf node is the resulting class that is being predicted. This allows a logic flow to be followed down the tree, seeing which features are contributing to the prediction of the class. The features that have a greater effect of the prediction, the higher they are up in the tree, which will give insights to clinicians as to why the model is outputting a certain result.

4. Random Forest - RF is a ensemble model that incorporates many decision trees and averages out the results of each individual tree. As this incorporates multiple trees, it inherently takes longer than a single tree, but a wisdom-of-the-crowd approach results in a decrease in variance of the results. Even though each tree is easily interpretable, the overall forest loses this due to the nature of averaging across a large number of trees.

All of these models will be compared across a range of metrics, and with the existing ROC results,s to see which is the best performing model for incorporation into a tool. The data that is being passed into these models will incorporate different levels of feature selection to identify those features which add predictive value to the model, as well as one of the following record selection methods:

1. All data - this will incorporate all of the WBT data across all pressures to see if there is any predictive power in any of the data that was collected.

2. Matching Pressures - only the records where the pressure recorded matches the tympanometric peak pressure which is recorded as the Adult Absorbance feature in the data.

Additional models such as K-Means Clustering or Neural Networks may be explored given the time constraints and the performance of the above four models in order to deliver the best results.

## Management & Timeline
See Appendix B for Gantt Chart.

## Appendix

### Appendix A
Individual Reflections

For this project I have taken on the role of team leader which works well with my software engineering skills, such as Git. As the leader, I set up the GitHub repository so that everyone could easily start working straight away, as well as writing guides for the team to follow for those that are less familiar with these technologies. I have walked the team in our developer meetings and individually through these so that the technology is not a barrier to anyone contributing.

I am fairly organised and so I create the agendas for each of our developer and client meetings, as well as take minutes so those that could not attend are able to keep up with what was discussed. I also liaise directly with the client to ensure that we understand the requirements and meet regularly to discuss progress. This involves some coordination with availabilities and balancing in-person and virtual team members.

I raise Issues on GitHub for tasks that need to be completed to allow the other members to easily identify what they are required to do. I am familiar with Python and so I review most work that is completed to ensure that it is compatible with the current working code, and then merge it into the main code that will form the basis of our tool. My Python skills also extend to machine learning and I have developed models using all of our proposed techniques. This allowed me to contribute to proposing these models, as well as being able to develop and review them, to ensure that they are working as intended.

My plan for this team is to develop their software engineering skills through the use of Git, and their Python development skills through building the tool. I am also a believer in collaborative development, so I split up the team in two to work on Phase 1 which is this proposal document and the data processing step. For Phase 2 and 3, these teams will switch so people get the chance to work with different people.

I am interested in deployment, and I am guiding the team towards writing high-quality, reproduceable code, to be translated into a working machine learning tool that the client can utilise as a clinician.

### Appendix B
Gantt Chart
