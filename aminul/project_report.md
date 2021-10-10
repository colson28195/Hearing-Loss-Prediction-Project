# My contribution:
At the very beginning of the project, i start working with data exploration as a part of data tranformation. i have scrutinize into different data type , missing data and any duplicate presentation of same feature(import.py). primary feature selection at the inception of the project done by me.
A breif data description has been made in the column_selection.md file.
# working with ML model:
i have chosen logistic regression method for our classifiaction task beacuse it the simplest and easily interpretable ML alghorithm for classification pproblem. Logistic regresion does not feature importance attribute. so that i have adopt differnt appoach for finding feature importance. i have done hyper parameter tunning with GridSearchCV(tunning.py) and found that the 'lblinear' solver with l2 panelty and c=10 provide best score.i have run my model with diffrent hyper-parameter (modelling.py)and finally found it works well with selectde parameters e. so i got 84.20% overall accuracy with training set and almost 85% with test set. we have chosen it as our baseline accuracy and our team have develop some complex model to find higher accuracy and sensitivity.
# Final model:
After discussion , we have chosen decision tree as our final model. i have worked the decision tree model after final selection (dtree.py). i got roughly 87% accuracy for both training and test set.
