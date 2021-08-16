
# Explanation for removing /selecting columns:

"Participant id" can be deleted. it does not required to make any prediction.it just idendity of patient."protocol" colunm has no vlaue/ empty column. so it can be deleted."Created" column is insignificant. Because it a time when the measurement is taken."data" column have similar value in each raw. so it is not significant."AdultAbsorbance" and "pressure" is required for appropriate raw selection . but as a column , they are not significant. As we are using "Age" directly as numerical value, so "AgeCategory" just duplicate of that age features."PressureCategory" only contain single char data which is "ambient ". so it is not significant. So we keep all the features related to frequency and add "Gender" and "EarSide" in our features set and the our prediction ground tools in "Target"

```python
from tyty import pipeline

result = pipeline.run_pipeline()

features, target = pipeline.split_target(result, ["Gender", "EarSide","Age"])


```
# Data type Explanation:

For investigating data type of the selected features , we taking help from dictionary_data.txt file and dictionary_demo.txt file. Gender is shown a character data type in data.csv file but int type in demo.csv file. so we have taken the int data type "Gender" feature as our selected feature. After processing and combining the the two data set, the data type of final features file is numerical type. Each column has 16550 rows . so there is no missing value.

```python
data_type= features.describe()
print(data_type)
```
