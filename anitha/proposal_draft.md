## Background 
Understanding the data 
- where was data collected (university audiology laboratory, Brisbane) 
- how many ears studied (with & without OME)
- tests performed : PTA (226Hz), WBT (226Hz to 8000Hz) / Pressure
- feature columns including gender (male & female), age range, ethinicity
- target column class to predict middle ear health

Current data interpretation practice
- ROC curve is used to analyze the diagnostic value of WBT
- Metrics used Sensitivity & Specificity (preferred)
- ROC graphs
- Sensitivity & Specificity table

## Value Proposition
Why use data science
- limited understanding and interpretation of WBT data
- studying ROC curve tedious process for multiple frequencies

Understanding bottleneck/challenges in client’s data interpretation
- statistical tests performance and understanding?
- expert knowledge

Have a clear value proposition (efficiency, consistency/reproduciblity, cost-saving etc)
- ML classification tool to support clinical diagnosis
- current best score
- can this tool be repurposed for other data like WBA?
- risk assessment (low risk and high reward)

## Deliverables 
1. A machine learning tool that can use WBA data to diagnose ears as normal or with conductive conditions. The key features of the tools will be
- Highly interactive user inferface which allows selection of a ML methods from a curated set of algorithms to evaluate the ear data 
- Better visualization of the output metrics for improveed interpretability 
- Comparing the result of differet ML methods and choosing the ideal method to diagnose the ear

![mock_tool_ui](https://raw.githubusercontent.com/danielchegwidden/tytonidae-tympanometry/dev-ar/anitha/images/tyty_web_app_ui.JPG?token=AT422D7YNYJPINUD7YFL2M3BDWW2I)

2. Code repository hosted on Github, which can be utilised for future development purposes
3. Final Report that summaries the data characteristics and methods used in the tool



## Methods
Preliminary interrogation of the data

Choice of different data exploration and analytical techniques
- logistic regression
- decision tree
- random forest
- svm

How to make the process understandable and the output usable (e.g. parameter control, incorporating domain knowledge, communicating output uncertainty)
- feature selection
- metrics

Visualisation to communicate the processes/results

## Timeline

## Appendix

### Di Yao

### Karan Rebello

### Anitha Raghupathy

### Cheng Nian

### Aminul Islam

### Daniel Chegwidden
