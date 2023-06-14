# plot_kfold_cv

This is a small plotting module that will run k-fold cross-validations, calculate AUC, specificity, and sensitivity, and make AUROC plots.

## Prerequisites

- sklearn
- numpy
- matplotlib

## Getting started

Before using this module you have to download and copy the file *plot_kfold_cv.py* to the same location where is your own python script that will use this module.

## Example usage in your script (assuming you use jupyter notebooks or google collab)

```
#NOTE THIS CODE IS JUST AN EXAMPLE IT WILL NOT RUN ALONE
#YOU HAVE TO LOAD YOUR DATA FIRST AND FOLLOW THE INSTRUCTIONS IN THE COMMENTS BELOW

#load the module
from plot_kfold_cv import run_cv_and_plot_auc

#load required tools for generating your moodel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
#load any other packages you may need like numpy, etc

#create your logistic regression model
logistic_regression_model = LogisticRegression(max_iter=10000)

#create kfold cross validation instance
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#run k-fold cv and plot results.
#inputs are:
# 1 A title for your plot. here in the toy example it is "My plot", but you can change to any title
# 2 Your model. Any model that supports a basic fit function should work
# 3 your instance of StratifiedKFold
# 4 your X values (predictors or model inputs) in numpy array format
# 5 your Y label(s) in numpy array format
run_cv_and_plot_auc("My plot", logistic_regression_model, cv, train_x_np, train_y_np)
```
