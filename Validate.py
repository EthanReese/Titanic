import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from xgboost.sklearn import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

def get_error(parameter, X, y):
    #Make a pipeline that can run the machine learning model
    my_pipeline = make_pipeline(SimpleImputer(), XGBRegressor(n_estimators = 100, learning_rate = 0.13, early_stopping_rounds = 5, objective="binary:hinge"))
    #Split the data apart into folds
    kfold = KFold(n_splits=10, random_state = 1) 
    results = cross_val_score(my_pipeline, X, y, cv = kfold, scoring = 'accuracy')
    return np.mean(results)


#Read in the data to train and validate the model
data = pd.read_csv('/Users/Ethan/Devel/data/Titanic/train.csv')

#Engineer the sex feature from strings
data = data.assign(Sex1 = [0] * data['Sex'].size)

for i in range(len(data['Sex'])):
    if data['Sex'][i] == 'male':
        data['Sex1'][i] = 0
    else:
        data['Sex1'][i] = 1

#Pick the Model's features
features = ['Sex1', 'Age', 'Parch', 'Fare', 'Pclass', 'SibSp']

data = data.dropna(axis = 0)

X = data[features]
y = data.Survived

#Loop through with multiple parameters
best = 0.0
best_index = 0.0
for i in np.arange(0.01, 0.011, 0.001):
    num = get_error(i, X, y)
    if num > best:
        best = num
        best_index = i
print(best)
print(best_index)
