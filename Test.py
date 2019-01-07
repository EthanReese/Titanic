#Take an already optimized model and predict based on input data
#Created by:Ethan Reese
#December 21, 2018

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

#Read in both the data to train and test the model
data = pd.read_csv('/Users/Ethan/Devel/data/Titanic/train.csv')
testData = pd.read_csv('/Users/Ethan/Devel/data/Titanic/test.csv')

#Create the gender feature through feature engineering
data = data.assign(Sex1 = [0] * data['Sex'].size)
testData = testData.assign(Sex1 = [0] * testData['Sex'].size)

for i in range(len(data['Sex'])):
    if data['Sex'][i] == 'male':
        testData['Sex1'][i] = 0
    else:
        testData['Sex1'][i] = 1

for i in range(len(testData['Sex'])):
    if testData['Sex'][i] == 'male':
        testData['Sex1'][i] = 0
    else:
        testData['Sex1'][i] = 1

#Pick the features that will be used by the model
features = ['Sex1', 'Age', 'Parch', 'Fare', 'Pclass', 'SibSp']

data = data.dropna(axis=0)

X = data[features]
y = data.Survived

#Make a train test split for validation
X_train, X_test, y_train, y_test = train_test_split(X,y)

val_X = testData[features]

#Make a pipeline that can do everything
my_pipeline = make_pipeline(SimpleImputer(), XGBRegressor(n_estimators = 10000, learning_rate = 0.127, early_stopping_rounds = 5, objective="binary:logistic", gamma = 0, max_depth = 3, min_child_weight = 0, subsample = 0.947, tree_method='exact'))
my_pipeline.fit(X,y)
survived_raw = my_pipeline.predict(val_X)
survived = list(map(round, survived_raw))

#Save the results into a two dimensional array and output it into CSV
data_set = pd.DataFrame({'PassengerId': testData['PassengerId'], 'Survived': survived})
data_set.to_csv('/Users/Ethan/Devel/data/Titanic/submit.csv')
