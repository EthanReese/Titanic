#Calculate the probability of death based on certain factors for titanic passengers
#Created by: Ethan Reese
#October 18, 2018

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from xgboost.sklearn import XGBClassifier


#For Testing
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    """
    #Create a pipeline that processes the data and predicts
    #Pipelines are dumb
    my_pipeline = make_pipeline(SimpleImputer(), XGBRegressor(n_estimators = 1000, learning_rate = max_leaf_nodes))
    my_pipeline.fit(train_X,train_y, early_stopping_rounds=5, eval_set=[(val_X, val_y)])
    """

    model = XGBRegressor(n_estimators = 5000, learning_rate = max_leaf_nodes, objective="binary:hinge")
    model.fit(train_X, train_y, early_stopping_rounds = 5, eval_set=[(val_X, val_y)],  eval_metric="error")
    val_predictions = model.predict(val_X)
    survived = list(map(round, val_predictions))
    #Loop through and figure out the percent error
    
    """
    correct = 0
    for i in survived.size:
        if(survived[i] == val_y[i]):
            correct += 1
    return correct/(survived.size())
    """
    #Calculate the Balanced Error
    return balanced_accuracy_score(val_y,survived)

#Read in both the data to train the model and test the model
data = pd.read_csv('/Users/Ethan/Devel/data/Titanic/train.csv')
testData = pd.read_csv('/Users/Ethan/Devel/data/Titanic/test.csv')

#Create the sex feature using essentially one hot encoding
data = data.assign(Sex1= [0] * data['Sex'].size)
testData = testData.assign(Sex1= [0] * testData['Sex'].size)

for i in range(len(data['Sex'])):
    if data['Sex'][i]  == 'male':
        data['Sex1'][i] = 0
    else:
        data['Sex1'][i] = 1
for i in range(len(testData['Sex'])):
    if testData['Sex'][i]  == 'male':
        testData['Sex1'][i] = 0
    else:
        testData['Sex1'][i] = 1

#Appoint the features for the model
features = [ 'Sex1', 'Age', 'Parch', 'Fare', 'Pclass', 'SibSp']

data = data.dropna(axis=0)

X = data[features]
y = data.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y)

val_X = testData[features]


my_imputer = SimpleImputer()
X_train = my_imputer.fit_transform(X_train)
X_test = my_imputer.fit_transform(X_test)
val_X = my_imputer.transform(val_X)

#Create a pipeline that makes it work
#pipe = Pipeline(SimpleImputer(), XGBRegressor(n_estimators = 1000, learning_rate = max_leaf_nodes))


model = XGBRegressor(n_estimators = 5000, learning_rate = 0.25, objective="binary:hinge")
model.fit(X_train, y_train, early_stopping_rounds = 5, eval_set=[(X_test, y_test)], eval_metric="error")
guesses = model.predict(val_X)

survived = (list(map(round, guesses)))

print(survived)
#save the results into a two dimensional array
data_set = pd.DataFrame({'PassengerId': testData['PassengerId'], 'Survived': survived})

print(data_set)

data_set.to_csv('/Users/Ethan/Devel/data/Titanic/submit.csv')

#For Testing
"""
best = 0.0
bestInd = 0.0
for i in np.arange(0.01, 0.3, 0.001):
    if get_mae(i, X_train, X_test, y_train, y_test) > best:
            best = get_mae(i, X_train, X_test, y_train, y_test)
            bestInd = i
print(best)
print(bestInd)
"""
