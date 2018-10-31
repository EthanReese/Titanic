#Calculate the probability of death based on certain factors for titanic passengers
#Created by: Ethan Reese
#October 18, 2018

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = XGBRegressor(n_estimators = 1000, learning_rate = max_leaf_nodes)
    model.fit(train_X,train_y, early_stopping_rounds=15, eval_set=[(val_X, val_y)])
    val_predictions = model.predict(val_X)
    return mean_absolute_error(val_y, val_predictions)

data = pd.read_csv('/Users/Ethan/Devel/data/Titanic/train.csv')

data = data.assign(Sex1= [0] * data['Sex'].size)

for i in range(len(data['Sex'])):
    if data['Sex'][i]  == 'male':
        data['Sex1'][i] = 0
    else:
        data['Sex1'][i] = 1

features = [ 'Sex1', 'Age', 'Parch', 'Fare', 'Pclass']

data = data.dropna(axis=0)

X = data[features]
y = data.Survived
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

my_imputer = SimpleImputer()
train_X = my_imputer.fit_transform(train_X)
val_X = my_imputer.transform(val_X)

best = 10.0
bestInd = 0.0

for i in np.arange(0.01, 0.3, 0.001):
    if get_mae(i, train_X, val_X, train_y, val_y) < best:
            best = get_mae(i, train_X, val_X, train_y, val_y)
            bestInd = i
print(best)
print(bestInd)
