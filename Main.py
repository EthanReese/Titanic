#Calculate the probability of death based on certain factors for titanic passengers
#Created by: Ethan Reese
#October 18, 2018

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

data = pd.read_csv('/Users/Ethan/Devel/data/Titanic/train.csv')

data = data.assign(Sex1= [0] * data['Sex'].size)

for i in range(len(data['Sex'])):
    if data['Sex'][i]  == 'male':
        data['Sex1'][i] = 0
    else:
        data['Sex1'][i] = 1

features = ['Pclass', 'Sex1', 'Age', 'Parch']

data = data.dropna(axis=0)

X = data[features]

y = data.Survived

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

model = DecisionTreeRegressor(random_state = 1)

model.fit(train_X,train_y)

val_predictions = model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))
