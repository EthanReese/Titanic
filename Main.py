#Calculate the probability of death based on certain factors for titanic passengers
#Created by: Ethan Reese
#October 18, 2018

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('/Users/Ethan/Devel/data/Titanic/train.csv')

np.append(data, np.zeros((data['Sex'].length, 1), dtype=int64)

for i in data['Sex']:
    if data['Sex'][i] == "female":
        data['Sex'][i] = 0
    else:
        X['Sex'][i] = 1


features = ['Pclass', 'Sex', 'Age', 'Parch']

X = data[features]

y = data.Survived

data = data.dropna(axis=0)

model = DecisionTreeRegressor(random_state = 1)

#model.fit(X,y)

X.head()

#model.predict(X.head())
