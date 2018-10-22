#Calculate the probability of death based on certain factors for titanic passengers
#Created by: Ethan Reese
#October 18, 2018

import pandas as pd

data = pd.read_csv('C:/Users/298970/train.csv')

features = ["pclass", "sex", "Age", "parch"]

X = data[features];