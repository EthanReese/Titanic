import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.pipeline import make_pipeline
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12,4

def substrings_in_string(big_string, substrings):
        for substring in substrings:
                if substring in big_string:
                        return substring
        return np.nan

def fit_model(alg, dtrain, predictors, useTrainCV = True, cv_folds = 5, early_stopping_rounds=50):
    
    #if we want to train the model at this time
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain["Survived"].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, early_stopping_rounds=early_stopping_rounds, verbose_eval=True, metrics='auc')
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm to data
    alg.fit(dtrain[predictors], dtrain["Survived"], eval_metric='auc')

    #Predict based on training set
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    #Generate a mode report
    print ("\nModel Report")
    print ("Accuracy: %.4g" % metrics.balanced_accuracy_score(dtrain['Survived'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Survived'], dtrain_predprob))

    #figure out the importance of each feature
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
"""
def get_error(parameter, X, y):
    #Make a pipeline that can run the machine learning model
    max_depth_input = 5
    min_child_weight_input = 1
    gamma_input = 0
    subsample_input = 0.8
    colsample_bytree_input = 0.8
    scale_pos_weight_input = 1
    my_pipeline = make_pipeline(SimpleImputer(), XGBClassifier(n_estimators = 1000, 
        learning_rate = 0.1, early_stopping_rounds = 5, objective="binary:logistic", gamma = 0, max_depth = 3, min_child_weight = 0, subsample = 0.947))
    #Split the data apart into folds
    kfold = KFold(n_splits=10, random_state = 1) 
    results = cross_val_score(my_pipeline, X, y, cv = kfold, scoring = 'neg_mean_squared_error')
    return np.mean(results)"""


#Read in the data to train and validate the model
data = pd.read_csv('/Users/Ethan/Devel/data/Titanic/train.csv')

#Engineer the sex feature from strings
data = data.assign(Sex1 = [0] * data['Sex'].size)

for i in range(len(data['Sex'])):
    if data['Sex'][i] == 'male':
        data['Sex1'][i] = 0
    else:
        data['Sex1'][i] = 1

#Engineer Deck from cabin number
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
data['Deck']= list(map(lambda cabin: (substrings_in_string(cabin, cabin_list)), data['Cabin']))
print(data['Deck'])
#Engineer the family_size
data['Family_Size'] = data['SibSp']+data['Parch']

#Pick the Model's features
features = ['Sex1', 'Age', 'Parch', 'Fare', 'Pclass', 'SibSp']

features_new = [x for x in data.columns if x not in ["Survived", "PassengerID", "Sex", "Name", "Embarked","Cabin", "Ticket"]]
data = data.dropna(axis = 0)

X = data[features]
y = data.Survived

xgb1 = XGBClassifier(
        learning_rate = 0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma = 0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread = 4,
        scale_pos_weight=1,
        seed=27)
xgb2 = XGBClassifier(
        learning_rate=0.127,
        n_estimators=1000,
        max_depth=3,
        min_child_weight=0,
        gamma=0,
        subsample=0.947,
        objective='binary:logistic'
)
fit_model(xgb1, data, features_new)
fit_model(xgb2, data, features_new)

#Initial attempt to tune max_depth and min_child weight

param_test1 = {
        'max_depth':[5],
        'min_child_weight':[1]
}

gsearch1 = GridSearchCV(XGBClassifier(learning_rate = 0.1,
        n_estimators=4,
        max_depth=5,
        min_child_weight=1,
        gamma = 0,
        subsample=0.8,
        colsample_bytree=0.8, objective= 'binary:logistic', 
        nthread = 4, scale_pos_weight=1, seed=27),
        param_grid=param_test1, 
        scoring='balanced_accuracy', 
        n_jobs=-1, 
        cv=5, iid=True)
        
gsearch1.fit(data[features_new], data['Survived'])
predictions = gsearch1.predict(data[features_new])
predprob = gsearch1.predict_proba(data[features_new])
print(metrics.balanced_accuracy_score(data['Survived'], predictions))
print(metrics.roc_auc_score(data['Survived'], predictions))
print(gsearch1.best_params_, gsearch1.best_score_, gsearch1.best_estimator_)
print(xgb2)
#Optimize the Gamma value
"""
param_test3 = {
        'gamma':[i/10.0 for i in range(0,10)]
}
gsearch3 = GridSearchCV(esimator=XGBClassifier(learning_rate = 0.1,
        n_estimators=65,
        max_depth=1,
        min_child_weight=1,
        gamma = 0,
        subsample=0.8,
        colsample_bytree=0.8, objective= 'binary:logistic', nthread = 4, scale_pos_weight=1, seed=27), param_grid=param_test3, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch3.fit(data[features_new], data['Survived'])

print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)
"""

#Loop through with multiple parameters
"""best = -100000.0
best_index = 0.0
for i in np.arange(0, 1, .0001): 
    num = get_error(i, X, y)
    if num > best:
        best = num
        best_index = i
print(best)
print(best_index)"""