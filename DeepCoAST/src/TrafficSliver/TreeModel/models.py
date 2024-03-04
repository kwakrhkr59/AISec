import pandas as pd
import numpy as np
from scipy.stats import skew
import xgboost as xgb
from sklearn.model_selection import KFold #from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from math import sqrt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import gc

NFOLDS = 3
SEED = 0
NROWS = None

class SklearnWrapper(object):
  def __init__(self, clf, seed=0, params=None):
    params['random_state'] = seed
    self.clf = clf(**params)

  def train(self, x_train, y_train):
    self.clf.fit(x_train, y_train)

  def predict(self, x):
    return self.clf.predict_proba(x)[:,1]
  
class CatboostWrapper(object):
  def __init__(self, clf, seed=0, params=None):
    params['random_seed'] = seed
    self.clf = clf(**params)
  
  def train(self, x_train, y_train):
    self.clf.fit(x_train, y_train)
  
  def predict(self, x):
    return self.clf.predict_proba(x)[:, 1]
  
class LightGBMWrapper(object):
  def __init__(self, clf, seed=0, params=None):
    params['feature_fraction_seed'] = seed
    params['bagging_seed'] = seed
    self.clf = clf(**params)
  
  def train(self, x_train, y_train):
    self.clf.fit(x_train, y_train)
  
  def predict(self, x):
    return self.clf.predict_proba(x)[:,1]

class XgbWrapper(object):
  def __init__(self, seed=0, params=None):
    self.param = params
    self.param['seed'] = seed
    self.nrounds = params.pop('nrounds', 250)
  
  def train(self, x_train, y_train):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    self.gbdt = xgb.train(self.param, dtrain, self.nrounds)
  
  def predict(self, x):
    return self.gbdt.predict(xgb.DMatrix(x))
  