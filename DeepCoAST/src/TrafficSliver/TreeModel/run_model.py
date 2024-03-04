from models import *
import pickle5 as pickle
import numpy as np
NFOLDS = 3
SEED = 0
NROWS = None


# Load data for non-defended dataset for CW setting
def LoadDataNoDefCW():

    print("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = '/data/TrafficSliver/BigEnough/splitted/mon/mode4/x5/'


    handle = open(dataset_dir + 'X_train_mon_x5.pkl', 'rb')
    X_train = np.array(pickle.load(handle, encoding='latin1'))
    handle = open(dataset_dir + 'y_train_mon_x5.pkl', 'rb')
    y_train = np.array(pickle.load(handle, encoding='latin1'))
    
    handle = open(dataset_dir + 'X_valid_mon_x5.pkl', 'rb')
    X_valid = np.array(pickle.load(handle, encoding='latin1'))
    handle = open(dataset_dir + 'y_valid_mon_x5.pkl', 'rb')
    y_valid = np.array(pickle.load(handle, encoding='latin1'))
    
    handle = open(dataset_dir + 'X_test_mon_x5.pkl', 'rb')
    X_test = np.array(pickle.load(handle, encoding='latin1'))
    handle = open(dataset_dir + 'y_test_mon_x5.pkl', 'rb')
    y_test = np.array(pickle.load(handle, encoding='latin1'))

    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataNoDefCW()
ntrain = X_train.shape[0]
ntest = X_test.shape[0]

# X_train = np.squeeze(X_train, axis=1)
# print(X_train.shape)
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)



kf = KFold(n_splits = NFOLDS, shuffle=True, random_state=SEED)

def get_oof(clf):
  oof_train = np.zeros((ntrain,))
  oof_test = np.zeros((ntest,))
  oof_test_skf = np.empty((NFOLDS, ntest))

  for i, (train_index, test_index) in enumerate(kf.split(X_train)):
    X_tr = X_train.loc[train_index]
    y_tr = y_train.loc[train_index]
    X_te = X_train.loc[test_index]

    clf.train(X_tr, y_tr)

    oof_train[test_index] = clf.predict(X_te)
    oof_test_skf[i, :] = clf.predict(X_test)
  
  oof_test[:] = oof_test_skf.mean(axis=0)
  return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

et_params = {
    'n_jobs': 16,
    'n_estimators': 200,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 200,
    'max_features': 0.2,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'binary:logistic',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'nrounds': 200
}

catboost_params = {
    'iterations': 200,
    'learning_rate': 0.5,
    'depth': 3,
    'l2_leaf_reg': 40,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.7,
    'scale_pos_weight': 5,
    'eval_metric': 'AUC',
    'od_type': 'Iter',
    'allow_writing_files': False
}

lightgbm_params = {
    'n_estimators':200,
    'learning_rate':0.1,
    'num_leaves':123,
    'colsample_bytree':0.8,
    'subsample':0.9,
    'max_depth':15,
    'reg_alpha':0.1,
    'reg_lambda':0.1,
    'min_split_gain':0.01,
    'min_child_weight':2    
}


# xg = XgbWrapper(seed=SEED, params=xgb_params)
print("ExtraTreesClassifier training start")
et = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
et_oof_train, et_oof_test = get_oof(et)
print("ET-CV: {}".format(sqrt(mean_squared_error(y_train, et_oof_train))))

print("RandomforestClassifier training start")
rf = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
rf_oof_train, rf_oof_test = get_oof(rf)
print("RF-CV: {}".format(sqrt(mean_squared_error(y_train, rf_oof_train))))
