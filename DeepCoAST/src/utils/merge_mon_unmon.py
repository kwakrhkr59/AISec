import pickle
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("-m", '--mon_directory', nargs='+', help='directory to open', default='/data/ts-final/BigEnough/path5/mon')
parser.add_argument("-u", '--unmon_directory', nargs='+', help='directory to open', default='/data/ts-final/BigEnough/path5/unmon')
parser.add_argument("-p", '--path', nargs='+', help='pickle file to open', default='/data/TrafficSliver/BigEnough/splitted/unmon19/unmon19_directions.pickle')
parser.add_argument("-o", "--outfolder", type=str, help="Folder for output files", default='/data/ts-final/BigEnough/path5') 

def merge_mon_unmon(mon_dir, unmon_dir, outfolder):
    X_train_mon_path=mon_dir+'/X_train_mon.pkl'
    X_valid_mon_path=mon_dir+'/X_valid_mon.pkl'
    X_test_mon_path=mon_dir+'/X_test_mon.pkl'
    y_train_mon_path=mon_dir+'/y_train_mon.pkl'
    y_valid_mon_path=mon_dir+'/y_valid_mon.pkl'
    y_test_mon_path=mon_dir+'/y_test_mon.pkl'
    X_train_unmon_path=unmon_dir+'/X_train_unmon.pkl'
    X_valid_unmon_path=unmon_dir+'/X_valid_unmon.pkl'
    X_test_unmon_path=unmon_dir+'/X_test_unmon.pkl'
    y_train_unmon_path=unmon_dir+'/y_train_unmon.pkl'
    y_valid_unmon_path=unmon_dir+'/y_valid_unmon.pkl'
    y_test_unmon_path=unmon_dir+'/y_test_unmon.pkl'
    
    with open(X_train_mon_path, 'rb') as f:
        X_train_mon=pickle.load(f)
    with open(X_valid_mon_path, 'rb') as f:
        X_valid_mon=pickle.load(f)
    with open(X_test_mon_path, 'rb') as f:
        X_test_mon=pickle.load(f)
    with open(y_train_mon_path, 'rb') as f:
        y_train_mon=pickle.load(f)
    with open(y_valid_mon_path, 'rb') as f:
        y_valid_mon=pickle.load(f)
    with open(y_test_mon_path, 'rb') as f:
        y_test_mon=pickle.load(f)
    with open(X_train_unmon_path, 'rb') as f:
        X_train_unmon=pickle.load(f)
    with open(X_valid_unmon_path, 'rb') as f:
        X_valid_unmon=pickle.load(f)
    with open(X_test_unmon_path, 'rb') as f:
        X_test_unmon=pickle.load(f)
    with open(y_train_unmon_path, 'rb') as f:
        y_train_unmon=pickle.load(f)
    with open(y_valid_unmon_path, 'rb') as f:
        y_valid_unmon=pickle.load(f)
    with open(y_test_unmon_path, 'rb') as f:
        y_test_unmon=pickle.load(f)
    
    X_train=[]
    y_train=[]
    X_valid=[]
    y_valid=[]
    X_test=[]
    y_test=[]
    
    for X, y in zip(X_train_mon, y_train_mon):
        X_train.append(X)
        y_train.append(y)
    for X, y in zip(X_train_unmon, y_train_unmon):
        X_train.append(X)
        y_train.append(y)
    for X, y in zip(X_valid_mon, y_valid_mon):
        X_valid.append(X)
        y_valid.append(y)
    for X, y in zip(X_valid_unmon, y_valid_unmon):
        X_valid.append(X)
        y_valid.append(y)
    for X, y in zip(X_test_mon, y_test_mon):
        X_test.append(X)
        y_test.append(y)
    for X, y in zip(X_test_unmon, y_test_unmon):
        X_test.append(X)
        y_test.append(y)
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_valid=np.array(X_valid)
    y_valid=np.array(y_valid)
    X_test=np.array(X_test)
    y_test=np.array(y_test)
    # print(X_train_mon.shape, y_train_unmon19.shape, X_train.shape, y_train.shape)
    # print(X_valid_mon.shape, y_valid_unmon19.shape, X_valid.shape, y_valid.shape)
    # print(X_test_mon.shape, y_test_unmon19.shape, X_test.shape, y_test.shape)

    train_len=np.arange(X_train.shape[0])
    valid_len=np.arange(X_valid.shape[0])
    test_len=np.arange(X_test.shape[0])
    np.random.shuffle(train_len)
    np.random.shuffle(valid_len)
    np.random.shuffle(test_len)

    X_train=X_train[train_len]
    y_train=y_train[train_len]
    X_valid=X_valid[valid_len]
    y_valid=y_valid[valid_len]
    X_test=X_test[test_len]
    y_test=y_test[test_len]
    print(X_train.shape, y_train.shape)
    print(X_valid.shape, y_valid.shape)
    print(X_test.shape, y_test.shape)
    
    with open(outfolder+'/X_train.pkl', 'wb') as f:
        pickle.dump(np.array(X_train), f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/y_train.pkl', 'wb') as f:
        pickle.dump(np.array(y_train), f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/X_valid.pkl', 'wb') as f:
        pickle.dump(np.array(X_valid), f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/y_valid.pkl', 'wb') as f:
        pickle.dump(np.array(y_valid), f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/X_test.pkl', 'wb') as f:
        pickle.dump(np.array(X_test), f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/y_test.pkl', 'wb') as f:
        pickle.dump(np.array(y_test), f, pickle.HIGHEST_PROTOCOL)

# def test():
#     X=np.array([1,2,3,4,5])
#     y=np.array(['a','b','c','d','e'])
#     s=np.arange(5)
#     np.random.shuffle(s)
#     X=X[s]
#     y=y[s]
#     print(X)
#     print(y)

if __name__=="__main__":
    args = parser.parse_args()
    mon_dir = args.mon_directory
    unmon_dir = args.unmon_directory
    outfolder = args.outfolder
    merge_mon_unmon(mon_dir, unmon_dir, outfolder)