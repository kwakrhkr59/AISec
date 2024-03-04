import pickle
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser()
parser.add_argument("-d", '--directory', nargs='+', help='directory to open', default='/data/TrafficSliver/BigEnough/splitted/unmon19/mode3')
parser.add_argument("-p", '--path', nargs='+', help='pickle file to open', default='/data/TrafficSliver/BigEnough/splitted/unmon19/mode3/unmon19_directions.pickle')
parser.add_argument("-o", "--outfolder", type=str, help="Folder for output files", default='/data/TrafficSliver/BigEnough/splitted/unmon19/mode3')
MAX_SITE=95

def merge_mon_data(path, outfolder):
    X=[]
    y=[]
    with open(path, 'rb') as f:
        dir_array=pickle.load(f)
        for i in range(len(dir_array)):
            print(i, len(dir_array[i]), end='\t')
            for j in range(len(dir_array[i])):
                instance=sequence.pad_sequences(dir_array[i][j], padding='post', maxlen=5000)
                X.append(np.array(instance))
                y.append(i)
            print(X[i].shape)
    X=np.array(X)
    y=np.array(y)
    print(X.shape, y.shape)
    with open(outfolder+'/X_total_mon.pkl', 'wb') as f:
        pickle.dump(X, f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/y_total_mon.pkl', 'wb') as f:
        pickle.dump(y, f, pickle.HIGHEST_PROTOCOL)
    print("pickle file complite")

def merge_unmon_data(path, outfolder):
    X=[]
    y=[]
    index=-1
    with open(path, 'rb') as f:
        dir_array=pickle.load(f)
        for i in range(len(dir_array)):
            instance=sequence.pad_sequences(dir_array[i], padding='post', maxlen=5000)
            X.append(np.array(instance))
            y.append(index)
    X=np.array(X)
    y=np.array(y)
    print(X.shape)
    print(y.shape)
    print(y)
    with open(outfolder+'/X_total_unmon19.pkl', 'wb') as f:
        pickle.dump(X, f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/y_total_unmon19.pkl', 'wb') as f:
        pickle.dump(y, f, pickle.HIGHEST_PROTOCOL)

def get_origin_data(path, outfolder):
    X=[]
    y=[]
    with open(path, 'rb') as f:
        dir_array=pickle.load(f)
        for i in range(len(dir_array)):
            print(i, end='\t')
            instance=sequence.pad_sequences(dir_array[i], padding='post', maxlen=5000)
            for inst in instance: 
                print(inst)
                X.append(np.array(inst))
            for _ in range(len(dir_array[i])):
                y.append(i)
    X=np.array(X)
    y=np.array(y)
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    with open(outfolder+'/X_origin_mon.pkl', 'wb') as f:
        pickle.dump(X, f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/y_origin_mon.pkl', 'wb') as f:
        pickle.dump(y, f, pickle.HIGHEST_PROTOCOL)

def merge_unmon_data(path, outfolder):
    X=[]
    y=[]
    index=-1
    with open(path, 'rb') as f:
        dir_array=pickle.load(f)
        for i in range(len(dir_array)):
            instance=sequence.pad_sequences(dir_array[i], padding='post', maxlen=5000)
            X.append(np.array(instance))
            y.append(index)
    X=np.array(X)
    y=np.array(y)
    print(X.shape)
    print(y.shape)
    print(y)
    
    with open(outfolder+'/X_total_unmon19.pkl', 'wb') as f:
        pickle.dump(X, f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/y_total_unmon19.pkl', 'wb') as f:
        pickle.dump(y, f, pickle.HIGHEST_PROTOCOL)
        
def split_mon_data(directory, outfolder):
    X_path=directory+'/X_total_mon.pkl'
    y_path=directory+'/y_total_mon.pkl'
    with open(X_path, 'rb') as f:
        X=pickle.load(f)
    with open(y_path, 'rb') as f:
        y=pickle.load(f)
    
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.1)
    X_train, X_valid, y_train, y_valid=train_test_split(X_train, y_train, test_size=0.1)
    print("train: ", X_train.shape, end='\t')
    print(y_train.shape)
    print("valid: ", X_valid.shape, end='\t')
    print(y_valid.shape)
    print("test: ", X_test.shape, end='\t')
    print(y_test.shape)

    with open(outfolder+'/X_train_mon.pkl', 'wb') as f:
        pickle.dump(X_train, f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/y_train_mon.pkl', 'wb') as f:
        pickle.dump(y_train, f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/X_test_mon.pkl', 'wb') as f:
        pickle.dump(X_test, f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/y_test_mon.pkl', 'wb') as f:
        pickle.dump(y_test, f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/X_valid_mon.pkl', 'wb') as f:
        pickle.dump(X_valid, f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/y_valid_mon.pkl', 'wb') as f:
        pickle.dump(y_valid, f, pickle.HIGHEST_PROTOCOL)

def split_unmon_data(directory, outfolder):
    X_path=directory+'/X_total_unmon19.pkl'
    y_path=directory+'/y_total_unmon19.pkl'
    with open(X_path, 'rb') as f:
        X=pickle.load(f)
    with open(y_path, 'rb') as f:
        y=pickle.load(f)
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.1)
    X_train, X_valid, y_train, y_valid=train_test_split(X_train, y_train, test_size=0.1)
    print("train: ", X_train.shape, end='\t')
    print(y_train.shape)
    print("valid: ", X_valid.shape, end='\t')
    print(y_valid.shape)
    print("test: ", X_test.shape, end='\t')
    print(y_test.shape)
    
    print(y_train)
    print(y_valid)
    print(y_test)
    with open(outfolder+'/X_train_unmon19.pkl', 'wb') as f:
        pickle.dump(X_train, f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/y_train_unmon19.pkl', 'wb') as f:
        pickle.dump(y_train, f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/X_test_unmon19.pkl', 'wb') as f:
        pickle.dump(X_test, f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/y_test_unmon19.pkl', 'wb') as f:
        pickle.dump(y_test, f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/X_valid_unmon19.pkl', 'wb') as f:
        pickle.dump(X_valid, f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/y_valid_unmon19.pkl', 'wb') as f:
        pickle.dump(y_valid, f, pickle.HIGHEST_PROTOCOL)

def multiply(directory, outfolder):
    mode='openworld'
    # X_train_path=directory+'X_train_'+mode+'.pkl'
    # X_valid_path=directory+'X_valid_'+mode+'.pkl'
    # X_test_path=directory+'X_test_'+mode+'.pkl'
    # y_train_path=directory+'y_train_'+mode+'.pkl'
    # y_valid_path=directory+'y_valid_'+mode+'.pkl'
    # y_test_path=directory+'y_test_'+mode+'.pkl'

    X_train_path=directory+'X_train.pkl'
    X_valid_path=directory+'X_valid.pkl'
    X_test_path=directory+'X_test.pkl'
    y_train_path=directory+'y_train.pkl'
    y_valid_path=directory+'y_valid.pkl'
    y_test_path=directory+'y_test.pkl'


    with open(X_train_path, 'rb') as f:
        X_train=pickle.load(f)
    with open(X_valid_path, 'rb') as f:
        X_valid=pickle.load(f)
    with open(X_test_path, 'rb') as f:
        X_test=pickle.load(f)
    with open(y_train_path, 'rb') as f:
        y_train=pickle.load(f)
    with open(y_valid_path, 'rb') as f:
        y_valid=pickle.load(f)
    with open(y_test_path, 'rb') as f:
        y_test=pickle.load(f)
        
    X_train_x5=[]
    X_valid_x5=[]
    X_test_x5=[]
    y_train_x5=[]
    y_valid_x5=[]
    y_test_x5=[]
    
    for X in X_train:
        for p in X: X_train_x5.append(p)
    for X in X_valid:
        for p in X: X_valid_x5.append(p)
    for X in X_test:
        for p in X: X_test_x5.append(p)
    for y in y_train:
        for _ in range(5): y_train_x5.append(y)
    for y in y_valid:
        for _ in range(5): y_valid_x5.append(y)
    for y in y_test:
        for _ in range(5): y_test_x5.append(y)
    X_train_x5=np.array(X_train_x5)
    X_valid_x5=np.array(X_valid_x5)
    X_test_x5=np.array(X_test_x5)
    y_train_x5=np.array(y_train_x5)
    y_valid_x5=np.array(y_valid_x5)
    y_test_x5=np.array(y_test_x5)
    print("train: ", X_train_x5.shape, y_train_x5.shape)
    print("valid: ", X_valid_x5.shape, y_valid_x5.shape)
    print("test: ", X_test_x5.shape, y_test_x5.shape)

    outfolder=outfolder+'/x5'
    with open(outfolder+'/X_train_'+mode+'_x5.pkl', 'wb') as f:
        pickle.dump(np.array(X_train_x5), f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/X_valid_'+mode+'_x5.pkl', 'wb') as f:
        pickle.dump(np.array(X_valid_x5), f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/X_test_'+mode+'_x5.pkl', 'wb') as f:
        pickle.dump(np.array(X_test_x5), f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/y_train_'+mode+'_x5.pkl', 'wb') as f:
        pickle.dump(np.array(y_train_x5), f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/y_valid_'+mode+'_x5.pkl', 'wb') as f:
        pickle.dump(np.array(y_valid_x5), f, pickle.HIGHEST_PROTOCOL)
    with open(outfolder+'/y_test_'+mode+'_x5.pkl', 'wb') as f:
        pickle.dump(np.array(y_test_x5), f, pickle.HIGHEST_PROTOCOL)


def openPickle(directory):
    # X_path=directory+'/X_total_unmon.pkl'
    y_path=directory+'/y_train.pkl'
    # y_path=directory+'/y_valid_unmon19.pkl'
    # y_path=directory+'/y_test_unmon19.pkl'
    # with open(X_path, 'rb') as f:
    #     X=pickle.load(f)
    #     print(X)
    with open(y_path, 'rb') as f:
        y=pickle.load(f)
        print(y.shape, y, end='    ')
        idx=np.argmax(y)
        print(y[idx])
    y_path=directory+'/y_valid.pkl'
    with open(y_path, 'rb') as f:
        y=pickle.load(f)
        print(y.shape, y, end='    ')
        idx=np.argmax(y)
        print(y[idx])
    X_path=directory+'/X_valid.pkl'
    with open(X_path, 'rb') as f:
        X=pickle.load(f)
        print(X)
        print(X.shape)

    y_path=directory+'/y_test.pkl'
    with open(y_path, 'rb') as f:
        y=pickle.load(f)
        print(y.shape, y, end='    ')
        idx=np.argmax(y)
        print(y[idx])


if __name__=="__main__":
    args = parser.parse_args()
    directory = args.directory
    pickle_file = args.path
    outfolder = args.outfolder
    # get_y(pickle_file, outfolder)
    # merge_mon_data(pickle_file, outfolder)
    # merge_unmon_data(pickle_file, outfolder)
    # split_mon_data(directory, outfolder)
    split_unmon_data(directory, outfolder)
    # openPickle(directory)
    # multiply(directory, outfolder)