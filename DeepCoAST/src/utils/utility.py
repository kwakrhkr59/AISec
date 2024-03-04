from natsort import natsorted
import glob
import argparse
import numpy as np
import pandas as pd
import h5py
import pickle5 as pickle
import os
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser()
parser.add_argument("-d", '--directory', nargs='+', help='directory to open', default='/data/TrafficSliver/wf-singlesite/mon_amazon/mode1/ts')
parser.add_argument("-t", '--traces', nargs='+', help='folder with TrafficSliver output(.cell)', default='/data/TrafficSliver/wf-singlesite/mon_amazon/mode1/ts')
parser.add_argument("-p", '--path', nargs='+', help='pickle file to open', default='/home/kwakrhkr59/Var-CNN/data/all_open_world.npz')
parser.add_argument("-o", "--outfolder", type=str, help="Folder for output files", default='/data/TrafficSliver/wf-singlesite/mon_amazon/mode1/DF') 

def get_max_len(path):
    file=open(path, 'rb')
    packets = pickle.load(file, encoding = 'latin1')
    packets=np.array(packets)
    maxlen=0
    for packet in packets:
        for path in packet:
            if(maxlen<len(path)): maxlen=len(path)
    print(maxlen)

def open_h5(path):
    file = h5py.File(path, 'r')
    keys = file.keys()
    print(keys)
    print(list(file['test_data']))
    

def open_npz(path):
    file = np.array(np.load(path))
    print(file.shape)
    print(file)

def open_pickle(path):
    handle = open(path, 'rb')
    # file = pickle.load(handle, encoding = 'latin1')
    file = pickle.load(handle)
    print(type(file))
    # print(file[0])
    print(np.array(file).shape)
    # print(file['label'])
    print(len(file['tor']))
    print(len(file['exit']))
    print(len(file['label']))
    new_file = open('tmp.txt', 'w')
    for line in file['label']:
        new_file.write(line)
    new_file.close()
    # print(directions.keys())
    # for k in directions.keys():
    #     print(k, len(directions[k]), len(directions[k][0]), directions[k][0])
    
    
    for arr in file[0]:
        print(len(arr))
    
    # for data in directions:
        # print(data.shape)

def open_cell(traces):
    traces_file = natsorted(glob.glob(traces+'/*.cell'))  # traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))]
    for f in traces_file:
        trace = open(f,'r')
        print(trace)

def getDirection(instance):
    dir_array=[]
    instance = instance.read().split('\n')[:-1]
    for line in instance:
        dir=int(line.split('\t')[1])
        dir_array.append(int(dir/abs(dir)))
    return np.array(dir_array)

def merge_cell_mon_5(traces, outfolder):
    traces_file = natsorted(glob.glob(traces+'/*.cell'))  # traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))
    addtime_data=[]
    time_data=[]
    direction_data=[]
    dir_data=[]
    instance_data=[]
    path_data=[]
    for f in traces_file:
        if("join" in f): continue
        trace = open(f,'r')
        site, instance=f.split('/')[-1].split('-')
        site=int(site)
        instance, type, path=instance.split('_')    # type: split / join
        instance=int(instance)
        path=int(path.split('.')[0])
        # append full dataset of one instance to dir_data
        if(instance!=0 and path==0):
            instance_data.append(np.array(path_data, dtype=object))
            path_data=[]
        if(site!=0 and instance==0 and path==0):
            instance_data.append(np.array(path_data, dtype=object))
            print(f, len(instance_data))
            dir_data.append(np.array(instance_data, dtype=object))
            instance_data=[]
            path_data=[]
        time_data, direction_data = getTimeDirection(trace)
        time_data_arr = np.array(time_data)
        dir_data_arr = np.array(direction_data)
        path_data.append(getAddtime(time_data_arr, dir_data_arr))
    if(path_data): instance_data.append(np.array(path_data, dtype=object))
    if(instance_data): dir_data.append(np.array(instance_data, dtype=object))
    dir_array=np.array(dir_data, dtype=object)
    # save as pickle file
    outfile=outfolder+'/'+'mon_directions.pickle'
    with open(outfile, 'wb') as f:
        pickle.dump(dir_array, f, pickle.HIGHEST_PROTOCOL)
    print("pickle file complete")

def merge_cell_mon_2(traces, outfolder):
    # dataset_dir = '/data/TrafficSliver/BigEnough/mon_standard'
    traces_file = natsorted(glob.glob(traces+'/*.cell'))  # traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))
    dir_data=[]
    instance_data=[]
    path_data=[]
    cnt=0
    for f in traces_file:
        # print(f)
        if("join" in f): continue
        trace = open(f,'r')
        site, instance=f.split('/')[-1].split('-')
        site=int(site)
        instance, _, path=instance.split('_')    # type: split / join
        instance = int(instance)
        path=int(path.split('.')[0])
        # site, instance, _, path = f.split('/')[-1].split('_')
        # print(site, instance, _)
        # site=int(site)
        # instance=int(instance)
        # path=int(path.split('.')[0])
        # append full dataset of one instance to dir_data
        if(instance!=0 and path==0):
            if(path_data[0].shape==path_data[1].shape):
                if(instance==0): origin_file = dataset_dir + '/' + str(site-1) + '-' + '199'
                else: origin_file = dataset_dir + '/' + str(site) + '-' + str(instance-1)
                origin = open(origin_file, 'r')
                print(f, path_data[0].shape, path_data[1].shape)
                print(origin_file, getDirection(origin).shape)
                print()
                cnt+=1
            else: 
                instance_data.append(np.array(path_data, dtype=object))
            path_data=[]
        if(site!=0 and instance==0 and path==0):
            instance_data.append(np.array(path_data, dtype=object))
            try:
                dir_data.append(np.array(instance_data, dtype=object))
            except:
                flag=True
                for inst in instance_data:
                    if(inst.shape!=(2, )): 
                        # print(inst.shape)
                        # print("=================(2, x)=================")
                        # print(inst, type(inst))
                        # print(inst[0], inst[0].shape, type(inst[0]))
                        # print(inst[1], inst[1].shape, type(inst[1]))
                        print(inst.shape, inst[0].shape, inst[1].shape)
                    # elif(flag):
                    else:
                        # print("=================(2, )=================")
                        # print(inst, type(inst))
                        # print(inst[0], inst[0].shape, type(inst[0]))
                        # print(inst[1], inst[1].shape, type(inst[1]))
                        print(inst.shape, inst[0].shape, inst[1].shape)
                        flag=False
            instance_data=[]
            path_data=[]
        d=getDirection(trace)
        path_data.append(np.array(d))
    if(path_data): instance_data.append(np.array(path_data, dtype=object).reshape(2, ))
    if(instance_data): dir_data.append(np.array(instance_data, dtype=object))
    dir_array=np.array(dir_data, dtype=object)
    #save as pickle file
    outfile=outfolder+'/'+'mon_directions.pickle'
    with open(outfile, 'wb') as f:
        pickle.dump(dir_array, f, pickle.HIGHEST_PROTOCOL)
    print("pickle file complete")
    print(cnt)

def merge_cell_unmon(traces, outfolder):
    dataset_dir = '/data/TrafficSliver/BigEnough/unmon_standard19'
    traces_file = natsorted(glob.glob(traces+'/*.cell'))  # traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))
    dir_data=[]
    path_data=[]
    cnt=0
    for f in traces_file:
        if("join" in f): continue
        trace = open(f,'r')
        # print(f)
        site, type, path=f.split('/')[-1].split('_')
        site=int(site.split('-')[0])
        path=int(path.split('.')[0])
        # append full dataset of one instance to dir_data
        if(site!=0 and path==0):
            if(path_data[0].shape==path_data[1].shape):
                origin_file = dataset_dir + '/' + str(site)
                origin = open(origin_file, 'r')
                print(f, path_data[0].shape, path_data[1].shape)
                print(origin_file, getDirection(origin).shape)
                print()
                path_data=[]
                cnt+=1
            else: 
                dir_data.append(np.array(path_data, dtype=object))
                path_data=[]
            # print(f, len(path_data))
        path_data.append(getDirection(trace))
    if(path_data): dir_data.append(np.array(path_data, dtype=object))
    dir_array=np.array(dir_data, dtype=object)
    print(dir_array.shape)
    print(dir_array[0].shape)
    # save as pickle file
    outfile=outfolder+'/'+'unmon19_directions.pickle'
    with open(outfile, 'wb') as f:
        pickle.dump(dir_array, f, pickle.HIGHEST_PROTOCOL)
    print("pickle file complete")
    print(cnt)

def get_data_mon(path, outfolder):
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

def get_data_unmon(path, outfolder):
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

def split_mon(directory, outfolder):
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

def split_unmon(directory, outfolder):
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
    if(mode=='openworld'):
        X_train_path=directory+'/X_train.pkl'
        X_valid_path=directory+'/X_valid.pkl'
        X_test_path=directory+'/X_test.pkl'
        y_train_path=directory+'/y_train.pkl'
        y_valid_path=directory+'/y_valid.pkl'
        y_test_path=directory+'/y_test.pkl'
    else:
        X_train_path=directory+'X_train_'+mode+'.pkl'
        X_valid_path=directory+'X_valid_'+mode+'.pkl'
        X_test_path=directory+'X_test_'+mode+'.pkl'
        y_train_path=directory+'y_train_'+mode+'.pkl'
        y_valid_path=directory+'y_valid_'+mode+'.pkl'
        y_test_path=directory+'y_test_'+mode+'.pkl'

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

def count_packet(traces):
    trace = natsorted(glob.glob(traces + "/*split*"))

    index = 0
    df = {}
    for trace_path in trace:
        index += 1
        if (index % 1000 == 0):
            print(index)

        with open(trace_path, 'r') as f:
            lines = f.readlines()
        df[trace_path.split('/')[-1].split('.')[0]] = len(lines)
    # print(df)
    df = pd.DataFrame.from_dict([df]).T
    df.to_csv("/home/kwakrhkr59/TrafficSliver/splitting_simulator/csv/amazon_timestamps_20.csv")

if __name__=="__main__":
    args = parser.parse_args()
    traces = args.traces
    directory = args.directory
    pickle_file = args.path
    outfolder = args.outfolder

    # count_packet(traces)
    # get_max_len('/data/TrafficSliver/BigEnough/splitted/unmon19/mode3/unmon19_directions.pickle')
    # get_max_len('/data/TrafficSliver/BigEnough/splitted/mon/mode3/mon_directions.pickle')
    # open_pickle('/home/kwakrhkr59/Deepcoffea/data/crawle_new_overlap_interval19000_win0_addn0_w_superpkt.pickle')
    open_pickle('/scratch/TrafficSliver/FINAL/ts3/dataset/50_70/0.33_0.33_0.34/tam1d/test_path0.pkl')
    # open_npz(pickle_file)
    # open_cell(traces)
    # 1단계: cell 파일들 -> directions.pickle
    # merge_cell_mon_2(traces, outfolder)
    # merge_cell_unmon(traces, outfolder)
    # 2단계: directions.pickle -> X.pkl, y.pkl
    # get_data_mon(pickle_file, outfolder)
    # get_data_unmon(pickle_file, outfolder)
    # 3단계: X.pkl, y.pkl -> train, valid, test
    # split_mon(directory, outfolder)
    # split_unmon(directory, outfolder)
    # multiply(directory, outfolder)