from natsort import natsorted
import glob
import argparse
import numpy as np
import pickle
import os
parser = argparse.ArgumentParser()
parser.add_argument("-t", '--traces', nargs='+', help='folder with TrafficSliver output(.cell)', default='/data/TrafficSliver/DeepCoffea/splitted/CrawlE_Proc/inflow/mode2/ts')
parser.add_argument("-p", '--path', nargs='+', help='pickle file to open', default='/data/TrafficSliver/BigEnough/splitted/unmon19/mode3/unmon19_directions.pickle')
parser.add_argument("-o", "--outfolder", type=str, help="Folder for output files", default='/data/TrafficSliver/DeepCoffea/splitted/CrawlE_Proc/inflow/mode2') 

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def getDirection(instance):
    dir_array=[]
    instance = instance.read().split('\n')[:-1]
    for line in instance:
        dir=int(line.split('\t')[1])
        dir_array.append(int(dir/abs(dir)))
    return np.array(dir_array)

def mergeFiles(traces, outfolder):
    traces_file = natsorted(glob.glob(traces+'/*'))  # traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))
    dir_data=[]
    site_data=[]
    for f in traces_file:
        if("join" in f): continue
        trace = open(f,'r')
        # site, instance=map(int, f.split('/')[-1].split('-'))  # cell 파일 내부 형식에 따라 적절히 변환
        site, instance=f.split('/')[-1].split('_')[:2]
        site=int(site)
        instance=int(instance.split('_')[0])
        if(site!=0 and instance==0):
            dir_data.append(np.array(site_data, dtype=object))
            site_data=[]
        site_data.append(getDirection(trace))
    if(site_data): dir_data.append(np.array(site_data, dtype=object))
    # print(dir_array)
    dir_array=np.array(dir_data, dtype=object)
    # save as pickle file
    outfile=outfolder+'/'+'origin_mon.pkl'
    with open(outfile, 'wb') as f:
        pickle.dump(dir_array, f, pickle.HIGHEST_PROTOCOL)
    print("pickle file complete")

def mergeMon(traces, outfolder):
    traces_file = natsorted(glob.glob(traces+'/*.cell'))  # traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))
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
        path_data.append(getDirection(trace))
    if(path_data): instance_data.append(np.array(path_data, dtype=object))
    if(instance_data): dir_data.append(np.array(instance_data, dtype=object))
    dir_array=np.array(dir_data, dtype=object)
    # save as pickle file
    outfile=outfolder+'/'+'mon_directions.pickle'
    with open(outfile, 'wb') as f:
        pickle.dump(dir_array, f, pickle.HIGHEST_PROTOCOL)
    print("pickle file complete")

def mergeUnmon(traces, outfolder):
    traces_file = natsorted(glob.glob(traces+'/*.cell'))  # traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))
    dir_data=[]
    path_data=[]
    for f in traces_file:
        if("join" in f): continue
        trace = open(f,'r')
        # print(f)
        site, type, path=f.split('/')[-1].split('_')
        site=int(site.split('-')[0])
        path=int(path.split('.')[0])
        # append full dataset of one instance to dir_data
        if(site!=0 and path==0):
            print(f, len(path_data))
            dir_data.append(np.array(path_data, dtype=object))
            path_data=[]
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

def openPickle():
    handle = open('/data/DF/TF/dataset/y_test_TF.pkl', 'rb')
    directions = pickle.load(handle, encoding = 'latin1')
    directions=np.array(directions)
    print(len(directions))
    print(directions)

if __name__ == '__main__':
    args = parser.parse_args()
    traces = args.traces
    pickle_file = args.path
    outfolder = args.outfolder
    mergeFiles(traces, outfolder)
    # mergeMon(traces, outfolder)
    # mergeUnmon(traces, outfolder)
    # openPickleDataset1(pickle_file)
    # openPickleDataset2()