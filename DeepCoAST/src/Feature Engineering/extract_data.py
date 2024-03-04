import numpy as np
from natsort import natsorted
from tqdm import tqdm
import glob
import pickle
from datetime import datetime
import argparse
from utility import *

parser = argparse.ArgumentParser()
parser.add_argument("-f", '--feature', type=str, default = "tam1d")
parser.add_argument("-o", '--output', type=str, default = "/scratch/TrafficSliver")
parser.add_argument("-i", '--input', type=str, default = "/scratch/TrafficSliver/TS_split3_raw")
parser.add_argument('-r', '--range', type=str, default="50_70")
parser.add_argument('-w', '--weight', type=str, default="0.33_0.33_0.34")
parser.add_argument("-m", '--path', type=int, default = 3)

INPUT_SIZE = 5000
SITE = 95

feature_dict = {
    'tiktok': getTiktok,
    'direction': getDirection,
    'tam1d': get1DTAM,
    'ipd': getIPD
}
def getFeature(feature, instance):
    feature_func = feature_dict.get(feature)
    return feature_func(instance)

def getMetadata(file_name):
    site, instance = map(int, file_name.split('/')[-1].split('.')[0].split('_')[0].split('-'))
    return site, instance

def extract(input_path, m, feature):
    total_path = []
    train_path = [[] for _ in range(m)]
    train_label = []
    test_path = [[] for _ in range(m)]
    test_label = []

    for i in range(m):
        total_path.append(natsorted(glob.glob(f"{input_path}/path{i}/*split*.cell")))
    test_check = [False] * SITE

    for i in range(len(total_path[0])):
        path_data = []
        flag = False
        for idx in range(m):
            with open(total_path[idx][i]) as f:
                path_data.append(getFeature(feature, f.readlines()))
                if len(path_data[idx]) < 10:
                    flag = True
                    break
                path_data[idx] = np.concatenate((path_data[idx], np.zeros(max(0, INPUT_SIZE - len(path_data[idx])))))[:INPUT_SIZE]
        
        if (flag): continue

        site, instance = getMetadata(total_path[0][i])
        if not test_check[site]:
            for idx in range(m):
                test_path[idx].append(path_data[idx])
            test_label.append(site)
            test_check[site] = True
        else:
            for idx in range(m):
                train_path[idx].append(path_data[idx])
            train_label.append(site)

    print("Total 데이터 개수:", (len(train_label)+len(test_label))*m)
    print("Train 데이터 개수:", len(train_path[0])*m)
    print("Test 데이터 개수:", len(test_path[0])*m)
    return train_path, test_path, train_label, test_label

if __name__ == "__main__":
    args = parser.parse_args()
    feature = args.feature
    output_path = args.output
    input_path = args.input
    r = args.range
    p = args.weight
    m = args.path

    input_path = input_path + "/" + r + "/" + p
    output_path = output_path + "/FINAL/ts" + str(m) + "/dataset/" + r + "/" + p + "/" + feature

    print(input_path)
    print(output_path)
    train_path, test_path, train_label, test_label = extract(input_path, m, feature=feature)

    for idx in range(m):
        with open(f"{output_path}/train_path{idx}.pkl", 'wb') as f:
            pickle.dump(train_path[idx], f, pickle.HIGHEST_PROTOCOL)
        with open(f"{output_path}/test_path{idx}.pkl", 'wb') as f:
            pickle.dump(test_path[idx], f, pickle.HIGHEST_PROTOCOL)
    with open(output_path + '/train_label.pkl', 'wb') as f:
            pickle.dump(train_label, f, pickle.HIGHEST_PROTOCOL)
    with open(output_path + '/test_label.pkl', 'wb') as f:
            pickle.dump(test_label, f, pickle.HIGHEST_PROTOCOL)

    f = open(output_path+"info.txt", 'w')
    f.write(str(datetime.now()))
    f.write('\n')
    #f.write("# total data: %d \n" % idx)
    print("# train data: %d \n" % len(train_path[0]))
    print("# test data: %d \n" % len(test_path[0]))
    f.write(str(test_path[0][:5]))
    f.write("\n****************************************\n")
    f.write(str(test_label[:5]))
    f.close()