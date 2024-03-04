from natsort import natsorted
import glob
import argparse
import numpy as np
import pickle
import os
# from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from os import listdir
import time
import sys
import math

parser = argparse.ArgumentParser()
parser.add_argument("-n", '--path_num', nargs=1, type=str, help='number of path', default = 5)
parser.add_argument("-m", '--mode', nargs=1, help='mon or unmon', required=True)
parser.add_argument("-f", '--feature', nargs=1, help='cumul or burst or interval_burst or twotime_burst', required=True)
parser.add_argument("-o", '--overlap', type=float, help='전체 tam overlap', default = 0.01)
parser.add_argument("-a", '--alpha', type=float, help='new overlap alpha', default = 0.01)
parser.add_argument("-i", '--interval', type=float, help='new overlap interval', default = 0.04)
parser.add_argument("-k", '--k', type=int, help='k', default = 11)

def getPunch5(k, instance):
    o_k = 0
    i_k = 0
    times = []
    sizes = []
    punch_num_o = 0
    punch_num_i = 0
    o_list = []
    i_list = []
    for line in map(str.strip, instance):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        if dir >0:
            o_k += 1
            if o_k == k:
                punch_num_o += 1
                o_list.append(punch_num_o)
                o_k=0
        if dir <0:
            i_k += 1
            if i_k == k:
                punch_num_i += 1
                i_list.append(-1*punch_num_i)
                i_k=0
        times.append(timestamp)
        sizes.append(dir)
    if times:
        pass
    else:
        return [0]
    return o_list+i_list


def getPunch4(k, instance): # intervalpunch
    o_k = 0
    i_k = 0
    times = []
    sizes = []
    o_list = []
    i_list = []
    beforetimestamp_i = 0
    beforetimestamp_o = 0
    for i, line in enumerate(map(str.strip, instance)):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        if dir >0:
            o_k += 1
            if o_k == k:
                interval= timestamp-beforetimestamp_o
                o_list.append(interval)
                o_k=0
                beforetimestamp_o = timestamp
        if dir <0:
            i_k += 1
            if i_k == k:
                interval= timestamp-beforetimestamp_i
                i_list.append(-1*interval)
                i_k=0
                beforetimestamp_i = timestamp
    if times:
        pass
    else:
        return [0]
    return o_list+i_list


def getPunch3(k, instance):
    o_k = 0
    i_k = 0
    times = []
    sizes = []
    o_list = []
    i_list = []
    for i, line in enumerate(map(str.strip, instance)):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        if dir >0:
            o_k += 1
            if o_k == k:
                o_list.append(i)
                o_k=0
        if dir <0:
            i_k += 1
            if i_k == k:
                i_list.append(-1*i)
                i_k=0
        times.append(timestamp)
        sizes.append(dir)
    if times:
        pass
    else:
        return [0]
    return o_list+i_list


def getPunch2(k, instance):
    o_k = 0
    i_k = 0
    times = []
    sizes = []
    feature = []
    
    for line in map(str.strip, instance):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        if dir >0:
            o_k += 1
            if o_k == k:
                feature.append(timestamp)
                o_k=0
        if dir <0:
            i_k += 1
            if i_k == k:
                feature.append(-1*timestamp)
                i_k=0
        times.append(timestamp)
        sizes.append(dir)
    if times:
        pass
    else:
        return [0]
    return feature

def getDualPunch(k, instance):
    o_k = 0
    i_k = 0
    o_k2 = 0
    i_k2 = 0
    times = []
    sizes = []
    o_list = []
    i_list = []
    o_list2 = []
    i_list2 = []
    for line in map(str.strip, instance):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        if dir >0:
            o_k += 1
            if o_k == k:
                o_list.append(timestamp)
                o_k=0
            o_k2 += 1
            if o_k2 == k:
                o_list2.append(timestamp)
                o_k2=0
        if dir <0:
            i_k += 1
            if i_k == k+1:
                i_list.append(-1*timestamp)
                i_k=0
            i_k += 1
            if i_k2 == k+1:
                i_list2.append(-1*timestamp)
                i_k2=0
        times.append(timestamp)
        sizes.append(dir)
    if times:
        pass
    else:
        return [0]
    return o_list+i_list+o_list2+i_list2

def getPunch(k, instance):
    o_k = 0
    i_k = 0
    times = []
    sizes = []
    o_list = []
    i_list = []
    for line in map(str.strip, instance):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        if dir >0:
            o_k += 1
            if o_k == k:
                o_list.append(timestamp)
                o_k=0
        if dir <0:
            i_k += 1
            if i_k == k:
                i_list.append(-1*timestamp)
                i_k=0
        times.append(timestamp)
        sizes.append(dir)
    if times:
        pass
    else:
        return [0]
    return o_list+i_list

def getInterval(instance):
    times = []
    sizes = []
    max_int = 0
    first = True
    for line in map(str.strip, instance):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        if first == True:
            first = False
            times.append(0)
        else:
            if max_int < timestamp-beforetimestamp:
                max_int = timestamp-beforetimestamp
            times.append(timestamp-beforetimestamp)
        sizes.append(dir)
        beforetimestamp = timestamp
    if times:
    # if timestamp:
        pass
    else:
        return [0]
    return np.array(times)


def getIntervalTAM(instance):
    max_matrix_len = 18000
    maximum_load_time = 120
    times = []
    sizes = []
    max_int = 0
    first = True
    for line in map(str.strip, instance):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        if first == True:
            first = False
            times.append(0)
        else:
            if max_int < timestamp-beforetimestamp:
                max_int = timestamp-beforetimestamp
            times.append(timestamp-beforetimestamp)
        sizes.append(dir)
        beforetimestamp = timestamp
    feature = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]
    for i in range(0, len(times)):
        if sizes[i] > 0:
            if times[i] >= maximum_load_time:
                feature[0][-1] += 1
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[0][idx] += 1
        if sizes[i] < 0:
            if times[i] >= maximum_load_time:
                feature[1][-1] += 1
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[1][idx] += 1
    feature1 = feature[0]
    feature2 = feature[1]
    result = feature1 + feature2
    return np.array(result), max_int

'''

def getIntervalTAM(instance):
    max_matrix_len = 400
    maximum_load_time = 72
    times = []
    sizes = []
    max_int = 0
    first = True
    for line in map(str.strip, instance):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        if first == True:
            first = False
            times.append(0)
        else:
            if max_int < 10*(timestamp-beforetimestamp):
                max_int = 10*(timestamp-beforetimestamp)
            times.append(10*(timestamp-beforetimestamp))
        sizes.append(dir)
        beforetimestamp = timestamp
    feature = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]
    for i in range(0, len(times)):
        if sizes[i] > 0:
            if times[i] >= maximum_load_time:
                feature[0][-1] += 1
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[0][idx] += 1
        if sizes[i] < 0:
            if times[i] >= maximum_load_time:
                feature[1][-1] += 1
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[1][idx] += 1
    feature1 = feature[0]
    feature2 = feature[1]
    result = feature1 + feature2
    return np.array(result), max_int
'''

def getPartialTiktok(instance):
    times = []
    first = True
    feature = []
    for i, line in enumerate(map(str.strip, instance)):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        times.append(timestamp * 100)
        dir = float(dir)
        if first == True:
            beforetimestamp = timestamp
            first = False
        if timestamp-beforetimestamp < 0.0001:
            interval = 1
        else:
            interval= timestamp-beforetimestamp
        if dir > 0: feature.append(timestamp/interval)
        if dir < 0: feature.append(-1*timestamp/interval)
        beforetimestamp = timestamp
    if times:
    # if timestamp:
        pass
    else:
        return [0]
    return feature

'''
timestamp 기준으로 slot 나눳음
timestamp 대신에 interval 기준으로

'''
def getTTPartialOrder(instance): # 땡긴다
    outgoing_array = []
    incoming_array = []
    times = []
    first = True
    first_in = True
    first_out = True
    first_i_in = 0
    first_i_out = 0
    for i, line in enumerate(map(str.strip, instance)):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        times.append(timestamp * 100)
        dir = float(dir)
        if first == True:
            beforetimestamp = timestamp
            first = False 
        if (first_in == True) and (dir <0):
            first_i_in = i
            first_in = False
        if (first_out == True) and (dir >0):
            first_i_out = i
            first_out = False
        if timestamp-beforetimestamp < 0.0001:
            interval = 1
        else:
            interval= timestamp-beforetimestamp
        if dir > 0: outgoing_array.append(int(i-first_i_in+1)/interval)
        if dir < 0: incoming_array.append(int(-1*i+first_i_out-1)/interval)
        beforedir = dir
        beforetimestamp = timestamp
    if times:
    # if timestamp:
        pass
    else:
        return [0]
    return outgoing_array+incoming_array

def getTTOrder(instance): # 땡긴다
    outgoing_array = []
    incoming_array = []
    times = []
    iatarr = [0]
    first = True
    first_in = True
    first_out = True
    first_i_in = 0
    first_i_out = 0
    for i, line in enumerate(map(str.strip, instance)):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        if first == True:
            beforedir = dir
            first = False 
        if (first_in == True) and (dir <0):
            first_i_in = i
            first_in = False
        if (first_out == True) and (dir >0):
            first_i_out = i
            first_out = False  
        if dir > 0: outgoing_array.append(int(i-first_i_in+1))
        if dir < 0: incoming_array.append(int(-1*i+first_i_out-1))
        beforedir = dir
        # beforetimestamp = timestamp
    if outgoing_array == []: outgoing_array = [0]
    if incoming_array == []: incoming_array = [0]
    return outgoing_array+[0]+incoming_array

def getSimpleminusOrder(instance):
    outgoing_array = []
    incoming_array = []
    first = True
    for i, line in enumerate(map(str.strip, instance)):
        if first == True:
            timestamp, dir, _ = line.split('\t')
            timestamp = float(timestamp)
            dir = float(dir)
            beforedir = dir
            first = False   
        else: 
            timestamp, dir, _ = line.split('\t')
            timestamp = float(timestamp)
            dir = float(dir)
        if dir > 0: outgoing_array.append(i)
        else: incoming_array.append(-1*i)
        beforedir = dir
        # beforetimestamp = timestamp
    if outgoing_array == []: outgoing_array = [0]
    if incoming_array == []: incoming_array = [0]
    return outgoing_array+incoming_array

def getSimpleOrder(instance):
    outgoing_array = []
    incoming_array = []
    first = True
    for i, line in enumerate(map(str.strip, instance)):
        if first == True:
            timestamp, dir, _ = line.split('\t')
            timestamp = float(timestamp)
            dir = float(dir)
            beforedir = dir
            first = False   
        else: 
            timestamp, dir, _ = line.split('\t')
            timestamp = float(timestamp)
            dir = float(dir)
        if dir > 0: outgoing_array.append(i)
        else: incoming_array.append(i)
        beforedir = dir
        # beforetimestamp = timestamp
    if outgoing_array == []: outgoing_array = [0]
    if incoming_array == []: incoming_array = [0]
    return outgoing_array+incoming_array

def getSummer(instance):
    max_matrix_len = 2100 # tam feature matrix len
    maximum_load_time = 80 # 최대 80초
    times = []
    sizes = []
    
    for line in map(str.strip, instance): 
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        times.append(timestamp)
        sizes.append(timestamp*dir)
    feature = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            if times[i] >= maximum_load_time:
                feature[0][-1] += 1
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[0][idx] += 1
        if sizes[i] < 0:
            if times[i] >= maximum_load_time:
                feature[1][-1] += 1
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[1][idx] += 1
    feature1 = feature[0]
    feature2 = feature[1]
    result = feature1 + feature2


def getChange(instance, overlap):
    max_matrix_len = 1800 # tam feature matrix len
    maximum_load_time = 80 # 최대 80초
    OVERLAP = float(overlap)/2 # 한쪽 OVERLAP
    times = []
    sizes = []
    for line in map(str.strip, instance):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        times.append(timestamp)
        sizes.append(dir)
    feature = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            if times[i] - OVERLAP >= maximum_load_time:
                feature[0][-1] += 2
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[0][idx] += 1
                idx = int((times[i]+OVERLAP) * (max_matrix_len - 1) / maximum_load_time)
                feature[0][idx] += 1
                idx = int((times[i]-OVERLAP) * (max_matrix_len - 1) / maximum_load_time)
                feature[0][idx] += 1
        if sizes[i] < 0:
            if times[i] - OVERLAP >= maximum_load_time:
                feature[1][-1] += 2
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[1][idx] += 1
                idx = int((times[i]+OVERLAP) * (max_matrix_len - 1) / maximum_load_time)
                feature[1][idx] += 1
                idx = int((times[i]-OVERLAP) * (max_matrix_len - 1) / maximum_load_time)
                feature[1][idx] += 1
    feature1 = feature[0]
    feature2 = feature[1]
    result = feature1 + feature2
    return np.array(result)



def getArrive(instance, overlap):
    OVERLAP = float(overlap)/2 # 한쪽 OVERLAP
    times = []
    sizes = []
    for line in map(str.strip, instance):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        times.append(timestamp)
        sizes.append(dir)
    if times:
        pass
    else:
        return [0]
    maximum_load_time = int(times[-1])
    max_matrix_len = int(math.trunc(float(len(sizes)) * 0.01)*100)+1
    # print(max_matrix_len)
    feature = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]
    for i in range(0, len(sizes)):
        # print(int(times[i] * (max_matrix_len - 1) / maximum_load_time))
        if sizes[i] > 0:
            if times[i] + OVERLAP >= maximum_load_time:
                feature[0][-1] += 2
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                # print(idx, max_matrix_len)
                feature[0][idx] += 1
                idx = int((times[i]+OVERLAP) * (max_matrix_len - 1) / maximum_load_time)
                feature[0][idx] += 1
                idx = int((times[i]-OVERLAP) * (max_matrix_len - 1) / maximum_load_time)
                feature[0][idx] += 1
        if sizes[i] < 0:

            if times[i] + OVERLAP >= maximum_load_time:
                feature[1][-1] += -2
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[1][idx] += -1
                idx = int((times[i]+OVERLAP) * (max_matrix_len - 1) / maximum_load_time)
                feature[1][idx] += -1
                idx = int((times[i]-OVERLAP) * (max_matrix_len - 1) / maximum_load_time)
                feature[1][idx] += -1
    feature1 = feature[0]
    feature2 = feature[1]
    result = feature1 + feature2
    # result.append(len(sizes))
    # result = list(map(int, result))
    return result

def getTAMTiktok(instance):
    max_matrix_len = 1800
    maximum_load_time = 80
    times = []
    sizes = []
    result = []
    for line in map(str.strip, instance):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        times.append(timestamp)
        sizes.append(dir)
    feature = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            if times[i] >= maximum_load_time:
                feature[0][-1] += times[i] * sizes[i]
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[0][idx] += times[i] * sizes[i]
        if sizes[i] < 0:
            if times[i] >= maximum_load_time:
                feature[1][-1] += times[i] * sizes[i]
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[1][idx] += times[i] * sizes[i]
    for i in range(len(feature)):
        result.append(feature[0][i])
        result.append(feature[1][i])
    return result

def getNEWoverlapTAM(instance, alpha=0.01, interval=0.05):
    times = []
    sizes = []
    for line in map(str.strip, instance):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        times.append(timestamp)
        sizes.append(dir)
    feature = []
    if times:
        start = times[0]
    else:
        return [0]
    end = start + interval
    
    while start < times[-1]:
        num_p = 0
        num_n = 0
        for j in range(len(times)):
            if times[j] >= start and times[j] <= end:
                if sizes[j] > 0:
                    num_p += 1
                if sizes[j] < 0:
                    num_n +=1
        feature.append(num_p)
        feature.append(num_n)
        start = end-alpha
        end = start + interval
    return feature

def getTAMOverlap(instance, overlap):
    max_matrix_len = 1800 # tam feature matrix len
    maximum_load_time = 80 # 최대 80초
    OVERLAP = float(overlap)/2 # 한쪽 OVERLAP
    times = []
    sizes = []
    for line in map(str.strip, instance):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        times.append(timestamp)
        sizes.append(dir)
    feature = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            if times[i] - OVERLAP >= maximum_load_time:
                feature[0][-1] += 2
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[0][idx] += 1
                idx = int((times[i]+OVERLAP) * (max_matrix_len - 1) / maximum_load_time)
                feature[0][idx] += 1
                idx = int((times[i]-OVERLAP) * (max_matrix_len - 1) / maximum_load_time)
                feature[0][idx] += 1
        if sizes[i] < 0:
            if times[i] - OVERLAP >= maximum_load_time:
                feature[1][-1] += 2
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[1][idx] += 1
                idx = int((times[i]+OVERLAP) * (max_matrix_len - 1) / maximum_load_time)
                feature[1][idx] += 1
                idx = int((times[i]-OVERLAP) * (max_matrix_len - 1) / maximum_load_time)
                feature[1][idx] += 1
    feature1 = feature[0]
    feature2 = feature[1]
    result = feature1 + feature2
    return np.array(result)

def getKVar(instance, k=11):
    times = []
    tiktok = []
    k = int((k-1)/2)

    for line in map(str.strip, instance):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        times.append(timestamp)
        tiktok.append(timestamp*dir)
    vars = [[0 for _ in range(len(tiktok))], 
            [0 for _ in range(len(times))]]
    
    # 양 옆 k개씩까지의 분산 구하기
    if times:
        pass
    else:
        return [0]
    
    for i in range(len(times)):
        if i<=k:
            vars[0][i]=np.var(times[:i+k])
            vars[1][i]=sum(tiktok[:i+k])
        elif i>= len(times)+k-1:
            vars[0][i]=np.var(times[i-k:])
            vars[1][i]=sum(tiktok[:i+k])
        else:
            vars[0][i]=np.var(times[i-k:i+k])
            vars[1][i]=sum(tiktok[:i+k])
        # vars[1][i]=i
    vars.sort(key=lambda x:x[0])
    return vars[1]


def getDualSlot(instance):
    max_matrix_len = 1800 # tam feature matrix len
    maximum_load_time = 80 # 최대 80초
    times = []
    sizes = []
    for line in map(str.strip, instance):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        times.append(timestamp)
        sizes.append(dir)
    feature44 = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]
    feature88 = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]

    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            if times[i] >= maximum_load_time:
                feature44[0][-1] += 1
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature44[0][idx] += 1
        if sizes[i] < 0:
            if times[i] >= maximum_load_time:
                feature44[1][-1] += 1
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature44[1][idx] += 1
    feature1 = feature44[0]
    feature2 = feature44[1]
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            if times[i] >= maximum_load_time:
                feature88[0][-1] += 1
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature88[0][idx] += 1
        if sizes[i] < 0:
            if times[i] >= maximum_load_time:
                feature88[1][-1] += 1
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time /2)
                feature88[1][idx] += 1
                
    feature3 = feature88[0]
    feature4 = feature88[1]
    result = feature1 + feature2 + feature3 + feature4
    return np.array(result)

def getDualSlotOverlap(instance, overlap):
    max_matrix_len = 1800 # tam feature matrix len
    maximum_load_time = 80 # 최대 80초
    OVERLAP = float(overlap)/2 # 한쪽 OVERLAP
    times = []
    sizes = []
    for line in map(str.strip, instance):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        times.append(timestamp)
        sizes.append(dir)
    feature44 = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]
    feature88 = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            if times[i] - OVERLAP >= maximum_load_time:
                feature44[0][-1] += 2
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature44[0][idx] += 1
                idx = int((times[i]+OVERLAP) * (max_matrix_len - 1) / maximum_load_time)
                feature44[0][idx] += 1
                idx = int((times[i]-OVERLAP) * (max_matrix_len - 1) / maximum_load_time)
                feature44[0][idx] += 1
        if sizes[i] < 0:
            if times[i] - OVERLAP >= maximum_load_time:
                feature44[1][-1] += 1
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature44[1][idx] += 1
                idx = int((times[i]+OVERLAP) * (max_matrix_len - 1) / maximum_load_time)
                feature44[1][idx] += 1
                idx = int((times[i]-OVERLAP) * (max_matrix_len - 1) / maximum_load_time)
                feature44[1][idx] += 1
    feature1 = feature44[0]
    feature2 = feature44[1]
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            if times[i] - OVERLAP >= maximum_load_time:
                feature88[0][-1] += 2
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature88[0][idx] += 1
                idx = int((times[i]+OVERLAP) * (max_matrix_len - 1) / maximum_load_time)
                feature88[0][idx] += 1
                idx = int((times[i]-OVERLAP) * (max_matrix_len - 1) / maximum_load_time)
                feature88[0][idx] += 1
        if sizes[i] < 0:
            if times[i] - OVERLAP >= maximum_load_time:
                feature88[1][-1] += 2
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time /2)
                feature88[1][idx] += 1
                idx = int((times[i]+OVERLAP) * (max_matrix_len - 1) / maximum_load_time/2)
                feature88[1][idx] += 1
                idx = int((times[i]-OVERLAP) * (max_matrix_len - 1) / maximum_load_time/2)
                feature88[1][idx] += 1
    feature3 = feature88[0]
    feature4 = feature88[1]
    result = feature1 + feature2 + feature3 + feature4
    return np.array(result)

def getTAM(instance):
    max_matrix_len = 1800
    maximum_load_time = 80
    times = []
    sizes = []
    for line in map(str.strip, instance):
        timestamp, dir, _ = line.split('\t')
        timestamp = float(timestamp)
        dir = float(dir)
        times.append(timestamp)
        sizes.append(dir)
    feature = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            if times[i] >= maximum_load_time:
                feature[0][-1] += 1
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[0][idx] += 1
        if sizes[i] < 0:
            if times[i] >= maximum_load_time:
                feature[1][-1] += 1
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[1][idx] += 1
    feature1 = feature[0]
    feature2 = feature[1]
    result = feature1 + feature2
    return np.array(result)



def getTimeBurst(instance):
    burst = 0
    timestamp = 0
    burst_array = []
    first = True
    for line in map(str.strip, instance):
        if first == True:
            timestamp, dir, _ = line.split('\t')
            timestamp = float(timestamp)
            dir = float(dir)
            beforedir = dir
            first = False   
        else: 
            timestamp, dir, _ = line.split('\t')
            timestamp = float(timestamp)
            dir = float(dir)
        
        # print(line)
        # print(burst)
        if beforedir * dir >= 0:
            burst += timestamp
        
        else:
            burst_array.append(burst)
            burst = 0
        beforedir = dir
        beforetimestamp = timestamp
    burst_array.append(burst)
    # print(burst_array)
    return np.array(burst_array)

def getTiktokBurst(instance):
    burst = 0
    timestamp = 0
    burst_array = []
    first = True
    for line in map(str.strip, instance):
        if first == True:
            timestamp, dir, _ = line.split('\t')
            timestamp = float(timestamp)
            dir = float(dir)
            beforedir = dir
            first = False   
        else: 
            timestamp, dir, _ = line.split('\t')
            timestamp = float(timestamp)
            dir = float(dir)
        
        # print(line)
        # print(burst)
        if beforedir * dir >= 0:
            burst += timestamp * dir
        
        else:
            burst_array.append(burst)
            burst = 0
        beforedir = dir
        beforetimestamp = timestamp
    burst_array.append(burst)
    # print(burst_array)
    return np.array(burst_array)


def getDirectionCumul(instance):
    dir_cumul = 0
    dir_cumul_array=[]
    for line in map(str.strip, instance):
        dir=float(line.split('\t')[1])
        dir_cumul += int(dir/abs(dir))
        dir_cumul_array.append(dir_cumul)
    return dir_cumul_array

def getIntervalBurst(instance):
    burst = 0
    beforetimestamp = 0
    timestamp = 0
    burst_array = []
    first = True
    for line in map(str.strip, instance):
        if first == True:
            timestamp, dir, _ = line.split('\t')
            timestamp = float(timestamp)
            dir = float(dir)
            beforedir = dir
            beforetimestamp = timestamp
            first = False   
        else: 
            timestamp, dir, _ = line.split('\t')
            timestamp = float(timestamp)
            dir = float(dir)
        
        # print(line)
        # print(burst)
        if beforedir * dir >= 0:
            burst += 1
        
        else:
            burst_array.append(timestamp-beforetimestamp)
            burst_array.append(burst)
            # print("me")
            burst = 1
        beforedir = dir
        beforetimestamp = timestamp
    burst_array.append(timestamp-beforetimestamp)
    burst_array.append(burst)
    # print(burst_array)
    return np.array(burst_array)

def TwoTimeStampBurst(instance):
    
    beforetimestamp = 0
    timestamp = 0
    burst_array = []
    first = True
    for line in map(str.strip, instance):
        
        if first == True:
            timestamp, dir, _ = line.split('\t')
            timestamp = float(timestamp)
            dir = float(dir)
            beforedir = dir
            beforetimestamp = timestamp
            first = False   
        else: 
            timestamp, dir, _ = line.split('\t')
            timestamp = float(timestamp)
            dir = float(dir)
        if beforedir * dir < 0:
            burst_array.append(beforetimestamp)
            burst_array.append(timestamp)
            # print("me")
            
        beforedir = dir
        beforetimestamp = timestamp
    burst_array.append(beforetimestamp)
    burst_array.append(timestamp)
    # print(burst_array)
    return np.array(burst_array)

def getcategorical(instance):
    burst = 0
    timestamp = 0
    direction_array = []
    timestamp_array = []
    feature_array = []
    outgoing_array = []
    first = True
    for i, line in enumerate(map(str.strip, instance)):
        if first == True:
            timestamp, dir, _ = line.split('\t')
            timestamp = float(timestamp)
            dir = float(dir)
            beforedir = dir
            first = False   
        else: 
            timestamp, dir, _ = line.split('\t')
            timestamp = float(timestamp)
            dir = float(dir)
        if beforedir * dir < 0:
            burst += 1
        if dir > 0: outgoing_array.append(i)
        direction_array.append(dir)
        timestamp_array.append(timestamp)
        beforedir = dir
    if direction_array == []: return [0, 0, 0,0, 0, 0,0, 0, 0,0]
    if outgoing_array == []: outgoing_array = [0]
    feature_array.append(timestamp) # 가장 마지막 timestamp
    feature_array.append(direction_array.count(-1)) # incoming
    feature_array.append(direction_array.count(1)) # outgoing
    feature_array.append(len(direction_array))
    feature_array.append(direction_array.count(-1)/len(direction_array))
    feature_array.append(direction_array.count(1)/len(direction_array))
    feature_array.append(direction_array.count(1)+direction_array.count(-1)+len(direction_array))
    feature_array.append(np.mean(outgoing_array))
    feature_array.append(np.std(outgoing_array))
    feature_array.append(burst) # 부호 바뀐 횟수

    return np.array(feature_array)

def getBurst(instance):
    burst = 0
    burst_array = []
    first = True
    for line in map(str.strip, instance):
       
        if first == True:
            dir=float(line.split('\t')[1])
            dir = int(dir/abs(dir))
            before = dir
            first = False   
        else: 
            dir=float(line.split('\t')[1])
            dir = int(dir/abs(dir))
        
        # print(line)
        # print(burst)
        if before * dir >= 0:
            burst += 1
        
        else:
            burst_array.append(burst)
            # print("me")
            burst = 1
        before = dir
    burst_array.append(burst)
    # print(burst_array)
    return np.array(burst_array)

def extract_feature(n, input_path, mon_or_unmon, feature, overlap=None, k=11):
    INPUT_SIZE = 5000
    max_interval = 0
    files = natsorted(glob.glob(input_path+'/*.cell'))
    direction_cumul_all = []
    y_all = []
    instance_data = []
    path_data = []
    max_len = 0
    print("Extracting Feature...")
    for file in tqdm(files):
        if ("join" in file): continue
        if os.path.isdir(file): continue
        # mon: 10(site)-162(instance)_split(type)_6(path).cell
        # unmon: 10481_split_1.cell
        with open(file, 'r') as f:
            instance = f.readlines()
        if feature == 'cumul':
            feature_data = getDirectionCumul(instance)
        elif feature == 'burst':
            feature_data = getBurst(instance)
        elif feature == 'interval_burst':
            feature_data = getIntervalBurst(instance)
        elif feature == 'twotime_burst':
            feature_data = TwoTimeStampBurst(instance)
        elif feature == 'categorical':
            feature_data = getcategorical(instance)
        elif feature == 'tiktokburst':
            feature_data = getTiktokBurst(instance)
            INPUT_SIZE = 1000
        elif feature == 'timeburst':
            feature_data = getTimeBurst(instance)
            INPUT_SIZE = 1000
        elif feature == 'tam':
            feature_data = getTAM(instance)
            INPUT_SIZE = 3600
        elif feature == 'overlaptam':
            feature_data = getTAMOverlap(instance, overlap)
            INPUT_SIZE = 3600
        elif feature == 'dualslot':
            feature_data = getDualSlot(instance)
            INPUT_SIZE = 5400
        elif feature == 'tamtiktok':
            feature_data = getTAMTiktok(instance)
        elif feature == 'newover':
            feature_data = getNEWoverlapTAM(instance)
        elif feature == 'kvar':
            feature_data = getKVar(instance, k)
        elif feature == 'arrive':
            feature_data = getArrive(instance, overlap)
        elif feature == 'change':
            feature_data = getChange(instance, overlap)
        elif feature == 'simpleorder':
            feature_data = getSimpleOrder(instance)
        elif feature == 'simpleminusorder':
            feature_data = getSimpleminusOrder(instance)
        elif feature == 'ttorder':
            feature_data = getTTOrder(instance)
        elif feature == 'ttpartialorder':
            feature_data = getTTPartialOrder(instance)
        elif feature == 'partialtiktok':
            feature_data = getPartialTiktok(instance)
        elif feature == 'intervaltam':
            feature_data, max_int = getIntervalTAM(instance)
        elif feature == 'interval':
            feature_data = getInterval(instance)
        elif feature == 'punch':
            feature_data = getPunch(k, instance)
        elif feature == 'dualpunch':
            feature_data = getDualPunch(k, instance)
        elif feature == 'punch2':
            feature_data = getPunch2(k, instance)
        elif feature == 'punch3':
            feature_data = getPunch3(k, instance)
        elif feature == 'punch4':
            feature_data = getPunch4(k, instance)
        elif feature == 'punch5':
            feature_data = getPunch5(k, instance)
        # if max_int>max_interval:
        #    max_interval = max_int

        if len(feature_data)>max_len:
            max_len = len(feature_data)
        # if feature_data == []: continue
        if mon_or_unmon=='mon':
            site, instance=file.split('/')[-1].split('-')
            site=int(site)
        else: 
            site = -1 # site
            instance = file.split('/')[-1]
        instance, _, path=instance.split('_')  # type: split / join
        instance=int(instance)
        path=int(path.split('.')[0])
        
        if path==0:
            y_all.append(site)
        if(instance!=0 and path==0): # path_data reset
            instance_data.append(np.array(path_data, dtype=object))
            path_data=[] 
        if(site!=0 and instance==0 and path==0): # instance_data reset
            instance_data.append(np.array(path_data, dtype=object))
            print(file, instance_data)
            direction_cumul_all.append(np.array(instance_data, dtype=object))
            # print(file, len(instance_data))
            instance_data=[]
            path_data=[]
        # y_all.append(site)
        if feature != 'categorical':
            if len(feature_data) < INPUT_SIZE:
                feature_data  = np.concatenate((feature_data , np.zeros((INPUT_SIZE - len(feature_data)))))
            feature_data  = feature_data[:INPUT_SIZE]
        path_data.append(feature_data)
    if(path_data): instance_data.append(np.array(path_data, dtype=object))
    if(instance_data): direction_cumul_all.append(np.array(instance_data, dtype=object))
    direction_cumul_all = np.array(direction_cumul_all, dtype=object)
    
    print("********")
    # print(f"{max_interval=}")
    print(f"{max_len=}")
    X=[]
    for i in range(len(direction_cumul_all)):
            print(i, len(direction_cumul_all[i]), end='\t')
            if direction_cumul_all[i].any():
                for j in range(len(direction_cumul_all[i])):
                    X_path=[]
                    for sample in direction_cumul_all[i][j]:
                        X_path.append(sample)
                    X.append(np.array(X_path))
    direction_cumul_all = np.array(X)
    y_all = np.array(y_all, dtype=object)
    print("\nX shape:", direction_cumul_all.shape)
    print("y shape:", y_all.shape)
    print("Feature Extraction Done\n")
    
    
    return direction_cumul_all, y_all
    


def multiply(n, X_train, X_test, X_valid, y_train, y_test, y_valid):  
    X_train_xn=[]
    X_valid_xn=[]
    X_test_xn=[]
    y_train_xn = []
    y_test_xn = []
    y_valid_xn = []

    for X in X_train:
        for p in X: X_train_xn.append(p)
    for X in X_valid:
        for p in X: X_valid_xn.append(p)
    for X in X_test:
        for p in X: X_test_xn.append(p)
    for y in y_train:
        for _ in range(n): y_train_xn.append(y)
    for y in y_valid:
        for _ in range(n): y_valid_xn.append(y)
    for y in y_test:
        for _ in range(n): y_test_xn.append(y)
    X_train_xn=np.array(X_train_xn)
    X_valid_xn=np.array(X_valid_xn)
    X_test_xn=np.array(X_test_xn)
    y_train_xn=np.array(y_train_xn)
    y_valid_xn=np.array(y_valid_xn)
    y_test_xn=np.array(y_test_xn)

    print("train: ", np.array(X_train_xn).shape, end='\t')
    print(np.array(y_train_xn).shape)
    print("valid: ", np.array(X_valid_xn).shape, end='\t')
    print(np.array(y_valid_xn).shape)
    print("test: ", np.array(X_test_xn).shape, end='\t')
    print(np.array(y_test_xn).shape)

    print("multiply done")
    return X_train_xn, X_test_xn, X_valid_xn, y_train_xn, y_test_xn, y_valid_xn

if __name__=="__main__":
    # Prompt
    # 3 mon c (BigEnough path3 mon cumul)
    # 5 unmon b (SingleSite path5 unmon burst)
    args = parser.parse_args()
    path_num = args.path_num[0]
    mode = args.mode[0]
    feature = args.feature[0]
    # overlap = args.overlap[0]
    overlap = args.overlap
    alpha = args.alpha
    interval = args.interval
    k = args.k

    # Input
    input_path = "/scratch/TrafficSliver/BigEnough/path"+path_num+"/"+mode+"/ts/"
    # input_path = "/data/ts-final/BigEnough/path" + str(path_num) + '/' + str(mode) + "/ts/"
    if feature == 'newover':
        output_path = "/scratch/TrafficSliver/pre-BE/" + str(feature) + "/path" + str(path_num) + '/s' + str(interval) + '_a'+str(alpha)+'/'
        # output_path = '/scratch/TrafficSliver/pre-BE/overlaptam/path2/yes_padding_0.01/'
    elif feature == 'overlaptam':
        output_path = "/scratch/TrafficSliver/pre-BE/" + str(feature) + "/path" + str(path_num) + '/' + str(overlap) + '/'
        # output_path = '/scratch/TrafficSliver/pre-BE/overlaptam/path2/yes_padding_0.01/'
    elif feature == 'arrive':
        output_path = "/scratch/TrafficSliver/pre-BE/" + str(feature) + "/path" + str(path_num) + '/' + str(overlap) + '/'
    elif feature == 'kvar' or feature=='punch' or feature=='dualpunch' or feature=='punch2'or feature=='punch3' or feature=='punch4' or 'punch' in feature:
        output_path = "/scratch/TrafficSliver/pre-BE/" + str(feature) + "/path" + str(path_num) + '/' + str(k) + '/'
    else:
        output_path = "/scratch/TrafficSliver/pre-BE/" + str(feature) + "/path" + str(path_num) + '/'

    # Feature Extraction
    t_beg = time.time()
    X, y = extract_feature(path_num, input_path, mode, feature, overlap, k)
    
    # train, test, valid split(8:1:1)
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)
    X_valid, X_test, y_valid, y_test=train_test_split(X_test, y_test, test_size=0.5)
    print("train: ", np.array(X_train).shape, end='\t')
    print(np.array(y_train).shape)
    print("valid: ", np.array(X_valid).shape, end='\t')
    print(np.array(y_valid).shape)
    print("test: ", np.array(X_test).shape, end='\t')
    print(np.array(y_test).shape)
    print("ttv split done\n")
    

    with open(output_path + mode + '_X_train.pkl', 'wb') as f:
        pickle.dump(np.array(X_train), f, pickle.HIGHEST_PROTOCOL)
    with open(output_path + mode + '_X_test.pkl', 'wb') as f:
        pickle.dump(np.array(X_test), f, pickle.HIGHEST_PROTOCOL)
    with open(output_path + mode + '_X_valid.pkl', 'wb') as f:
        pickle.dump(np.array(X_valid), f, pickle.HIGHEST_PROTOCOL)
    
    with open(output_path + mode + '_y_train.pkl', 'wb') as f:
        pickle.dump(np.array(y_train), f, pickle.HIGHEST_PROTOCOL)
    with open(output_path + mode + '_y_test.pkl', 'wb') as f:
        pickle.dump(np.array(y_test), f, pickle.HIGHEST_PROTOCOL)
    with open(output_path + mode + '_y_valid.pkl', 'wb') as f:
        pickle.dump(np.array(y_valid), f, pickle.HIGHEST_PROTOCOL)

    # multiply
    X_train, X_test, X_valid, y_train, y_test, y_valid = multiply(int(path_num), X_train, X_test, X_valid, y_train, y_test, y_valid)

    # Output
    
    with open(output_path + mode + '_X_train_x'+path_num+'.pkl', 'wb') as f:
        pickle.dump(np.array(X_train), f, pickle.HIGHEST_PROTOCOL)
    with open(output_path + mode + '_X_test_x'+path_num+'.pkl', 'wb') as f:
        pickle.dump(np.array(X_test), f, pickle.HIGHEST_PROTOCOL)
    with open(output_path + mode + '_X_valid_x'+path_num+'.pkl', 'wb') as f:
        pickle.dump(np.array(X_valid), f, pickle.HIGHEST_PROTOCOL)
    with open(output_path + mode + '_y_train_x'+path_num+'.pkl', 'wb') as f:
        pickle.dump(np.array(y_train), f, pickle.HIGHEST_PROTOCOL)
    with open(output_path + mode + '_y_test_x'+path_num+'.pkl', 'wb') as f:
        pickle.dump(np.array(y_test), f, pickle.HIGHEST_PROTOCOL)
    with open(output_path + mode + '_y_valid_x'+path_num+'.pkl', 'wb') as f:
        pickle.dump(np.array(y_valid), f, pickle.HIGHEST_PROTOCOL)

    sys.stdout.write("\nAll Done in "+str(time.time()-t_beg)+" seconds\n")
    print(f"{path_num=}, {mode=}, {feature=}, {overlap=}")
    if feature=="newover":
        print(f"{alpha=}, {interval=}")

    with open(output_path + mode + '_X_test_x'+path_num+'.pkl', 'rb') as f:
        X_test=pickle.load(f)
    print(f"{len(X_test)=}")
    for i in range(7):
        print(X_test[i])

    with open(output_path + mode + '_y_test_x'+path_num+'.pkl', 'rb') as f:
        y_test=pickle.load(f)
    print(f"{len(y_test)=}")
    for i in range(7):
        print(y_test[i])

    

            


    
