from statistics import mean
from natsort import natsorted
import glob
import argparse
import matplotlib.pyplot as plt
import pandas as pd
MAX_COUNT = 100
parser = argparse.ArgumentParser()
parser.add_argument("-p", '--path', nargs='+',
                    help=' fiPath of folder with instancesles (wang_format)', default='dataset/')


def count_packet_size(traces):
    traces_file = natsorted(glob.glob(traces[0]+'/*'))
    counts_dict = {}
    counts_list = []
    for instance_file in traces_file:
        instance = open(instance_file, 'r')
        instance = instance.read().split('\n')[:-1]
        inst_name, inst_num = instance_file.split('-')
        inst_name = int(inst_name.split('\\')[1])
        inst_num = int(inst_num)
        if(inst_num == 0):
            print(inst_name)
            if(inst_name != 0):
                counts_list.append(counts_dict)
            counts_dict = {}
            if(inst_name == 5):
                break
        cnt = len(instance)
        if(cnt not in counts_dict):
            counts_dict[cnt] = 0
        counts_dict[cnt] += 1
    # last one
    if(len(counts_dict)):
        counts_list.append(counts_dict)
    for inst_num in range(len(counts_list)):
        df = pd.Series(counts_list[inst_num])
        df = pd.DataFrame(df, columns=['counts'])
        x = df.index.to_list()
        y = df['counts']
        plt.figure(figsize=(10, 6))
        plt.xlabel('packet')
        plt.ylabel('number of packets')
        plt.bar(x, y)
        plt.savefig("./images/traffic counts-"+str(inst_num)+".png")


def count_mean(traces):
    traces_file = natsorted(glob.glob(traces[0]+'/*'))
    mean_list = []
    packet_list = []
    packet_dict = {}
    for instance_file in traces_file:
        instance = open(instance_file, 'r')
        instance = instance.read().split('\n')[:-1]
        inst_name, inst_num = instance_file.split('-')
        inst_name = int(inst_name.split('\\')[1])
        inst_num = int(inst_num)
        if(inst_num == MAX_COUNT):
            mean_list.append(mean(packet_list))
            print(inst_name, end='\t')
            print(mean(packet_list), min(packet_list), max(packet_list))
            packet_dict[inst_name] = {'mean': mean(packet_list),
                                      'min': min(packet_list),
                                      'max': max(packet_list)}
            packet_list = []
        cnt = len(instance)
        packet_list.append(cnt)
    df = pd.DataFrame(packet_dict).T
    df.to_csv('./csv/packet size.csv')

    x = df.index.to_list()
    y_mean = df['mean']
    y_min = df['min']
    y_max = df['max']
    plt.figure(figsize=(10, 6))
    plt.xlabel('packet')
    plt.ylabel('num of instance')
    plt.plot(x, y_mean, label='mean')
    plt.plot(x, y_min, label='min')
    plt.plot(x, y_max, label='max')
    plt.legend(loc='best')
    plt.savefig("./images/mean.png")


if __name__ == '__main__':
    args = parser.parse_args()
    traces = args.path
    count_mean(traces)
    count_packet_size(traces)
