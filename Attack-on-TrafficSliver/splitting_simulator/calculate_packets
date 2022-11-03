from statistics import mean
from struct import pack
from natsort import natsorted
import glob
import argparse
import matplotlib.pyplot as plt
import pandas as pd
MAX_COUNT=100
parser = argparse.ArgumentParser()
parser.add_argument("-p", '--path', nargs='+',
                    help=' fiPath of folder with instancesles (wang_format)')


def count_traffic(traces):
    traces_file = natsorted(glob.glob(traces[0]+'/*'))
    # counts_dict={}
    mean_list = []
    packet_list = []
    # sum = 0
    # cur_cnt = 0
    for instance_file in traces_file:
        instance = open(instance_file, 'r')
        instance = instance.read().split('\n')[:-1]
        inst_name, inst_num = instance_file.split('-')
        inst_name = int(inst_name.split('\\')[1])
        inst_num=int(inst_num)
        if(inst_num == MAX_COUNT):
            if(inst_name != 0):
                print(inst_name, end='\t')
                # df=pd.Series(counts_dict)
                # df=pd.DataFrame(df, columns=['counts'])
                # x=df.index.to_list()
                # y=df['counts']
                # plt.figure(figsize=(10,6))
                # plt.xlabel('packet')
                # plt.ylabel('num of instance')
                # plt.bar(x, y)
                # plt.savefig("./images/traffic counts-"+inst_name+".png")
                mean_list.append(mean(packet_list))
                print(mean(packet_list), min(packet_list), max(packet_list))
            # counts_dict={}
            # sum = 0
            packet_list = []
            # cur_cnt = 0
        cnt = len(instance)
        packet_list.append(cnt)
        # if(cnt not in counts_dict): counts_dict[cnt]=0
        # counts_dict[cnt]+=1
        # sum += cnt
    df_mean=pd.Series(mean_list)
    df_mean=pd.DataFrame(df_mean, columns=['counts'])
    x=df_mean.index.to_list()
    y=df_mean['counts']
    plt.figure(figsize=(10,6))
    plt.xlabel('packet')
    plt.ylabel('num of instance')
    plt.plot(x, y)
    plt.savefig("./images/mean.png")


if __name__ == '__main__':
    args = parser.parse_args()
    traces = args.path
    count_traffic(traces)
