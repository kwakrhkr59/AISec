from statistics import mean
from natsort import natsorted
import glob
import argparse
# import matplotlib.pyplot as plt
import pandas as pd
MAX_COUNT = 100
parser = argparse.ArgumentParser()
parser.add_argument("-p", '--path', nargs='+', help=' fiPath of folder with instancesles (wang_format)', default='/data/TrafficSliver/BigEnough/splitted/mon/mode1/ts')

# def count_packet_thresh(output, traces, thresh):
#     traces_file = natsorted(glob.glob(traces[0]+'/*'))
#     for instance_file in traces_file:
#         instance = open(instance_file, 'r')
#         instance = instance.read().split('\n')[:-1]
#         if (len(instance) == 0):



def count_packet_size2(traces, thresh):
    traces_file = natsorted(glob.glob(traces[0]+'/*'))
    counts_dict = {}
    counts_list = []
    for instance_file in traces_file:
        instance = open(instance_file, 'r')
        instance = instance.read().split('\n')[:-1]
        circit, site, _, path = instance_file.split('-')
        circit = int(circit.split('/')[-1])
        site = int(site)
        path = int(path.split('.')[0])
        if(site == 0):
            print(circit)
            if(circit != 0):
                counts_list.append(counts_dict)
            counts_dict = {}
        cnt = len(instance)
        if(cnt not in counts_dict):
            counts_dict[cnt] = 0
        counts_dict[cnt] += 1
    # last one
    if(len(counts_dict)):
        counts_list.append(counts_dict)
    for site in range(len(counts_list)):
        df = pd.Series(counts_list[site])
        df = pd.DataFrame(df, columns=['counts'])
        x = df.index.to_list()
        y = df['counts']
        plt.figure(figsize=(10, 6))
        plt.xlabel('packet')
        plt.ylabel('number of packets')
        plt.bar(x, y)
        plt.savefig("./images/traffic counts-"+str(site)+".png")

def count_packet_size(traces):
    traces_file = natsorted(glob.glob(traces[0]+'/*'))
    counts_dict = {}
    counts_list = []
    for instance_file in traces_file:
        instance = open(instance_file, 'r')
        instance = instance.read().split('\n')[:-1]
        inst_name, inst_num = instance_file.split('-')
        inst_name = int(inst_name.split('/')[-1])
        inst_num = int(inst_num.split('_')[0])
        if(inst_num == 0):
            print(inst_name)
            if(inst_name != 0):
                counts_list.append(counts_dict)
            counts_dict = {}
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
        # plt.figure(figsize=(10, 6))
        # plt.xlabel('packet')
        # plt.ylabel('number of packets')
        # plt.bar(x, y)
        # plt.savefig("./images/traffic counts-"+str(inst_num)+".png")

def count_timestamp(traces):
    traces_file = natsorted(glob.glob(traces+'/*'))
    packet_list = []
    packet_dict = {}
    for instance_file in traces_file:
        if('join' in instance_file): continue
        if('tail' in instance_file): continue
        if('csv' in instance_file): continue
        instance_name=instance_file.split('/')[-1].split('.')[0]
        instance = open(instance_file, 'r')
        instance = instance.read().split('\n')[:-1]
        path_num=int(instance_name.split('_')[-1])
        instance_num=-1
        if('-' in instance_name): site_num, instance_num=map(int, instance_name.split('_')[0].split('-'))
        else: site_num=int(instance_name.split('_')[0])
        if(site_num!=0 and path_num==0):
            print(site_num-1, end='\t')
            print(mean(packet_list), min(packet_list), max(packet_list))
            if(instance_num!=-1):
                packet_dict[str(site_num-1)+'-'+str(instance_num-1)] = {'mean': mean(packet_list),
                                            'min': min(packet_list),
                                            'max': max(packet_list)}

            else: 
                packet_dict[site_num-1] = {'mean': mean(packet_list),
                                            'min': min(packet_list),
                                            'max': max(packet_list)}
            packet_list = []
        last_timestamp = 0 if len(instance) == 0 else float(instance[-1].split('\t')[0])
        packet_list.append(last_timestamp)
    if(instance_num!=-1):
        packet_dict[str(site_num)+'-'+str(instance_num)] = {'mean': mean(packet_list),
                                    'min': min(packet_list),
                                    'max': max(packet_list)}

    else: 
        packet_dict[site_num] = {'mean': mean(packet_list),
                                    'min': min(packet_list),
                                    'max': max(packet_list)}
    packet_dict[site_num] = {'mean': mean(packet_list),
                                'min': min(packet_list),
                                'max': max(packet_list)}
    df = pd.DataFrame(packet_dict).T
    df.to_csv('/home/kwakrhkr59/TrafficSliver/splitting_simulator/csv/mon_mode1_timestamp.csv')

def count_mean(traces):
    traces_file = natsorted(glob.glob(traces+'/*'))
    packet_list = []
    packet_dict = {}
    for instance_file in traces_file:
        if('join' in instance_file): continue
        if('csv' in instance_file): continue
        instance_name=instance_file.split('/')[-1].split('.')[0]
        instance = open(instance_file, 'r')
        instance = instance.read().split('\n')[:-1]
        path_num=int(instance_name.split('_')[-1])
        instance_num=-1
        if('-' in instance_name): site_num, instance_num=map(int, instance_name.split('_')[0].split('-'))
        else: site_num=int(instance_name.split('_')[0])
        if(site_num!=0 and path_num==0):
            print(site_num-1, end='\t')
            print(mean(packet_list), min(packet_list), max(packet_list))
            if(instance_num!=-1):
                packet_dict[str(site_num-1)+'-'+str(instance_num-1)] = {'mean': mean(packet_list),
                                            'min': min(packet_list),
                                            'max': max(packet_list)}

            else: 
                packet_dict[site_num-1] = {'mean': mean(packet_list),
                                            'min': min(packet_list),
                                            'max': max(packet_list)}
            packet_list = []
        cnt = len(instance)
        packet_list.append(cnt)
    if(instance_num!=-1):
        packet_dict[str(site_num)+'-'+str(instance_num)] = {'mean': mean(packet_list),
                                    'min': min(packet_list),
                                    'max': max(packet_list)}

    else: 
        packet_dict[site_num] = {'mean': mean(packet_list),
                                    'min': min(packet_list),
                                    'max': max(packet_list)}
    packet_dict[site_num] = {'mean': mean(packet_list),
                                'min': min(packet_list),
                                'max': max(packet_list)}
    df = pd.DataFrame(packet_dict).T
    df.to_csv('/home/kwakrhkr59/TrafficSliver/splitting_simulator/csv/mon_mode1_size.csv')
    # save as an image
    x = df.index.to_list()
    y_mean = df['mean']
    y_min = df['min']
    y_max = df['max']
    # plt.figure(figsize=(10, 6))
    # plt.xlabel('packet')
    # plt.ylabel('num of instance')
    # plt.plot(x, y_mean, label='mean')
    # plt.plot(x, y_min, label='min')
    # plt.plot(x, y_max, label='max')
    # plt.legend(loc='best')
    # plt.savefig("./images/mean.png")

if __name__ == '__main__':
    args = parser.parse_args()
    traces = args.path
    # count_mean(traces)
    # count_timestamp(traces)
    # count_packet_size(traces)
