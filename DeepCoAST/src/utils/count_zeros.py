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


def count_zeros(traces):
    traces_file = natsorted(glob.glob(traces[0]+'/*'))
    zeros=0
    for instance_file in traces_file:
        instance = open(instance_file, 'r')
        instance=instance.read().split('\n')[:-1]
        print(instance_file)
        if(len(instance)==0): zeros+=1
    return zeros

if __name__ == '__main__':
    args = parser.parse_args()
    traces = args.path
    print(count_zeros(traces))
