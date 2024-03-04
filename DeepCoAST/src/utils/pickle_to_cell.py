from natsort import natsorted
import glob
import argparse
import numpy as np
import pickle
import os

def open_pickle(directory, path):
    handle = open(path, 'rb')
    directions = pickle.load(handle, encoding = 'latin1')
    print(type(directions))
    print(directions.keys())


    for site in directions.keys():
        print(site)
        for instance in range(0, len(directions[site])):
            file_name = directory + "/" + str(site) + "-" + str(instance)
            f = open(file_name, 'w')
            for p in directions[site][instance]:
                if p > 0: dir = 1
                else:     dir = -1
                data = str(abs(p)) + "\t" + str(dir) + "\t" + str(dir*512) + "\n"
                f.write(data)
            f.close()
