from argparse import ArgumentParser
import os
import pickle


parser = ArgumentParser()
parser.add_argument('--file', required=True)
parser.add_argument('--out_dir', required=True)
args = parser.parse_args()


#TOTAL_URLS = 55000  # total number in dataset
TOTAL_URLS = 19000  # cut down number to match monitored data size

"""
Open the pickle file and load samples into a flat array.
The data structure of the unmonitored pickle is just a 
flat array of samples.
"""
print("Loading datafile...")
with open(args.file, 'rb') as fi:
    x = pickle.load(fi)
size = len(x)


print(f'Total samples: {size}')
try:
    os.makedirs(args.outdir)
except:
    pass

"""
Sample arrays are singular lists in which each data value is the cell 
timestamped multiplied by direction. The timestamp and direction information 
is then easily seperable, and the size of each cell is 512 (the standard Tor 
cell size.
"""
print("Converting samples to plaintext...")
for i,sample in zip(range(size),x):
    print(f'\tprogress: {i}/{size}', end='\r', flush=True)
    with open(os.path.join(args.out_dir, f'{i}'), 'w') as fi:
        for c in sample:
            ts = abs(c)
            dr = 1 if c > 0 else -1
            fi.write(f'{ts}\t{dr*512}\n')
print('')

