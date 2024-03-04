from argparse import ArgumentParser
import os
import pickle


parser = ArgumentParser()
parser.add_argument('--file')
parser.add_argument('--out_dir')
args = parser.parse_args()


USE_SUBLABEL = False
URL_PER_SITE = 10
TOTAL_URLS   = 950


"""
Open the pickle file and load samples into a flat array.

The monitored data structure is a dictionary containing lists of 
samples in which the keys to the dictionary is the URL number.
Every 10 URLs in sequence (e.g. 0-19, 20-39, ...) belong to one domain.
"""
print("Loading datafile...")
with open(args.file, 'rb') as fi:
    data = pickle.load(fi)
X = []
y = []
for i in range(TOTAL_URLS):
    if USE_SUBLABEL:
        label = i
    else:
        label = i // URL_PER_SITE
    for sample in data[i]:
        X.append(sample)
        y.append(label)
size = len(y)


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
sample_idx = {i: 0 for i in range(len(set(y)))}
for i,sample,label in zip(range(size),X,y):
    print(f'\tprogress: {i}/{size}', end='\r', flush=True)
    with open(os.path.join(args.out_dir, f'{label}-{sample_idx[label]}'), 'w') as fi:
        for c in sample:
            ts = abs(c)
            dr = 1 if c > 0 else -1
            fi.write(f'{ts}\t{dr*512}\n')
    sample_idx[label] += 1
print('')

