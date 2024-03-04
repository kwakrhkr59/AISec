from natsort import natsorted
import glob
import os
traces = '/data/TrafficSliver/wf-singlesite/unmon/mode1/ts'
traces_file = natsorted(glob.glob(traces+'/*join*'))
for t in traces_file:
    os.remove(t)