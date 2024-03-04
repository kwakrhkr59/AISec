from natsort import natsorted
import glob
import pandas as pd

traces = "/data/TrafficSliver/DeepCoffea/splitted/CrawlE_Proc/inflow/mode2/ts"
trace = natsorted(glob.glob(traces + "/*split*"))

index = 0
df = {}
for trace_path in trace:
    index += 1
    if (index % 1000 == 0):
        print(index)

    with open(trace_path, 'r') as f:
        lines = f.readlines()

    last_timestamp = 0 if len(lines) == 0 else float(lines[-1].split('\t')[0])
    df[trace_path] = last_timestamp
    # if (last_timestamp > 10):
    #     print(trace_path + '\t' + str(last_timestamp))

df = pd.DataFrame(df)
df.to_csv("/home/kwakrhkr59/TrafficSliver/splitting_simulator/csv/timestamps.csv")