from natsort import natsorted
import glob
import os

dir = '/data/TrafficSliver/DeepCoffea/origin/inflow/'

print("i_*"+"\t"+"shortest_time")

count = 0

for i in range(342):
    traces_file = natsorted(glob.glob(dir+str(i)+"_*"))

    shortest_time = float('inf')
    tmp = 0
    for t in traces_file:
        tmp+=1
        f1 = open(t, 'r')
        last_line = f1.readlines()[-1].strip()
        # filename = t.split('/')[-1]
        
        time_str, size_str = last_line.split('\t')
        time = float(time_str)

        if time < shortest_time:
            shortest_time = time
            
    print(str(i)+'\t'+str(shortest_time))
    # if tmp==0: print(i)
    if(shortest_time < 50):
        count += 1
        # print(i)
        # for t in traces_file:
        #     os.remove(t)

print("under 50:", count)
    