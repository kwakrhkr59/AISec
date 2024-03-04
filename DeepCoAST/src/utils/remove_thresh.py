from natsort import natsorted
import glob
import os
traces = '/data/TrafficSliver/wf-singlesite/mon_amazon/mode1/ts'
traces_file = natsorted(glob.glob(traces+'/*'))
out_folder = '/data/TrafficSliver/wf-singlesite/df_30/mon_amazon'
thresh = 30
delete_cnt = 0
write_cnt = 0

for t in traces_file:
    instance = open(t, 'r')
    instance = instance.read().split('\n')[:-1]
    new_instance = ""
    if (len(instance) == 0 or float(instance[-1].split('\t')[0]) < thresh):
        # print(t, "삭제", end='\t삭제 이유: ')
        # if (len(instance) == 0): print('빈 파일')
        # else: print('last timepacket =', instance[-1].split('\t')[0])
        delete_cnt += 1
        continue
    for packet in instance:
        time = float(packet.split('\t')[0])
        if (time >= thresh): break
        new_instance = new_instance + str(packet) + '\n'
    
    out_file = out_folder + '/' + t.split('/')[-1]
    with open(out_file, 'w') as f:
        f.write(new_instance)
    write_cnt += 1

print(delete_cnt, "개 삭제")
print(write_cnt, "개 파일 생성")