# Code for simulating traffic-splitting strategies
# Used in: https://dl.acm.org/doi/10.1145/3372297.3423351
# Wladimr De la Cadena

#wdlc: Simulator of multipath effect over wang-style instances, IMPORTANT!!! use files in WANG Format with .cell extension. Three colums can be considered for each packet in the .cell file timestamp|direction|size
#Working methods Random, RoundRobin, Weighted Random and Batched Weighted Random and their variation to variable number of paths during the same page load. 

import numpy as np, numpy.random
import sys
import argparse
import glob
import random
from natsort import natsorted
import multipathN
import time
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--scheme", type=str, help="Splitting scheme", default='batched_weighted_random') 
parser.add_argument("-c", "--cells", type=int, help="Cells per circuit in RR (in others it is the initial value)", default=1) 
parser.add_argument("-m", "--circuits", type=int, help="In how many paths is the traffic divided", default=3)
parser.add_argument("-min", "--circuitsmin", type=int, help="In how many paths is the traffic divided", default=3)
parser.add_argument("-i", "--inputs", type=str, help="Circuit latencies file", default='circuits_latencies_new.txt')
parser.add_argument("-o", "--outfolder", type=str, help="Folder for output files", default='/home/kwakrhkr59/TrafficSliver/splitting_simulator/dataset_3/10_40/0.33_0.33_0.34') 
parser.add_argument("-w", "--weights", type=str, help="Weights for circuit (comma separated)", default='0.33,0.33,0.34') 
parser.add_argument("-r", "--ranges", type=str, help="Range of cells after of which the wr or wrwc schduler design again", default='10,40') 
parser.add_argument("-p",'--path', nargs='+', help=' fiPath of folder with instancesles (wang_format)', default='/scratch/TrafficSliver/BE-original/mon_standard')
parser.add_argument("-a", "--alpha", type=str, help="alpha values for the Dirichlet function default np.ones(m)", default='1,1,1')

schemes = ['round_robin','random','weighted_random', 'in_and_out', 'batched_weighted_random', 'bwr_var_paths', 'bwr_var_paths_strict', 'bwr_blocked', 'wr_var_paths', 'rr_var_paths', 'random_var_paths']
bwoh = [0]
MAX_CNT = 30

def saveInFile(input_name,split_inst,r,outfolder):
    # numberOfFiles = max(r)+1 # How many files, one per route
    numberOfFiles = 3
    outfiles = []
    for k in range(0,numberOfFiles):
        input_name2 = input_name.split('/')[-1]
        out_file_name = outfolder + "/" + input_name2 + "_split_" + str(k) + '.cell'
        outfiles.append(open(out_file_name,'w'))

    jointfilename = outfolder + "/" + input_name.split('/')[-1] + "_join"+ '.cell'
    jointfile = open(jointfilename,'w')
    for i in range(0,len(split_inst)):
        x_arrstr = np.char.mod('%.15f', split_inst[i][:-1])
        x_arrstr[1] = int(float(x_arrstr[1]))
        jointfile.write('\t'.join(x_arrstr) + '\n')

    fs = [0] * numberOfFiles
    ts_o = [0] * numberOfFiles
    cnt = [0] * numberOfFiles

    for i in range(0,len(split_inst)):
        rout = int(split_inst[i][3])
        if (fs[rout] == 0):
            ts_o[rout] = float(split_inst[i][0])
        fs[rout] = 1
        x_arrstr = np.char.mod('%.15f', split_inst[i])
        x_arrstr[1] = int(float(x_arrstr[1]))
        x_arrstr = x_arrstr.astype(float)
        strwrt =  str(x_arrstr[0] - ts_o[rout]) + '\t' + str(int(x_arrstr[1])) + '\t' + str(x_arrstr[2])
        outfiles[rout].write(strwrt+ '\n')	
        cnt[rout] += 1

    flag = True
    print("Size: ", end=" ")
    for i in range(numberOfFiles):
        print(cnt[i], end="\t")
        if cnt[i] < 10: flag = False
    return flag

def getCircuitLatencies(l,n):
    file_latencies = open(l,'r')
    row_latencies = file_latencies.read().split('\n')[:-1]
    numberOfClients = int(row_latencies[-1].split(' ')[0])
    randomclient = random.randint(1,numberOfClients)
    ## Get the multiple circuits of the selected client:
    multipath_latencies = []
    for laten in row_latencies:
        clientid = int(laten.split(' ')[0])
        if (clientid == randomclient):
            multipath_latencies.append(laten.split(' ')[2].split(','))	
    ## I only need n circuits, it works when n <  number of circuits in latency file (I had max 6)
    multipath_latencies = multipath_latencies[0:n]
    return multipath_latencies

def sim_bwr(n, instance, mplatencies, w_out, w_in, ranlow, ranhigh):
    routes_client = []
    routes_server = []
    sent_incomming = 0
    sent_outgoing = 0

    last_client_route =  np.random.choice(list(range(0,n)),p = w_out)
    last_server_route = np.random.choice(np.arange(0,n),p = w_in)

    for i in range(0,len(instance)):
        packet = instance[i]
        packet = packet.replace(' ','\t') #For compatibility when data is space sperated not tab separated
        direction = multipath.getDirfromPacket(packet)
        
        if (direction == 1):
            routes_server.append(-1) # Just to know that for this packet the exit does not decide the route
            routes_client.append(last_client_route) 
            sent_outgoing += 1
            C = random.randint(ranlow,ranhigh) #After how many cells the scheduler sets new weights
            if (sent_outgoing % C == 0): #After C cells are sent, change the circuits
                last_client_route =  np.random.choice(np.arange(0,n),p = w_out)

        if (direction == -1): 
            routes_client.append(-1) # Just to know that for this packet the client does not decide the route
            routes_server.append(last_server_route)
            sent_incomming += 1
            C = random.randint(ranlow,ranhigh) #After how many cells the scheduler sets new weights
            if (sent_incomming % C == 0): #After C cells are sent, change the circuits
                last_server_route = np.random.choice(np.arange(0,n),p = w_in)

    return routes_client, routes_server

def simulate(n,latencies,traces,outfiles,range_, alphas):
    print("Simulating BWR multi-path scheme...")
    traces_file = natsorted(glob.glob(traces+'/*'))  # traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))

    ranlow = int(range_.split(',')[0])
    ranhigh = int(range_.split(',')[1])

    failed = 0
    for instance_file in traces_file:
        for i in range(MAX_CNT):
            w_out = multipath.getWeights(n, alphas)
            w_in = multipath.getWeights(n, alphas)

            # if(int(instance_file.split('/')[-1].split('-')[0])%5==0): print(instance_file, end="")
            print(instance_file, end="\t")
            instance = open(instance_file,'r')
            instance = instance.read().split('\n')[:-1]
            # print(instance_file, len(instance))
            mplatencies = getCircuitLatencies(latencies, n)	# it is a list of list of latencies for each of m circuits. length = m

            routes_client, routes_server = sim_bwr(n, instance, mplatencies, w_out, w_in, ranlow, ranhigh)
            routes = multipath.joingClientServerRoutes(routes_client,routes_server)
            ##### Routes Created, next to the multipath simulation
            new_instance = multipath.simulate(instance,mplatencies,routes) # Simulate the multipath effect for the given latencies and routes

            if saveInFile(instance_file,new_instance,routes,outfiles):
                print("success.")
                break
            else:
                print("failed to create files.")
                if i == MAX_CNT - 1: failed += 1

    print(f"Splitting complete. Failed: {failed}")
       
if __name__ == '__main__':
    args = parser.parse_args()
    scheme_ = args.scheme
    if (scheme_ not in schemes):
        sys.exit("ERROR: Splitting scheme not supported")
    cells_per_circuit_ = args.cells
    paths_ = args.circuits
    paths_min = args.circuitsmin
    latencies_ = args.inputs
    traces_ = args.path
    outfolder_ = args.outfolder
    weights_ = args.weights
    range_ = args.ranges
    alpha_ = args.alpha
    val = 0
    starttime = time.time()

    simulate(paths_, latencies_,traces_,outfolder_,range_, alpha_)

    endtime = time.time()
    print("Multi-path Simulation done!!! I took (s):", (endtime - starttime))
    print("bandwidth OH: ", val)