

'''
Changes: 
define a condition object (that wraps a lock) which each of our main
classes gets as an attribute. Now before each thread expands:
	1. thread takes the lock and then waits on it(blocks)
	2. 

NOTE: python has a global interpreter lock only allowing 1 thread to be in 
execution at any point. If do multi process instead, each process
gets it's own python interpreter and memory so GIL isn't a problem. 
'''


import torch.distributed as dist
from torch.multiprocessing import Process
import torch
import os
import globalsFile
from load_data import createIterators

def run(rank,numProcesses,group):
	tensor = torch.ones(1)*rank
	
	#dst (int) – Destination rank, dist.gather(tensor, dst, gather_list, group)
	#dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group)
	#print('Rank ',rank,' has data ', tensor[0])
	gather_list = None
	dist.gather(tensor=tensor,gather_list=gather_list, dst=0,group=group) #send to process 2

	outputTens = torch.ones(1)
	dist.scatter(tensor=outputTens,scatter_list=None,src=0,group=group)
	print('Rank ',rank,' has data ', outputTens)

def main_func(numProcesses,group):
	#gather_list = [torch.ones(1),torch.ones(1)]
	t = torch.ones(1)
	gather_t = [torch.ones_like(t) for _ in range(dist.get_world_size())]
	dist.gather(tensor=torch.zeros(1),gather_list=gather_t,dst=0,group=group)
	#tensor = torch.ones(1)*9
	#dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group)
	print('IN MAIN gather: ', gather_t)
	gather_t = [i+1 for i in gather_t]
	#now add 1 to each of these then scatter back
	outputTens = torch.ones(1)
	dist.scatter(tensor=outputTens,scatter_list=gather_t,src=0,group=group)



def init_processes(rank,numProcesses):
	#create the processes here then have our workers call one function
	#and our main process[0] call another
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '29500'
	dist.init_process_group('gloo', rank=rank, world_size=numProcesses)

	group = dist.new_group([i for i in range(numProcesses)])

	if rank == 0:
		main_func(numProcesses,group)

	else:
		run(rank,numProcesses,group)



if __name__ == '__main__':

	#want to load data and loop over it
	datasetDict = createIterators(globalsFile.BATCHSIZE,globalsFile.DATAPATH)

	#load models

	'''
	pseudo:
	for batch in train_iter: (large batches)
		create process for each sentence and process to control gpu
		and have main process which stays in this loop here. 

		Create 2 groups, one contains just the sentences and gpu processes,
		and the other contains the sentences and this main process. 

		This allows the main process to wait on a gather from the other processes. 
		Need to figure out size of exact return values from each. 

		return all the generated from these processes


	'''





if __name__ == "__main__":
    size = 15
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size))
        p.start()
        processes.append(p)


    #dist.gather(None, 2, gather_list, group) #send to process 2

    processes[0].join() #wait for our main process to finish
    