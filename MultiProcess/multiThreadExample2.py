
import torch.distributed as dist
from torch.multiprocessing import Process
import torch
import os
import globalsFile
from load_data import createIterators
from policy_net import *
from mcts_translator import *
import time
import json


def run(rank,numProcesses,group,trg_tensor):
	
	print('gathering rank: ',rank)

	#now just continually gather and scatter until scatter gives a 
	#negative value which means we can exit
	#and also tell main_func that length is 0
	while(True):
		
		padded_output = torch.rand((15))
		print('Gathering rank: ',rank)
		print('rank: {}, sending to gather: {}'.format(rank,padded_output))
		dist.gather(tensor=padded_output,gather_list=None, dst=0,group=group) #send to process 2
		print('Finished gather: ',rank)

		model_response = torch.rand(5)
		dist.scatter(tensor=model_response,scatter_list=None,src=0,group=group)
		print('scatter rank: {}, given: {}'.format(rank,model_response))
		

#this is where the main process runs
def main_func(numProcesses,group,src_tensor):
	
	while(True):
		t = torch.zeros(15) #THE FINAL ELEMENT IS LENGTH WHEN NOT PADDED
		gather_t = [torch.ones_like(t) for _ in range(numProcesses)]
		
		#every process in group sends tensor to this gather_t list
		dist.gather(tensor=t,gather_list=gather_t,dst=0,group=group)
		
		print('GATHERED DATA')
		print(gather_t[1][:15])
		print(gather_t[2][:15])

		to_scatter = torch.rand((5,3))

		outputTens = torch.rand((5))
		
		#SIZE OF EACH TENSOR to scatter is main_params.num_children*2 +1
		#where first part is the actions, then probs, then leaf value
		#print('len to scatter: {}'.format(len(to_scatter)))
		print(to_scatter)
		to_scatter = np.split(to_scatter,3,axis=1)

		#this is vital to make sure memory isn't shared among these vectors
		to_scatter = [torch.clone(t).squeeze() for t in to_scatter]
		
		#to_scatter = [x.view(1,-1) for x in to_scatter]
		
		#print('TO SCATTER: ',to_scatter)
		print('just before scattering: ')
		#print(to_scatter[1].type)
		#print(to_scatter[1][:15])
		#print(to_scatter[2][:15])
		dist.scatter(tensor=outputTens,scatter_list=to_scatter,src=0,group=group)

		time.sleep(5)
		exit(1)
	

def init_processes(rank,numProcesses,src_tensor,trg_tensor):
	#using spawn_processes passes process index as first param

	#create the processes here then have our workers call one function
	#and our main process (rank=0) call another
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '29501'
	print('rank: {}, numProcesses: {}'.format(rank,numProcesses))
	dist.init_process_group('gloo', rank=rank, world_size=numProcesses)
	groupMCTS = dist.new_group([i for i in range(numProcesses)])
	
	if rank == 0:

		main_func(numProcesses,groupMCTS,src_tensor)
	else:
		run(rank,numProcesses,groupMCTS,trg_tensor[:,rank-1])



if __name__ == '__main__':
	
	
	size = 3
	src_tensor = torch.rand((15,2))
	trg_tensor = torch.rand((15,2))		
	torch.multiprocessing.spawn(init_processes,
				args=(size,src_tensor,trg_tensor),
				nprocs=size)
	
	#except:
	#	print('caught the exception')
	
	#print('sentence len: ',src_tensor.shape[0])
	#print('totalTime: ',time.time()-starting_time)

	'''
	for rank in range(size):
		modelToPass = network if rank==0 else None
		p = Process(target=init_processes, args=(rank, size,src_tensor,trg_tensor,modelToPass,main_params))
		p.start()
		processes.append(p)

	for 
	processes[0].join() #wait for our main process to finish
	'''
	print('Finished BATCH')
    #JUST DO ONE BATCH FOR NOW
	exit(0)

    #what we'll do is have processes write to files when
    #they're done. when get to here, means all other processes
    #are finished so can read files
    

