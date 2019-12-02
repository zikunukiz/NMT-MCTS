
#example of using point to point communication between processes



import torch.distributed as dist
from torch.multiprocessing import Process
from torch.multiprocessing import Queue
import torch
import os
import globalsFile
from load_data import createIterators
from policy_net import *
from mcts_translator import *
import time
import json


#rank,numProcesses,group,queue
def run(rank,group,queue,queue_exit):
	
	print('running rank: ',rank)

	#now just continually gather and scatter until scatter gives a 
	#negative value which means we can exit
	#and also tell main_func that length is 0
	for iteration in range(rank):
		
		padded_output = torch.ones(11)*rank + 1
		padded_output[-1] = rank
		print('rank: {}, sending to gather: {}'.format(rank,padded_output))
		queue.put(padded_output)
		#this request object has methods: is_completed() and wait()
		#print('sent to queue')
		model_response = torch.rand(10)
		req = dist.irecv(tensor=model_response,src=0)
		req.wait()
		print('rank: {}, received: {}'.format(rank,model_response))
	
	print('rank: {}, EXITING'.format(rank))	
	queue_exit.put([rank])
	print ('put onto exit queue')
	exit(0)

#this is where the main process runs
def main_func(numProcesses,group,queue,queue_exit):
	print('main func running')
	starting_time2 = time.time()
	exited_processes = []

	while(True):
		#time.sleep(1)

		recv_tensors = []
		start_time = time.time()
		while(True):
			if not queue.empty():
				rec_tens = queue.get()
				recv_tensors.append(rec_tens)
			
			if len(recv_tensors) > 1 or time.time()-start_time>0.01:
					break

		#print('main received tensors',recv_tensors)

		for r in recv_tensors:
			#last element corresponds to rank it came from
			print('sending it back: ',r[:-1].clone()+3)
			print('destination: ',int(r[-1].item()))
			dist.isend(tensor=r[:-1].clone()+3,dst=int(r[-1].item()))

		if not queue_exit.empty():
			exited_processes.append(queue_exit.get())

		#print(len(exited_processes))
		if len(exited_processes) == numProcesses-1:
			print('everyones exited')
			print(time.time()-starting_time2)
			exit()


def init_processes(rank,numProcesses,queue,queue_exit):
	#create the processes here then have our workers call one function
	#and our main process[0] call another
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '29500'
	dist.init_process_group('gloo', rank=rank, world_size=numProcesses)

	#group = dist.new_group([i for i in range(numProcesses)])
	group=None
	if rank == 0:
		main_func(numProcesses,group,queue,queue_exit)

	else:
		run(rank,group,queue,queue_exit)



if __name__ == '__main__':
	
	
	size = 4	
	queue = Queue()
	queue_exit = Queue() #processes let main know when they've exited
	ps = []
	for rank in range(size):
		p = Process(target=init_processes,args=(rank,size,queue,queue_exit))
		p.start()
		ps.append(p)
	'''
	for i in range(size):
		p = Process(target=init_processes,)
	torch.multiprocessing.spawn(init_processes,
				args=[size,queue],
				nprocs=size)

	'''
	ps[0].join()






