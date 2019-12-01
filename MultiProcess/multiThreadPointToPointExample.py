
#example of using point to point communication between processes



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
		
		padded_output = torch.rand(10)
		print('rank: {}, sending to gather: {}'.format(rank,padded_output))
	
		#this request object has methods: is_completed() and wait()
		req = dist.isend(tensor=padded_output,dst=0)
		req.wait()

		model_response = torch.rand(10)
		req = dist.irecv(tensor=model_response,src=0)
		req.wait()
		print('scatter rank: {}, given: {}'.format(rank,model_response))
		

class Manager:
	#manages the send and received tensors from main process
	#min_requests means as soon as min_requests requests come in
	#we run the model with those. 
	def __init__(self,min_requests,num_workers,maxlen):
		self.min_requests = min_requests
		self.num_workers = num_workers
		self.len_receive = len_receive #is the size of tensors we'll receive
		self.len_send = len_send 
		self.to_receive = [torch.zeros(self.len_receive) for i in range(self.num_workers)]
		self.recv_reqs = []
		for i in range(self.num_workers):
			self.recv_reqs.append(dist.irecv(tensor=self.to_receive[i],src=i))


	def gather_recv(self):
		#spin gathering receivable tensors
		start_time = time.time() 
		while(num_recv < min_requests and time.time()-start_time < 0.2):
			num_recv = 0
			for req in self.recv_reqs:
				num_recv += req.is_completed

		if num_recv > 0:
			
	def req_tensors(self):
		#ask every worker who we don't have a current request out for to receive a tensor
		for req in self.recv_reqs:
			if req.is_completed()


	#if send a tensor out (make copy before sending)
	#before calling receive function: create new tensor to receive in
	def receive_tensors(self):
		while(reqs):
			pass
	def send_tensors(self):



#this is where the main process runs
def main_func(numProcesses,group,src_tensor):
	
	while(True):
		t = torch.zeros(10) #THE FINAL ELEMENT IS LENGTH WHEN NOT PADDED
		gather_t = [torch.ones_like(t) for _ in range(numProcesses-1)]
		
		reqs = [dist.irecv(tensor=gather_t[i],src=i) for i in range(1,numProcesses)]

		#now need to manage what we've received and what we haven't


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
