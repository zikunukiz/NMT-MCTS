
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



def run(rank,numProcesses,group):
	
	print('running rank: ',rank)

	#now just continually gather and scatter until scatter gives a 
	#negative value which means we can exit
	#and also tell main_func that length is 0
	while(True):
		
		padded_output = torch.ones(10)*rank
		print('rank: {}, sending to gather: {}'.format(rank,padded_output))
	
		#this request object has methods: is_completed() and wait()
		req = dist.isend(tensor=padded_output,dst=0,group=group)
		req.wait()

		model_response = torch.rand(10)
		req = dist.irecv(tensor=model_response,src=0,group=group)
		req.wait()
		print('scatter rank: {}, given: {}'.format(rank,model_response))
		

class Manager:
	#manages the send and received tensors from main process
	#min_requests means as soon as min_requests requests come in
	#we run the model with those. 
	def __init__(self,min_requests,num_workers,len_receive,len_send):
		self.min_requests = min_requests
		self.num_workers = num_workers #doesn't count this process
		self.len_receive = len_receive #is the size of tensors we'll receive
		self.len_send = len_send 
		self.to_receive = [torch.zeros(self.len_receive) for i in range(self.num_workers)]
		self.recv_reqs = []
		for i in range(self.num_workers):
			#receive tensor from process with rank i+1
			self.recv_reqs.append(dist.irecv(tensor=self.to_receive[i],src=i+1))


	def gather_recv(self):
		#spin gathering receivable tensors
		start_time = time.time() 
		inds_recv = []
		while(len(inds_recv) < self.min_requests and time.time()-start_time < 3):
			inds_recv = []

			#REMEMBER TO REQUEST FROM ONE HIGHER THAN 
			#this ind since first element corresponds to rank 1
			for ind,req in enumerate(self.recv_reqs):
				if req.is_completed():
					inds_recv.append(ind)

		
		print('past here: ',len(inds_recv))
		print(inds_recv)

		if len(inds_recv) > 0:
			#now use the received tensors and call model with them
			#and send results back to these processes
			sent_reqs = []
			for ind in inds_recv:
				print('main received from rank: {}, {}'.format(ind+1,self.to_receive[ind]))
				
				to_send = self.to_receive[ind].clone()+1
				req = dist.isend(tensor=to_send,dst=ind+1)
				sent_reqs.append(req)
				self.to_receive[ind] = torch.zeros(self.len_receive) #reset this

			while(any([not req.is_completed() for req in sent_reqs])):
				print('passing here')
				pass

			#now ask to receive from each of these and do this all again
			for ind in to_recv:
				req = dist.irecv(tensor=self.to_receive[ind],src=ind+1)
				self.recv_reqs[ind] = req
			


			#now run model and send these the results
			#suppose add one to received



#this is where the main process runs
def main_func(numProcesses,group):
	print('main func running')
	
	torec1 = torch.zeros(10)
	torec2 = torch.zeros(10)
	req1 = dist.irecv(tensor=torec1,src=1,group=group)
	#req1.wait()
	#req2 = dist.irecv(tensor=torec1,src=2,group=group)

	start_time =time.time()
	while (time.time() - start_time < 1):# or not req2.is_completed()):
		pass

	print('RESULTS')
	req1.wait()
	print(req1.is_completed())
	print('received: {}'.format(torec1))#,torec2))

	#self.recv_reqs.append(dist.irecv(tensor=self.to_receive[i],src=i+1))

	'''
	manager = Manager(min_requests=2,num_workers=numProcesses-1,
						len_receive=10,len_send=10)


	manager.gather_recv()

	'''
	

def init_processes(rank,numProcesses):
	#using spawn_processes passes process index as first param

	#create the processes here then have our workers call one function
	#and our main process (rank=0) call another
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '29501'
	print('rank: {}, numProcesses: {}'.format(rank,numProcesses))
	dist.init_process_group('gloo', rank=rank, world_size=numProcesses)
	groupMCTS = dist.new_group([i for i in range(numProcesses)])
	#groupMCTS = None
	if rank == 0:

		main_func(numProcesses,groupMCTS)
	else:
		run(rank,numProcesses,groupMCTS)



if __name__ == '__main__':
	
	
	size = 2	
	torch.multiprocessing.spawn(init_processes,
				args=[size],
				nprocs=size)
