

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
from policy_net import *
from mcts_translator import *
import time

def run(rank,numProcesses,group,maxlen,main_params,trg_tensor):
	
	mcts = MCTS(trg_tensor=trg_tensor,group=group,rankInGroup=rank,
				max_len=max_len,main_params=main_params)


	bleu, output_states, mcts_probs = mcts.translate_sentence()
	output_states = [x.tolist() for x in output_states]
	mcts_probs = [x.tolist() for x in mcts_probs]
	#write to file
	fileName = globalsFile.BASEPATH+'/MCTSFiles/rank'+str(rank)+'.json'
	with open(fileName,'w') as f:
		json.dump([bleu,output_states.tolist(),mcts_probs.tolist()],f)

	print('rank: ',rank, ' is done')

	#now just continually gather and scatter until scatter gives a 
	#negative value which means we can exit
	#and also tell main_func that length is 0
	while(True):
		padded_output = np.zeros(max_len+1)*globalsFile.BLANK_WORD_ID
		dist.gather(tensor=padded_output,gather_list=None, dst=0,group=self.group) #send to process 2

	    dist.scatter(tensor=model_response,scatter_list=None,src=0,group=self.group)
	    leaf_value = model_response[-1]
	    if leaf_value < 0:
	    	break
	    

#this is where the main process runs
def main_func(numProcesses,group,src_tensor,maxlen,model):
	
	#POSSIBLE PROBLEM:
	#Some of the processes will finish before others from needing less calls if somehow continually hit EOS on their sims
	#now continually gather then scatter until we see that
	#the results from gather
	while(True):
		t = torch.zeros(maxlen+1) #THE FINAL ELEMENT IS LENGTH WHEN NOT PADDED
		gather_t = [torch.ones_like(t) for _ in range(dist.get_world_size())]
		print('gathering in main')
		dist.gather(tensor=t,gather_list=gather_t,dst=0,group=group)
		
		#trim them down to the maximum length of all gathered when remove padding
		
		dec_lengths = [x[-1] for x in gather_t[1:]] #since first is this process
		#assert(min(dec_lengths) > 0)
		maxlen = max(dec_lengths)
		if maxlen == 0:
			#then now tell all the other processes they can quit
			#and we'll exit this process as well 
			#exit procedure will be, that after a scatter, if a process
			#receives a negative value back, then it exits,
			#and until it exits, always puts a 0 at the end of it's tensor
			#to show that it's empty. MAKE SURE DOESN'T CAUSE PROBLEMS
			#IN SENTENCE_LENS
			to_scatter = [np.zeros(model.num_children*2+1)*(-1)]*len(gather_t)
			outputTens = torch.ones(len(to_scatter[0]))
		
			#SIZE OF EACH TENSOR to scatter is main_params.num_children*2 +1
			#where first part is the actions, then probs, then leaf value

			dist.scatter(tensor=outputTens,scatter_list=to_scatter,src=0,group=group)

			break


		print('max len:',maxlen)

		#TO DO: 
		#ONCE THIS IS WORKING: don't send blanks through, can filter

		dec_input = torch.tensor([x[:maxlen] for x in gather_t[1:]]).transpose()
		dec_lengths[dec_lengths==0]+=1 
		#mask for decoder_input happens within this function
		log_probs, values = model.forward(src_tensor,dec_input,
										sentence_lens=dec_lengths,req_grad=False)

		#need to get top model.num_children probs and their corresponding actions
		#which are the indices
		sorted_probs,inds = torch.sort(log_probs,dim=1,descending=True)

		#now stack sorted_probs under inds then put value underneath, (this is what 
		#the other processes are expecting as format)
		to_scatter = torch.cat([inds[:,:model.num_children].transpose(0,1),
							sorted_probs[:,:model.num_children].transpose(0,1),
							values], dim=0).numpy().to('cpu')

		#now have a tensor which we need to split column wise into lists
		to_scatter = np.split(to_scatter,to_scatter.shape[1],axis=1)
		#now add fake tensor for this process to start of this list
		to_scatter.insert(0,np.ones(len(to_scatter[0])))

		
		outputTens = torch.ones(len(to_scatter[0]))
		
		#SIZE OF EACH TENSOR to scatter is main_params.num_children*2 +1
		#where first part is the actions, then probs, then leaf value

		dist.scatter(tensor=outputTens,scatter_list=to_scatter,src=0,group=group)


def init_processes(rank,numProcesses,src_tensor,trg_tensor,modelToPass,main_params):
	#create the processes here then have our workers call one function
	#and our main process[0] call another
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '29500'
	dist.init_process_group('gloo', rank=rank, world_size=numProcesses)

	groupMCTS = dist.new_group([i for i in range(numProcesses)])
	
	maxlen = trg_tensor.shape[1] + 5
	if rank == 0:
		main_func(numProcesses,groupMCTS,src_tensor,maxlen,modelToPass)

	else:
		run(rank,numProcesses,groupMCTS,maxlen,main_params,trg_tensor[:,rank-1])



if __name__ == '__main__':
	
	#now create data_set iterators
	datasetDict = createIterators(globalsFile.BATCHSIZE,globalsFile.DATAPATH)

    eng_vocab = dataset_dict['TGT'].vocab.itos
    src_vocab_size = len(dataset_dict['SRC'].vocab.itos)
    tgt_vocab_size = len(eng_vocab)
    main_params = MainParams(dropout=0.2, src_vocab_size=src_vocab_size,
                  tgt_vocab_size=len(eng_vocab), batch_size=-1,l2_const=1e-4,
                  c_puct=5,num_sims=100,temperature=1e-3,
                  tgt_vocab_stoi=dataset_dict['TGT'].vocab.stoi,
                  num_children=200,is_training=False)


	policy_path = globalsFile.MODELPATH + 'policy_supervised_RLTrained.pt'
	value_path = globalsFile.MODELPATH + 'value_supervised_RLTrained.pt'
	network = PolicyValueNet(main_params=main_params,path_to_policy=policy_path,
							path_to_value=value_path)
	
	for epoch in range(1000): #number iterations over dataset
		data_iter = 'train_iter'
		for batch_count, batch in enumerate(dataset_dict[data_iter]):
			#print('src shape: {}, tgt shape: {}'.format(vars(batch)['de'].shape,vars(batch)['en'].shape))
			src_tensor = vars(batch)['de'].to(main_params.device) #will be (S,batch_size) where S is max num tokens of sentence in this batch
			trg_tensor = vars(batch)['en'].to(main_params.device) 
			
			main_params.is_training = True if 'train' in data_iter else False
			
			#now create processes to use this
			size = src_tensor.shape[1]+1 #one process per sentence and one to do computation
		    processes = []
		    for rank in range(size):
		    	modelToPass = network if rank==0 else None
		        p = Process(target=init_processes, args=(rank, size,src_tensor,trg_tensor,modelToPass,main_params))
		        p.start()
		        processes.append(p)


		    processes[0].join() #wait for our main process to finish


		    #what we'll do is have processes write to files when
		    #they're done. when get to here, means all other processes
		    #are finished so can read files
		    




    
