

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
import json
from dataBuffer import DataBuffer

def run(rank,numProcesses,group,maxlen,main_params,trg_tensor):
	
	mcts = MCTS(tgt_tensor=trg_tensor,group=group,rankInGroup=rank,
				max_len=maxlen,main_params=main_params)


	#here actions is list of actions corresponding to the 
	#200 probabilities in mcts_probs
	bleu, output_states, mcts_probs,actions = mcts.translate_sentence()
	#write to file
	fileName = globalsFile.CODEPATH+'MCTSFiles/rank'+str(rank)+'.json'
	with open(fileName,'w') as f:
		json.dump([bleu,output_states,mcts_probs,actions],f)

	print('rank: ',rank, ' is done NOW WAITING FOR REST')


	while(True):
		#now just gathering and scattering until main exits
		padded_output = torch.zeros(maxlen+1)*globalsFile.BLANK_WORD_ID
		dist.gather(tensor=padded_output,gather_list=None, dst=0,group=group) #send to process 2
		model_response = torch.ones(2*main_params.num_children + 1).double()
		dist.scatter(tensor=model_response,scatter_list=None,src=0,group=group)



#this is where the main process runs
def main_func(numProcesses,group,src_tensor,maxlen,model):
	#exit('Exiting 1')
	starting_time = time.time()
	#POSSIBLE PROBLEM:
	#Some of the processes will finish before others from needing less calls if somehow continually hit EOS on their sims
	#now continually gather then scatter until we see that
	#the results from gather
	
	while(True):
		t = torch.zeros(maxlen+1) #THE FINAL ELEMENT IS LENGTH WHEN NOT PADDED
		gather_t = [torch.ones_like(t) for _ in range(numProcesses)]
		
		#every process in group sends tensor to this gather_t list
		dist.gather(tensor=t,gather_list=gather_t,dst=0,group=group)
		
		#print('GATHERED DATA')
		#print(gather_t[1][:15])
		#print(gather_t[2][:15])

		#trim them down to the maximum length of all gathered when remove padding
		#don't use first element in list since it's from this process
		dec_lengths = torch.tensor([x[-1] for x in gather_t[1:]]).long()
		#print('Dec_lengths: ',dec_lengths)
		#assert(min(dec_lengths) > 0)
		max_gathered_len = int(dec_lengths.max().item())
		if max_gathered_len == 0:
			#for some reason last scatter not seen by other processes so best 
			#way to shut them all down is throw exception which returns 
			#control our main function.
			exit(1) 

		#print('max gathered len:',max_gathered_len)
		#print('gather_t: ',gather_t)
		#TO DO: 
		#ONCE THIS IS WORKING: don't send blanks through, can filter

		dec_input = torch.cat([x[:max_gathered_len].view(-1,1) for x in gather_t[1:]],1).long()
		
		#print('dec_input: ',dec_input)
		dec_lengths[dec_lengths==0]+=1 #allows function to work for trees that are finished
		#mask for decoder_input happens within this function
		log_probs, values = model.forward(src_tensor,dec_input,
										sentence_lens=dec_lengths,req_grad=False)

		#need to get top model.num_children probs and their corresponding actions
		#which are the indices
		#print('log probs shape: ',log_probs.shape)
		sorted_probs,inds = torch.sort(log_probs,dim=1,descending=True)
		inds = inds.double() #so that concat with probs and values

		#now stack sorted_probs under inds then put value underneath, (this is what 
		#the other processes are expecting as format)
		#print('INDS shape: ',inds.shape)
		#print('values shape: ',values.shape)
		to_scatter = torch.cat([inds[:,:model.num_children].transpose(0,1),
							sorted_probs[:,:model.num_children].transpose(0,1),
							values.unsqueeze(0)], dim=0).to('cpu')
		#print('to_scatter shape: ',to_scatter.shape)
		#print('values: ',values)
		#print(to_scatter[-1,:])
		#print('shape to_scatter: ',to_scatter.shape)
		#print('to scatter type : ',to_scatter.type())
		#print('shape to_scatter: ',to_scatter.shape)
		#now have a tensor which we need to split column wise into lists
		to_scatter = list(np.split(to_scatter,to_scatter.shape[1],axis=1))
		#print('after split: len: {}, first el: {}'.format(len(to_scatter),to_scatter[0]))
		#print('first 50 of to_scatter')
		#print(to_scatter[1][:50])
		#exit(1)

		#need to clone compoennets so that they don't share memory
		to_scatter = [t.clone().squeeze(1) for t in to_scatter]
		#print('len to_scatter: {}, shape to_scatter[0]: {}'.format(len(to_scatter),to_scatter[0].shape))
		#now add fake tensor for this process to start of this list
		to_scatter.insert(0,torch.ones(len(to_scatter[0])).double())
		
		outputTens = torch.ones(len(to_scatter[0])).double()
		
		#SIZE OF EACH TENSOR to scatter is main_params.num_children*2 +1
		#where first part is the actions, then probs, then leaf value
		#print('len to scatter: {}'.format(len(to_scatter)))
		#print('just before scattering: ')
		#print(to_scatter[1].type)
		#print(to_scatter[1][:15])
		#print(to_scatter[2][:15])
		dist.scatter(tensor=outputTens,scatter_list=to_scatter,src=0,group=group)

	

def init_processes(rank,numProcesses,src_tensor,trg_tensor,modelToPass,main_params):
	#using spawn_processes passes process index as first param

	#create the processes here then have our workers call one function
	#and our main process (rank=0) call another
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '29500'
	print('rank: {}, numProcesses: {}'.format(rank,numProcesses))
	dist.init_process_group('gloo', rank=rank, world_size=numProcesses)
	groupMCTS = dist.new_group([i for i in range(numProcesses)])
	
	maxlen = trg_tensor.shape[0] + 5 #max number of decode steps/ length in MCTS
	if rank == 0:
		main_func(numProcesses,groupMCTS,src_tensor,maxlen,modelToPass)
	else:
		run(rank,numProcesses,groupMCTS,maxlen,main_params,trg_tensor[:,rank-1])




if __name__ == '__main__':
	
	#now create data_set iterators
	dataset_dict = createIterators(globalsFile.BATCHSIZE,globalsFile.DATAPATH)

	eng_vocab = dataset_dict['TGT'].vocab.itos

	src_vocab_size = len(dataset_dict['SRC'].vocab.itos)
	main_params = MainParams(dropout=0.2, src_vocab_size=src_vocab_size,
                  tgt_vocab_size=len(eng_vocab), batch_size=5,l2_const=1e-4,
                  c_puct=5,num_sims=10,temperature=1e-3,
                  tgt_vocab_itos=dataset_dict['TGT'].vocab.itos,
                  num_children=200,is_training=False)


	policy_path = globalsFile.MODELPATH + 'policy_supervised_RLTrained.pt'
	value_path = globalsFile.MODELPATH + 'value_supervised_RLTrained.pt'
	policy_path = None
	value_path = None #want to use cpu for now
	network = PolicyValueNet(main_params=main_params,path_to_policy=policy_path,
							path_to_value=value_path)
	
	
	data_buffer = DataBuffer()
	for epoch in range(1000): #number iterations over dataset
		data_iter = 'train_iter'
		sentences_processed = 0
		for batch_count, batch in enumerate(dataset_dict[data_iter]):
			#print('src shape: {}, tgt shape: {}'.format(vars(batch)['de'].shape,vars(batch)['en'].shape))
			src_tensor = vars(batch)['de'] #will be (S,batch_size) where S is max num tokens of sentence in this batch
			trg_tensor = vars(batch)['en'] 
			sentences_processed += src_tensor.shape[1]
			
			#REMOVE THIS
			#if src_tensor.shape[0] > 4:
			#		continue

			main_params.is_training = True if 'train' in data_iter else False
			
			#now create processes to use this
			size = src_tensor.shape[1]+1 #one process per sentence and one to do computation
			print('sentence len: ',src_tensor.shape[0])
			starting_time = time.time()
			
			try: 
				torch.multiprocessing.spawn(init_processes,
						args=(size,src_tensor,trg_tensor,network,main_params),
						nprocs=size)
			
			except Exception as e:
				if (str(e)=='process 0 terminated with exit code 1'):
					#this is how we wanted to terminate
					pass
				else:
					print('EXITING IN BAD SCENARIO')
					exit(1)
			
			print('sentence len: ',src_tensor.shape[0])
			print('totalTime: ',time.time()-starting_time)

			print('Finished BATCH')

			for rankNum in range(1,size):
				fileName = globalsFile.CODEPATH+'/MCTSFiles/rank'+str(rankNum)+'.json'
				with open(fileName) as f:
					data = json.load(f)
					#[bleu,output_states,mcts_probs,actions]
					src_ = src_tensor[:,rankNum-1]
					data_buffer.add_examples(src_,data)

			#print('sent all to files')
			#exit()

			if sentences_processed % 500:
				#now do some updates to the model
				while(True):
					data_buffer.set_iterator(batch_size=main_params.batch_size,shuffle=True)
					src_,dec_,actions,probs,bleus = data_buffer.next()	
					if src_.shape[1] < main_params.batch_size:
						break
					loss = network.train_step(src_,dec_,probs,actions,bleus)

					#exit()
			
			
		
		#JUST DO ONE BATCH FOR NOW
		#exit(0)


			#now want to read in the written files to have examples 
			#to update model with




    
