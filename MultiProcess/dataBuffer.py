

import torch
import numpy as np
import globalsFile

class DataBuffer:
	def __init__(self):
		self.src_tensors = []
		self.dec_tensors = []
		self.bleus = []
		self.dists = [] #is pi in alpha go zero
		self.actions = [] #is actions that each prob in dists corresponds to

	def set_iterator(self,batch_size,shuffle=True):
		self.num_iters = 0 #number of examples we've iterated over
		self.batch_size = batch_size
		self.examples_order = np.arange(self.len())
		if shuffle:
			np.random.shuffle(self.examples_order)


	def len(self):
		return len(self.src_tensors)

	
	def add_examples(self,src_,data):
		bleu,output_states,mcts_probs,actions = data
		
		#print('len output_state: {}, type: {}'.format(len(output_states[0]),type(output_states[0])))
		for i in range(len(output_states)):
			self.src_tensors.append(src_.clone())
			self.actions.append(torch.tensor(actions[i]))
			self.dec_tensors.append(torch.tensor(output_states[i]))
			self.dists.append(torch.tensor(mcts_probs[i]))
			self.bleus.append(bleu)

	def next(self):
		#draw batch of size batch_size
		inds = np.arange(self.num_iters,min(self.num_iters+self.batch_size,self.len()))
		src_list = [self.src_tensors[i] for i in inds]
		max_src_len = max([len(x) for x in src_list])

		#now pad
		src_mat = torch.ones((max_src_len,len(inds)))*globalsFile.BLANK_WORD_ID
		for col,ind in enumerate(inds):
			src_mat[:len(src_list[ind]),col] = src_list[ind]

		dec_list = [self.dec_tensors[i] for i in inds]
		max_dec_len = max([len(x) for x in dec_list])

		#now pad
		dec_mat = torch.ones((max_dec_len,len(inds)))*globalsFile.BLANK_WORD_ID
		for col,ind in enumerate(inds):
			dec_mat[:len(dec_list[ind]),col] = dec_list[ind]

		bleus = torch.tensor([self.bleus[i] for i in inds])

		#make batch dim=1 (so columns)
		actions = torch.cat([self.actions[i].view(-1,1) for i in inds],dim=1)
		probs = torch.cat([self.dists[i].view(-1,1) for i in inds],dim=1)

		if self.num_iters+self.batch_size >= self.len():
			return None,None,None,None,None #reached end of dataset
		else:
			self.num_iters = self.num_iters + self.batch_size
		
		#convert to appropriate types
		src_mat = src_mat.long()
		dec_mat = dec_mat.long()
		actions = actions.long()
		probs = probs.double()
		bleus = bleus.double()


		return src_mat,dec_mat,actions,probs,bleus








