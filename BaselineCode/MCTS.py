

'''
Here we load in trained value and policy networks and then 
jointly train them using MCTS from Alpha go zero

To do:
 1. To test if this makes any improvement over our current policy and value nets can also train
 	model who's prediction for next word is based off linear combo of the value of policy networks and the 
 	linear combo would be be the hyper parameter. (ie 0.3*policy + (1-0.3)*value) is pred

'''

'''
Algorithm:

1. at each sim we start at root
2. now sim out path until leaf is hit based on the Q(s,a)+U(s,a) rule
3. Now expand leaf (so init its branches probs as well as est value through value net)
4. Backup value through path taken, also increment N(s,a) along the path
	When backing up value, we set 
	One difference from paper we can do, is when simmed out and got EOS token, calculate actual BLEU
	from this leaf and back it up instead of estimated value (make sure no sims go past these terminal nodes)

5. We denote time steps of this MCTS as t=1 is haven't output anything, then t=2 have chosen first word
	and trying to thing of second word and so on. Now at each timestep t, we have (s_t,pi_t,z_t)
	where s_t is state at root at that timestep, pi_t is final move probs from s_t (based on proportion of 
	times each of it's branches were traversed. Finally, z_t is BLEU we get at time T after finishing our
	completed sentence. We now sample several of these (s_t,pi_t,z_t) and update our policy and value
	to more closely resemble these. 
	Now the reason we use this final z_T for all timesteps, is that this is the good estimate of the 
	value of following the policy from any of our s_t's. 
	What we can do is gather t sets of (s_t,pi_t,z_t) from each sentence, do this for a bunch of sentences,
	and then do updates. This will make it so that when sampling from these, our updates aren't so 
	correlated. 




I think the easiest way to show this algorithm is to split it up 
into one function that does the 1600 sims and another 

#this function takes in a german 'inputSentence' and english true 'trueOutput'
#then runs our MCTS algorithm and returns tuples of training data for our network
function runMCTS(inputSentence,trueOutput)
	root = BOS token, init prior probs
	dist_list = empty list (will have final move probs at each step t)
	states_used = empty list
	while root.curWord isn't EOS:
		states_used.append(root)
		for i in range(1600): #Do 1600 simulations each one until leaf is hit then choose next word
			curNode = root
			while (curNode isn't leaf or terminal)
				a = edge with max Q+U
				N(curNode,a) += 1 #update visit count
				curNode = getState(curNode,a) #gets new state when take action a from state curNode

			if curNode is terminal (ie EOS) and we aren't on test/valid set:
				v = Bleu(target, curNode sentence)
			else:
				priorProbs,v = network(curNode.state)
				#only keep say top 200 of these later on
			
			backup v through all Q values of edges on path taken
			(GET EQUATION FOR THIS)

		let prob of each action be proportional to N(root,a)^(1/Tau)
		dist_list.append(distribution)
		action = sample from dist
		root = getState(root,action) #child of root when take action
		
	bleuScore = BLEU(root.sentence, trueOutput)
	return all tuples of (bleuScore,dist_list[t],states_used[t]) for 0<=t<=T
	#now gradient update of policy and value using these
	#May want to do several sentences, shuffle these then update. 



TO ADD:
Each edge contains Q-value which is Q(parent, action)


Possible Improvements/differences in our alg:
 1. They seem to use all the (s_t,pi_t,z_t) for training and they are equally weighted, in reality
 	though, later on as t increases, since we get to reuse subtrees, there is much more information in those
 	subtrees since way more simulations have traversed through it. I think we should try taking this 
 	into account. 
 3. Make sure that Jerry is calculating true BLEU if in training and hit terminal state,
 	this will be much better est than using our value and backing up.  

 2. Don't want to have number of branches per node == vocab size since too large, instead we'll just
 	use maybe top 200 or maybe all the top ones such that they add up to like 0.99. To do this 
 	before updating our actual policy using these, maybe we should normalize them by their original prob (
 	if they added to 95% before then divide the new probs by 0.95) and then also don't include words we 
 	didn't use in the cross entropy (don't want to give 0 prob to these, instead should just ignore)
 	THIS PORTION NEEDS SOME MORE THINKING

 3. Right now using torch.tensor and each time we sim we append to it which it isn't meant for so maybe could use 
 	different datastructure

Questions:
 1. They set temperature to 1 for first 30 moves so that our move will be just based on visit counts from
    the MCTS. How should we do this? They also add in some noise to the prior probs to allow more exploration
    which is also a possibility. 
 2. Do we need to cite the code source?

'''

import torch
from lossAndTestClasses import get_bleu_scores


#I'm using some code from https://github.com/tensorflow/minigo/blob/master/mcts.py in this file

class MCTSNode(object):

	'''
	state: describes state of output so far so is words we've output up until this node
	c_puct: constant in the U variable of the UCB condition described in the paper
	children_inds: is vector of word indices that lead us to the child. For ex: if child at indice 0
					of child_prior corresponds to word with indice 14 in vocab, then first element of 
					children_inds is 14.
	num_children: is number of children each node should have, if 200 then just take 200 top prob words
					from this state.

	side note: action that led to being in this node is just last word in our state vector
	'''
	def __init__(self, state, prior, c_puct, num_children, is_terminal_node,parent=None):
		self.parent = parent
		self.state = state
		self.is_expanded = False
		self.is_terminal_node = False
		#self.losses_applied = 0  # number of virtual losses on this node (DO WE NEED THIS?)
		# save a copy of the original prior before it gets mutated by d-noise.
		
		#now want to trim down passed prior (only take say 200 highest prob words to make children)
		if not is_terminal_node:
			sorted_prior, inds = prior.sort(descending=True) 
			self.children_inds = inds[:num_children]
			self.original_prior = sorted_prior[:num_children]
			self.child_prior = original_prior.clone() #is copy but now will mutate the original slightly
			self.child_N = torch.zeros(child_prior.shape) 
			self.child_W = torch.zeros(child_prior.shape)
			self.children = {}  # dict of word inds to resulting MCTSNode (for ex can take word with ind 1 from here and would be children[1])
								#only add in children to this dict when traversed to the child
		else:
			self.children_inds = self.original_prior=self.child_prior=self.child_N=self.child_W=self.children=None					

	#get all child action scores at once
	def child_action_scores(self):
		#adding 1 to where N=0 doesn't matter since W also 0 there
        child_Q = self.child_W / (self.child_N + (self.child_N==0))  #want -inf
        child_U = self.c_puct*self.child_prior*torch.sqrt(self.child_N.sum())/(1+self.child_N)
        return child_Q + child_U



    #chooses an action in the MCTS sim (this action isn't actually a word we output though)
    def choose_sim_action(self,bos_ind,src_tensor,encoder_output_policynet,num_children,c_puct):
		action_scores = self.child_action_scores()
		action = action_scores.argmax().item()
		#now check if have child yet there
		child = None
		if action not in self.children:
			#need to create child and expand it
			child_state = torch.cat((self.state,torch.tensor([action]))) #can make this more efficient if need to
			if action == bos_ind: #then have hit terminal state so no need to expand
				child = MCTSNode(state=child_state,prior=None, 
								c_puct=c_puct, num_children=num_children, parent=self)

			else: #expand and initialize child
				output,encoder_output = policy_net.forward(src_tensor,child_state[:,None],
							src_key_padding_mask=src_key_padding_mask, tgt_mask=None,tgt_key_padding_mask=None,
                            memory_key_padding_mask=src_key_padding_mask,memory=encoder_output_policynet)
				#initialize
				prior = F.softmax(output[0,:,:],dim=1)
				child = MCTSNode(state=child_state,prior=prior, 
								c_puct=c_puct, num_children=num_children, parent=self)

		self.children[action] = child
		return child

    #differs from choose_action because here we're choosing real action (so word we output) based 
    #on visitation frequency from our MCTS (only call this from the current root)
    def choose_real_action(self):    


'''
args: bos_ind is indice in vocab of BOS
	  src_tensor,trg_tensor have batch dim already added
	  num_children: is number of children each node should have, if 200 then just take 200 top prob words
					from this state.
	  num_expansions: number of MCTS expansions we make before choosing an action 

We call MCTS with a single target sentence and this will return a bunch of pairs (s_l,pi_l,r_l)
'''
def runMCTS(bos_ind,src_tensor,trg_tensor,policy_net,value_net,c_puct,num_children,num_expansions,TGT):
	
	#TO DO: CREATE A MCTSClass which has all the constants we need as well as function for running these 
	#MCTS and keeping track of results and then finally updating our policy and value


	#can MAKE THIS MORE EFFICIENT ONCE WORKING (COULD DO SOME OPERATIONS HERE IN BATCHES)
	#save encoder hidden state for future
	root_state = torch.tensor([bos_ind])
	dec_input = root_state[:,None] #need to inject a batch dim as second dim
	output,encoder_output_policy = policy_net.forward(src_tensor,dec_input,src_key_padding_mask=src_key_padding_mask,
                          tgt_mask=None,tgt_key_padding_mask=None,
                          memory_key_padding_mask=src_key_padding_mask,memory=None)
	
	#output has dim [T,batch_size,vocab_size]
    priors = F.softmax(output[0,:,:],dim=1)

    #doing this so don't have to reput src_tensor through value net encoder over and over
    output,encoder_output_value = policy_net.forward(src_tensor,dec_input,src_key_padding_mask=src_key_padding_mask,
                          tgt_mask=None,tgt_key_padding_mask=None,
                          memory_key_padding_mask=src_key_padding_mask,memory=None)
	



	#we have a current root (that will change as we take actions)
	root = MCTSNode(state=root_state,prior=priors, c_puct=c_puct, 
					num_children=num_children, parent=None)

	#now start the actual MCTS

	'''
	PSEUDO
	while(root isn't terminal state)
		for i in range(1600): #do 1600 expansions from root (can have this as hyperparam)
			cur_node = root #now exand until hit leaf then backprop value
			while (cur_node is not leaf (and not terminal))
				cur_node = cur_node.choose_action() #chooses action to take (need new cur_node to set parent)
			#now have hit leaf
			if cur_node.is_terminal then don't expand
			else: expand
			v = get value est at cur_node (or true BLEU if at terminal state and back it up to top
			cur_node.backup_info(v)

		root = root.choose_root_action() #the choice here will be different than .choose_action() since 
										 #want to make choice based off visitation frequencies here
	'''
	#Implementation
	while(not root.is_terminal):
		for i in range(num_expansions):
			cur_node = root #do we need to deepcopy? (or when change cur_node does it just change pointer?)
			while (cur_node.is_expanded and not cur_node.is_terminal_node):
				cur_node = cur_node.choose_sim_action(bos_ind)
				#now need to get value at cur_node state and back it up through tree
				val = 0
				if cur_node.is_terminal_node:
					#get true BLEU
					val = get_bleu_scores(trg_tensor=trg_tensor,pred_tensor=cur_node.state[:-1,None],TGT).item()
				else:
					#get estimated BLEU from value net
					output,encoder_output = value_net.forward(src_tensor,cur_node.state[:,None],
														src_key_padding_mask=src_key_padding_mask,
                          								tgt_mask=None,tgt_key_padding_mask=None,
                          								memory_key_padding_mask=src_key_padding_mask,
                          								memory=encoder_output_value)
					#shape of output is [T,batch_dim,1] so 
					val = output[-1,0,0] #CHECK THAT THIS IS RIGHT
					

if __name__=='__main__':
	#here we test out using this MCTS

	#create root
	#suppose BOS = indice 0
	prior = torch.normal(0,0.3)
	prior /= prior.norm()
	root = MCTSNode(state=torch.tensor([0]),prior, children_inds, c_puct, 
					num_children, is_terminal_node=False,parent=None)





