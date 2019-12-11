# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes
Code Base: https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/mcts_alphaZero.py
"""

import numpy as np
import copy
import torch
from torch.autograd import Variable
import torch.distributed as dist
import globalsFile
from translate import Translation
import time
import torch.nn.functional as F
cimport numpy as np
from cpython cimport array
import array
from translate cimport Translation
#remaking TreeNodes to speed things up.
#now TreeNode just has parent and dict to children but also 
#has several fields which now correspond to the edges leaving this node
#and can store as arrays which will speed up playouts

INT = np.int
DOUBLE = np.double

ctypedef np.double_t DOUBLE_t
ctypedef np.int_t INT_t

#create an extension type
cdef class TreeNode:
    #define attributes
    cdef object [:]  _children
    cdef DOUBLE_t [:] _n_visits
    cdef INT_t [:] _actions 
    cdef DOUBLE_t [:] _priors
    cdef DOUBLE_t [:] _Q
    cdef object _parent
    cdef INT_t parent_ind_action_taken
    cdef DOUBLE_t _V
    cdef INT_t isleaf
    cdef DOUBLE_t sum_visits #total number of visits to this node (used in select function)
    #cdef DOUBLE_t [:] _n_visits_nonzero #this is number of backups of non zero values 
    
    def __init__(self,object parent,INT_t parent_ind_action_taken):
        self._parent = parent
        self.parent_ind_action_taken = parent_ind_action_taken #is index of action in parent action vector taken from parent to get here
        self._children = None #ARRAY OF CHILDREN TreeNodes WHICH CORRESPOND TO THE ACTIONS VECTOR
        self._n_visits = None #number of visits to each of it's branches
        self._actions = None #array of actions that the edges correspond to
        self._priors = None #prior probs that the edges correspond to 
        self._Q = None #Q values of edges
        self._V = -0.1 #is value computed for this state
        self.isleaf = 1
        self.sum_visits = 1
        
    def expand(self,np.ndarray[INT_t] top_actions,np.ndarray[DOUBLE_t, ndim=1] top_probs):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """   
        #assert(len(self._children)==0)
        cdef INT_t len_actions
        len_actions = top_actions.size
        
        #print('len_actions: ',len_actions)
        
        self.isleaf = 0
        self._actions = np.copy(top_actions) #top_actions.clone().detach()
        self._priors = np.copy(top_probs) #.clone().detach()
        
        #we have each each element in a corresponding to Child at same position in _children
        self._children = np.empty(shape=(len_actions)).astype(dtype=TreeNode)
        for ind in range(len_actions):
            self._children[ind] = TreeNode(self,ind)
         
        self._n_visits = np.zeros(len_actions)
        #self._n_visits_nonzero = np.zeros(len_actions)
        self._Q = np.zeros(len_actions)
        
        
    def select(self, DOUBLE_t c_puct):
        #HAVE Copied this into playout to speed things up, also nested all the numpy calls 

        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        #number of times to the parent is just sum of children visits + 1 for when parent was expanded
        #self.sum_visits is parent number visits
        cdef DOUBLE_t u
        cdef DOUBLE_t v
        
        u = np.multiply(c_puct,self._priors)
        u = np.multiply(u,(self.sum_visits**0.5))
        u = np.divide(u, np.add(self._n_visits,1.))
        
        #u = (c_puct*self._priors*np.sqrt(self.sum_visits) / (1 + self._n_visits))
        v = np.add(u,self._Q)
        indice_best_action = np.argmax(v)
        
        return indice_best_action   
    
    def backup(self,DOUBLE_t leaf_value):
        #have moved this inside playout function for speed
        
        cdef TreeNode node
        node = self
        node.sum_visits += 1
        while(node._parent):
            #w:
            act_ind = node.parent_ind_action_taken
            node = node._parent
            node.sum_visits += 1
            node._n_visits[act_ind] += 1
            node._Q[act_ind] += (leaf_value - node._Q[act_ind]) / node._n_visits[act_ind]


            
class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""
    
    #group is for the group of processes which allows us 
    #to send and receive tensors with other processes in group
    def __init__(self, tgt_tensor, group, rankInGroup, max_len,main_params,queue):
        """
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        
        #cdef TreeNode self._root
        
        self.group = group
        self.queue = queue
        self.max_len = max_len #max translation length (need to path our tensors up to this for interprocess communication)
        self.rankInGroup =rankInGroup#is way of identifying this process in the group
        self._root = TreeNode(None, 1)
        self._c_puct = main_params.c_puct
        self._n_playout = main_params.num_sims
        self.num_children = main_params.num_children
        self.temperature = main_params.temperature 
        #self.time_last_scatter = 0 #want to make sure never go mroe than 0.5 seconds without scattering so other processes don't wait on this
        # with the default temp=1e-3, it is almost equivalent
        # to choosing the move with the highest prob
        
        self.translation = Translation(tgt=tgt_tensor[tgt_tensor!=globalsFile.BLANK_WORD_ID].numpy(), vocab=main_params.tgt_vocab_itos)
        self.is_training = main_params.is_training #lets us know if on train set            
        self.possible_inds = np.arange(self.num_children)
        
    #def _playout(self, Translation state):
    """
    #THIS HAS PROBLEMS (mixes use of state and translation variables wrongly
    def _playout(self, Translation state):  
        #Run a single playout from the root to the leaf, getting a value at
        #the leaf and propagating it back through its parents.
        #State is modified in-place, so a copy must be provided.
        
        cdef DOUBLE_t [:]  u
        cdef DOUBLE_t [:] v
        cdef INT_t indice_best_action
        cdef INT_t best_action
        cdef TreeNode node
        cdef DOUBLE_t _c_puct
        cdef INT_t EOS_ID
        
        EOS_ID = globalsFile.EOS_WORD_ID
        _c_puct = self._c_puct
        #playout_t1 = time.time()
        node = self._root
        while(1):
            if node.isleaf or state.last_word_id == EOS_ID:
                break
            
            #if nest all these calls we get decent speed boost
            indice_best_action = np.argmax(np.add(np.divide(np.multiply(np.multiply(_c_puct,node._priors),(node.sum_visits**0.5)),np.add(node._n_visits,1.)),node._Q))
            
            
            
            #u = np.multiply(_c_puct,node._priors)
            #u = np.multiply(u,(node.sum_visits**0.5))
            #u = np.divide(u, np.add(node._n_visits,1.))

            #u = (c_puct*self._priors*np.sqrt(self.sum_visits) / (1 + self._n_visits))
            #indice_best_action = np.argmax(np.add(u,node._Q))
            #indice_best_action = np.argmax(u)

            
            #number of times to the parent is just sum of children visits + 1 for when parent was expanded
            #self.sum_visits is parent number visits
            #u = (self._c_puct*node._priors*(node.sum_visits**0.5) / (1 + node._n_visits))
            #indice_best_action = np.argmax(u+node._Q)
            best_action = node._actions[indice_best_action]
            node = node._children[indice_best_action]
            
            #directly copy in the do_move code to speed things up: replaces: state.do_move(best_action)
            state.last_word_id = best_action
            state.output[state.len_output] = best_action
            state.len_output += 1

        
        #p2 = time.time()-playout_t1
        #print('playout: ',p2)
        # print("output: {}".format(state.output.tolist()))
        
        if node._V >= 0: #means it has been set
            leaf_value = node._V 

        else:
            # Check for end of translation 
            end, bleu = state.translation_end()

            if (not end or not self.is_training) and not ((end or state.len_output==self.max_len)and self.is_training):

                #last element of padded output will be rank of this process
                #and second last element is length of the output without padding
                padded_output = torch.ones(self.max_len+2)*globalsFile.BLANK_WORD_ID
                padded_output[:state.len_output]= torch.tensor(state.output[:state.len_output])
                padded_output[-2] = state.len_output
                padded_output[-1] = self.rankInGroup
                #print('rank: {}, Sending to queue: {}'.format(self.rankInGroup,padded_output[:15]))
                self.queue.put(padded_output)

                model_response = torch.ones(2*self.num_children + 1).double()
                req = dist.irecv(tensor=model_response,src=0)
                req.wait()

                top_actions = model_response[:self.num_children].long()
                #print('Top actions received',top_actions[:15])
                top_probs = model_response[self.num_children:-1]
                leaf_value = model_response[-1].item()

                
                #HERE is where we call gather with group containing model
                #want to send main process our output so far padded
                #padded_output = torch.ones(self.max_len+1)*globalsFile.BLANK_WORD_ID
                #padded_output[:len(self.translation.output)]= self.translation.output
                #padded_output[-1] = len(self.translation.output)
                #print('Sending gatherer: ',padded_output[:15])
                #dist.gather(tensor=padded_output,gather_list=None, dst=0,group=self.group) #send to process 2

                #model_response = torch.ones(2*self.num_children + 1).double()
                #dist.scatter(tensor=model_response,scatter_list=None,src=0,group=self.group)
                #self.time_last_scatter = time.time()
                
                if not end and state.len_output < self.max_len:
                    #assert(len(top_actions)==self.num_children)
                    #print('expanding at new state, rank: ',self.rankInGroup)
                    node.expand(top_actions.numpy(),top_probs.numpy())

            else:
                #print('USING BLEU')
                leaf_value = bleu
        
        # Update value and visit count of nodes in this traversal
        node._V = leaf_value
        
        #p3 = time.time()
        node.backup(leaf_value)
        
        #playout_t2 = time.time()-p3
        #print('backup time: ',playout_t2)
        #print('Time for playout: ',playout_t2)
    """
    def get_move_probs(self,Translation translation, TreeNode root):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current state (input and its translation so far)
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        #_n_playout is number of simulations per action chosen
        cdef INT_t copy_last_word
        cdef INT_t [:] copy_output
        cdef INT_t copy_len_output
        cdef DOUBLE_t [:] children_n_visits
        cdef INT_t n_playout
        cdef TreeNode node
        cdef DOUBLE_t _c_puct
        cdef INT_t EOS_ID
        cdef INT_t indice_best_action
        cdef INT_t best_action
        cdef INT_t end
        cdef INT_t is_training
        cdef INT_t max_len
        cdef DOUBLE_t leaf_value
        
        max_len = self.max_len
        is_training = self.is_training
        EOS_ID = globalsFile.EOS_WORD_ID
        _c_puct = self._c_puct
        
        #get_move_time = time.time()
        
         
        n_playout = self._n_playout
        
        #print('Start:rank: {}, output: {}, len: {}, lastWord: {}'.format(self.rankInGroup,[x for x in translation.output],translation.len_output,translation.last_word_id))
        
        #translation.output = np.copy(translation.output) #need to get type assigned
        
        #Tested that translation doens't change here. 
        
        for n in range(n_playout):
            # print("simulation - {}".format(n))
            copy_last_word = translation.last_word_id #copy.deepcopy(self.translation.last_word_id)
            copy_output = np.copy(translation.output)
            copy_len_output = translation.len_output
            
            #INSERTING CODE FOR PLAYOUT TO INCREASE SPEED
            
            node = root
            while(1):
                if node.isleaf or translation.last_word_id == EOS_ID:
                    break

                #if nest all these calls we get decent speed boost
                indice_best_action = np.argmax(np.add(np.divide(np.multiply(node._priors, (node.sum_visits**0.5)*_c_puct),np.add(node._n_visits,1.)),node._Q))
                
                best_action = node._actions[indice_best_action]
                node = node._children[indice_best_action]

                #directly copy in the do_move code to speed things up: replaces: state.do_move(best_action)
                translation.last_word_id = best_action
                translation.output[translation.len_output] = best_action
                translation.len_output += 1

            
            if node._V >= 0: #means it has been set
                leaf_value = node._V 

            else:
                
                # Check for end of translation 
                end = (translation.last_word_id == EOS_ID)
                
                if is_training and (end or translation.len_output==max_len):
                    end, bleu = translation.translation_end() #only need to fully run this function here.
                    #print('USING BLEU')
                    leaf_value = bleu
                
                else:

                    #last element of padded output will be rank of this process
                    #and second last element is length of the output without padding
                    
                    
                    padded_output = torch.ones(self.max_len+2)*globalsFile.BLANK_WORD_ID
                    padded_output[:translation.len_output]= torch.tensor(translation.output[:translation.len_output])
                    padded_output[-2] = translation.len_output
                    padded_output[-1] = self.rankInGroup
                    #print('rank: {}, Sending to queue: {}'.format(self.rankInGroup,padded_output[:15]))
                    self.queue.put(padded_output)

                    model_response = torch.ones(2*self.num_children + 1).double()
                    req = dist.irecv(tensor=model_response,src=0)
                    req.wait()

                    top_actions = model_response[:self.num_children].long()
                    #print('Top actions received',top_actions[:15])
                    top_probs = model_response[self.num_children:-1]
                    leaf_value = model_response[-1].item()
                    
                    
                    if not end and translation.len_output < self.max_len:
                        #assert(len(top_actions)==self.num_children)
                        #print('expanding at new state, rank: ',self.rankInGroup)
                        node.expand(top_actions.numpy(),top_probs.numpy())

                
            # Update value and visit count of nodes in this traversal
            node._V = leaf_value
            
            if leaf_value != 0:  #ADDED THIS SO THAT DONT BACKUP ZERO VALUES (ONLY BLEU SCORES)
                #CODE FOR BACKUP
                node.sum_visits += 1
                while(node._parent):
                    act_ind = node.parent_ind_action_taken
                    node = node._parent
                    node.sum_visits += 1
                    node._n_visits[act_ind] += 1
                    node._Q[act_ind] += (leaf_value - node._Q[act_ind]) / node._n_visits[act_ind]

            
            #node.backup(leaf_value)
    
            #END OF CODE FOR PLAYOUT
            
            #reset translation for next playout
            translation.last_word_id = copy_last_word
            translation.output = np.array(copy_output) #np.copy(copy_output)
            translation.len_output = copy_len_output
        
        
        #get_move_time = time.time()-get_move_time
        #print('get_move_time: ',get_move_time)
        '''
        if time.time() - self.time_last_scatter > globalsFile.MAX_TIME_BETWEEN_SCATTERS:
            #do a fake gather and scatter
            padded_output = torch.ones(self.max_len+1)*5 #don't want main process thinking we want to terminate. 
            dist.gather(tensor=padded_output,gather_list=None, dst=0,group=self.group) #send to process 2
            model_response = torch.ones(2*self.num_children + 1).double()
            dist.scatter(tensor=model_response,scatter_list=None,src=0,group=self.group)
            self.time_last_scatter = time.time()
        '''
        # print("simluations finished")
        # calc the move probabilities based on visit counts at the root node
        
        act_probs = F.softmax(torch.tensor(1.0/self.temperature * np.log(np.add(root._n_visits,1e-5))), dim=0)
        
        
        #print('assigning translation to translation')
        #self.translation = translation
        #print('End: rank: {}, output: {}, len: {}, lastWord: {}'.format(self.rankInGroup,[x for x in translation.output],translation.len_output,translation.last_word_id))
        
        
        
        return root._actions, act_probs,root

    
    def get_action(self,Translation translation, TreeNode root):
        
        cdef INT_t word_ind_in_acts
        
        end, bleu = translation.translation_end()
        # the pi vector returned by MCTS as in the alphaGo Zero paper

        if not end:
            
            #TO NOTE: Passing translation and root actually passes them as copies. 
            #MAKE SURE translation and root stay same (they do)
            #if not root._priors is None:   
            #    print('TransSTART: rank: {}, output: {}, len: {}, lastWord: {}'.format(self.rankInGroup,[x for x in translation.output],translation.len_output,translation.last_word_id))
            #    print('Root before: ind_parent: {}, priors[:10]: {}, _n_visits[:10]: {}'.format(root.parent_ind_action_taken, [x for x in root._priors[:10]],[x for x in root._n_visits[:10]]))

            acts, probs,root = self.get_move_probs(translation, root) # output vocab size
            
            #if not root._priors is None:
            #    print('TransEND: rank: {}, output: {}, len: {}, lastWord: {}'.format(self.rankInGroup,[x for x in translation.output],translation.len_output,translation.last_word_id))
            #    print('Root after: ind_parent: {}, priors[:10]: {}, _n_visits[:10]: {}'.format(root.parent_ind_action_taken, [x for x in root._priors[:10]],[x for x in root._n_visits[:10]]))
            
            sum_prob = np.sum(root._priors)
  
            word_ind_in_acts = np.random.choice(self.possible_inds, p=probs)
            word_id = acts[word_ind_in_acts] #chosen word
            
            #if not root._n_visits is None:
            #    print('rank: {}, chose word: {}, num visits: {}'.format(self.rankInGroup,word_id,root._n_visits[word_ind_in_acts]))
        
        
            #move root to this child
            #print('word_id chosen: ',word_id)
            
            #self._root = self._root._children[word_ind_in_acts]
            #self._root._parent = None
            root = root._children[word_ind_in_acts]
            root._parent = None
            #self._root = root
            
            
            #word = self.translation.word_index_to_str(word_id)  

            probs *= sum_prob #scale probs by sum of their priors

            # if self.is_train:
            #     move = np.random.choice(
            #         acts,
            #         # additional exploration by adding Dirichlet noise (self play)
            #         p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
            #     )
            #     self.mcts.update_with_move(move)
            # else:
            #     # with the default temp=1e-3, it is almost equivalent
            #     # to choosing the move with the highest prob
            #     word_id = np.random.choice(acts, p=probs)
            #     # reset the root node
            #     self.mcts.update_with_move(-1)
            #     word = translation.select_next_word(word_id)
            #     print("system chose word: {}".format(word))
            
            return word_id, probs, acts,root
            
        else:
            print("WARNING: no word left to select")


    def translate_sentence(self):
        """ start a translation using a MCTS, reuse the search tree,
            and store the translation data: (state, mcts_probs, z) for training
            state: list of current states (src, output) of tranlations
            mcts_probs: list of probs
            bleus_z: list of bleu scores
        """
       
        cdef Translation translation
        cdef TreeNode root
        translation = self.translation #WE DONT USE Self.TRANSLATION ANYMORE
        root = self._root #dont use self._root anymore either
    
        #print('Translating sentence rank: ',self.rankInGroup)
        output_states, mcts_probs, actions, bleu = [], [],[], -1
        while True:
            # start_time = time.time()
            trans_start_time = time.time()
            
            
            #MAKE SURE VISIT COUNTS CHANGE ON ROOT FROM HERE TO AFTER get_action
            #if not root._priors is None:   
            #    print('TransSTART: rank: {}, output: {}, len: {}, lastWord: {}'.format(self.rankInGroup,[x for x in translation.output],translation.len_output,translation.last_word_id))
            #    print('Root before: ind_parent: {}, priors[:10]: {}, _n_visits[:10]: {}'.format(root.parent_ind_action_taken, [x for x in root._priors[:10]],[x for x in root._n_visits[:10]]))
            
            word_id, probs, acts, root = self.get_action(translation,root)   
            
            # store the data: all we need to store is output since have src in main process
            curTrans = [x for x in translation.output[:translation.len_output]]
            output_states.append(curTrans)
            mcts_probs.append(probs.tolist())
            
            act_list = [x for x in acts]
            actions.append(act_list)
            # choose a word (perform a move)
            #print('from translate sentence')
            translation.do_move(word_id)
            
            #if not root._n_visits is None:
            #    print('maxlen: ',self.max_len)
            #    print('Got action: rank: {}, output: {}'.format(self.rankInGroup,[x for x in translation.output]))
            #    print('Got action: rank: {}, priors: {}'.format(self.rankInGroup, [x for x in root._priors[:15]]))

            
            end, bleu = translation.translation_end()
            #print('time for word: ',time.time()-trans_start_time)
            #print('CURRENT TRANSLATION Rank: {}: {}'.format(self.rankInGroup,curTrans))
            # print("sentence produced: {}".format(self.translation.vocab[self.translation.output].tolist()))
            # print("time: {:.3f}".format(time.time() - start_time))

            if end or translation.len_output == self.max_len:
                end, bleu = translation.translation_end(forceGetBleu=True)
            
                # reset MCTS root node
                # print("states collected: {}".format(states))
                # print("states len: {}".format(len(states)))
                # print("mcts_probs collected: {}".format(mcts_probs))
                # print("mcts_probs len: {}".format(len(mcts_probs)))
                #print('prediction: rank ')
                output2 = translation.output[:translation.len_output]
                #print([translation.vocab[output2[i]] for i in range(len(output2))])
                
                predict_tokens = [translation.vocab[x] for x in translation.output[:translation.len_output]]
           
                source_tokens = [translation.vocab[x] for x in translation.tgt]
                
                prediction = translation.fix_sentence(predict_tokens[1:-1])
                source = translation.fix_sentence(source_tokens[1:-1])
                #print("source rank: {}: {}".format(self.rankInGroup,source))
                #print("translation: {}".format(prediction))
                #print("bleu: {:.3f}".format(bleu))
                return bleu, output_states, mcts_probs,actions



