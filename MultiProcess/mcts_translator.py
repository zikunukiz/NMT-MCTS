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
from translate import *
import time

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.
    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0 # number of times this TreeNode has been visited
        # TO DO rescale by 0.95 (the aggregate prob from the 200 words)
        self._Q = 0 # the mean value of the next state
        self._u = 0
        self._V = None  #keep value at this node so can reuse if we end up here again and it's EOS or maxlen
        self._P = prior_p

    def expand(self,top_actions,top_probs):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """   
        assert(len(self._children)==0)
        for i in range(len(top_actions)):
            action = top_actions[i].clone().detach().item()
            prob = top_probs[i] 
            #print(action)
            assert(action not in self._children)
            self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
            It is a hyperparameter that controls exporation vs. exploitation
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return len(self._children)==0 

    def is_root(self):
        return self._parent is None



class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    #group is for the group of processes which allows us 
    #to send and receive tensors with other processes in group
    def __init__(self, tgt_tensor, group, rankInGroup, max_len,main_params):
        """
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self.group = group
        self.max_len = max_len #max translation length (need to path our tensors up to this for interprocess communication)
        self.rankInGroup =rankInGroup#is way of identifying this process in the group
        self._root = TreeNode(None, 1.0)
        self._c_puct = main_params.c_puct
        self._n_playout = main_params.num_sims
        self.num_children = main_params.num_children
        self.temperature = main_params.temperature 
        # with the default temp=1e-3, it is almost equivalent
        # to choosing the move with the highest prob
            
        self.translation = Translation(tgt=tgt_tensor, vocab=main_params.tgt_vocab_itos)
        self.is_training = main_params.is_training #lets us know if on train set            

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        playout_t1 = time.time()
        node = self._root
        while(1):
            if node.is_leaf() or state.output[-1] == globalsFile.EOS_WORD_ID:
                break

            action, node = node.select(self._c_puct)
            state.do_move(action)
            
        # print("output: {}".format(state.output.tolist()))

        
        if not node._V is None:
            leaf_value = node._V 

        else:
            # Check for end of translation 
            end, bleu = state.translation_end()

            if (not end or not self.is_training) and not ((end or len(state.output)==self.max_len)and self.is_training):

                #HERE is where we call gather with group containing model
                #want to send main process our output so far padded
                padded_output = torch.ones(self.max_len+1)*globalsFile.BLANK_WORD_ID
                padded_output[:len(self.translation.output)]= self.translation.output
                padded_output[-1] = len(self.translation.output)
                #print('Sending gatherer: ',padded_output[:15])
                dist.gather(tensor=padded_output,gather_list=None, dst=0,group=self.group) #send to process 2

                model_response = torch.ones(2*self.num_children + 1).double()
                dist.scatter(tensor=model_response,scatter_list=None,src=0,group=self.group)


                top_actions = model_response[:self.num_children].long()
                #print('Top actions received',top_actions[:15])
                top_probs = model_response[self.num_children:-1]
                leaf_value = model_response[-1]
                if not end and len(state.output)<self.max_len:
                    assert(len(top_actions)==self.num_children)
                    print('expanding at new state, rank: ',self.rankInGroup)
                    node.expand(top_actions,top_probs)

            else:
                #print('USING BLEU')
                leaf_value = bleu
        
        # Update value and visit count of nodes in this traversal
        node._V = leaf_value
        
        while(node._parent):
            node = node._parent
            node._n_visits += 1
            node._Q += (leaf_value - node._Q) / node._n_visits 

        
        playout_t2 = time.time()-playout_t1
        print('Time for playout: ',playout_t2)

    def get_move_probs(self):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current state (input and its translation so far)
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        #_n_playout is number of simulations per action chosen
        for n in range(self._n_playout):
            # print("simulation - {}".format(n))
            copy_last_word = copy.deepcopy(self.translation.last_word_id)
            copy_output = copy.deepcopy(self.translation.output)
            #copy_state = copy.deepcopy(self.translation)
            #self._playout(copy_state)
            
            self._playout(self.translation)
            self.translation.last_word_id = copy_last_word
            self.translation.output = copy_output
        
        # print("simluations finished")
        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/self.temperature * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    
    def get_action(self):
        end, bleu = self.translation.translation_end()
        # the pi vector returned by MCTS as in the alphaGo Zero paper
    
        if not end:
            acts, probs = self.get_move_probs() # output vocab size
            
            sum_prob = np.sum([node._P for node in self._root._children.values()])
            
            word_id = np.random.choice(acts, p=probs)
            #move root to this child
            #print('word_id chosen: ',word_id)
            
            self._root = self._root._children[word_id]
            self._root._parent = None

            word = self.translation.word_index_to_str(word_id)  

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
            
            return word_id, probs, acts
            
        else:
            print("WARNING: no word left to select")


    def translate_sentence(self):
        """ start a translation using a MCTS, reuse the search tree,
            and store the translation data: (state, mcts_probs, z) for training
            state: list of current states (src, output) of tranlations
            mcts_probs: list of probs
            bleus_z: list of bleu scores
        """
       
        print('Translating sentence rank: ',self.rankInGroup)
        output_states, mcts_probs, actions, bleu = [], [],[], -1
        while True:
            # start_time = time.time()
            trans_start_time = time.time()
            word_id, probs, acts = self.get_action()                                              
            # store the data: all we need to store is output since have src in main process
            output_states.append(self.translation.output.tolist())
            mcts_probs.append(probs.tolist())
            
            actions.append(list(acts))
            # choose a word (perform a move)
            #print('from translate sentence')
            self.translation.do_move(word_id)
            end, bleu = self.translation.translation_end()
            print('time for word: ',time.time()-trans_start_time)
            print('CURRENT TRANSLATION Rank: {}: {}'.format(self.rankInGroup,self.translation.output))
            # print("sentence produced: {}".format(self.translation.vocab[self.translation.output].tolist()))
            # print("time: {:.3f}".format(time.time() - start_time))

            if end or len(self.translation.output) == self.max_len:
                end, bleu = self.translation.translation_end(forceGetBleu=True)
            
                # reset MCTS root node
                # print("states collected: {}".format(states))
                # print("states len: {}".format(len(states)))
                # print("mcts_probs collected: {}".format(mcts_probs))
                # print("mcts_probs len: {}".format(len(mcts_probs)))
                print('prediction: ')
                output2 = self.translation.output
                print([self.translation.vocab[output2[i]] for i in range(len(output2))])
                predict_tokens = self.translation.vocab[self.translation.output].tolist()
                source_tokens = self.translation.vocab[self.translation.tgt].tolist()
                prediction = self.translation.fix_sentence(predict_tokens[1:-1])
                source = self.translation.fix_sentence(source_tokens[1:-1])
                print("source: {}".format(source))
                print("translation: {}".format(prediction))
                print("bleu: {:.3f}".format(bleu))
                return bleu, output_states, mcts_probs,actions



