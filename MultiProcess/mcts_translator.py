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
        self._P = prior_p

    def expand(self,top_actions,top_probs):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        assert(len(top_actions)==self.num_children)
        for i in range(self.num_children):
            action = top_actions[i]
            prob = top_probs[i] 
            assert(action not in self._children)
            self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1

        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits 

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

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
        self.group
        self.max_len = max_len #max translation length (need to path our tensors up to this for interprocess communication)
        self.rankInGroup #is way of identifying this process in the group
        self._root = TreeNode(None, 1.0)
        self._c_puct = main_params.c_puct
        self._n_playout = main_params.num_sims
        self.num_children = main_params.num_children
        self.temperature = main_params.temperature 
        # with the default temp=1e-3, it is almost equivalent
        # to choosing the move with the highest prob
            
        self.translation = Translation(tgt=tgt_tensor, vocab=main_params.tgt_vocab_stoi)
        self.is_training = main_params.is_training #lets us know if on train set            

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if node.is_leaf() or state.output[-1] == globalsFile.EOS_WORD_ID:
                break

            action, node = node.select(self._c_puct)
            state.do_move(action)
            
        # print("output: {}".format(state.output.tolist()))


        # Check for end of translation 
        end, bleu = state.translation_end()
        if not end or not self.is_training:
            
            #HERE is where we call gather with group containing model
            #want to send main process our output so far padded
            padded_output = np.zeros(max_len+1)*globalsFile.BLANK_WORD_ID
            padded_output[:len(self.translation.output)]=self.translation.output
            padded_output[-1] = len(self.translation.output)
            dist.gather(tensor=padded_output,gather_list=None, dst=0,group=self.group) #send to process 2

            dist.scatter(tensor=model_response,scatter_list=None,src=0,group=self.group)
            top_actions = model_response[:self.num_children]
            top_probs = model_response[self.num_children:-1]
            leaf_value = model_response[-1]
            if not end and len(state.output)<self.max_len:
                node.expand(top_actions,top_probs)
           
        else:
            leaf_value = bleu
        
        # Update value and visit count of nodes in this traversal
        node.update_recursive(leaf_value)


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

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def get_action(self, return_prob=0):
        end, bleu = self.translation.translation_end()
        # the pi vector returned by MCTS as in the alphaGo Zero paper
    
        word_probs = np.zeros(len(self.translation.vocab))
        if not end:
            acts, probs = self.get_move_probs() # output vocab size
            word_probs[list(acts)] = probs

            sum_prob = np.sum([node._P for node in self._root._children.values()])
            
            word_id = np.random.choice(acts, p=probs)
            self.update_with_move(word_id) #move root to this child
            word = translation.word_index_to_str(word_id)  

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
            
            
            if return_prob:
                return word_id, word_probs
            else:
                return word
        else:
            print("WARNING: no word left to select")


    def translate_sentence(self):
        """ start a translation using a MCTS, reuse the search tree,
            and store the translation data: (state, mcts_probs, z) for training
            state: list of current states (src, output) of tranlations
            mcts_probs: list of probs
            bleus_z: list of bleu scores
        """
       
        print('New MCTS Simulations ...')
        output_states, mcts_probs, bleu = [], [], -1
        while True:
            # 55 seconds per loop (100 simulations)
            # start_time = time.time()
            word_id, word_probs = self.get_action(return_prob=1)                                              
            # store the data: all we need to store is output since have src in main process
            output_states.append(self.translation.output)
            mcts_probs.append(word_probs)
            # choose a word (perform a move)
            self.translation.do_move(word_id)
            end, bleu = self.translation.translation_end()
            
            # print("sentence produced: {}".format(self.translation.vocab[self.translation.output].tolist()))
            # print("time: {:.3f}".format(time.time() - start_time))

            if end or len(self.translation.output) == self.max_len:
                end, bleu = self.translation.translation_end(forceGetBleu=True)
            
                # reset MCTS root node
                # print("states collected: {}".format(states))
                # print("states len: {}".format(len(states)))
                # print("mcts_probs collected: {}".format(mcts_probs))
                # print("mcts_probs len: {}".format(len(mcts_probs)))
                predict_tokens = self.translation.vocab[self.translation.output].tolist()
                source_tokens = self.translation.vocab[self.translation.tgt].tolist()
                prediction = self.translation.fix_sentence(predict_tokens[1:-1])
                source = self.translation.fix_sentence(source_tokens[1:-1])
                print("source: {}".format(source))
                print("translation: {}".format(prediction))
                print("bleu: {:.3f}".format(bleu))
                return bleu, output_states, mcts_probs




    def __str__(self):
        return "MCTS"

