# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes
Code Base: https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/mcts_alphaZero.py
@author: Jerry Zikun Chen
"""

import numpy as np
import copy

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
        self._Q = 0 # the mean value of the next state
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
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
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_fn, value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in the state (i.e. input sentence and previous 
            translated words) and outputs a list of (action, probability) tuples and also 
            a score in [0, 1] (i.e. the expected value of the end game (BLEU) score from the current
            player's perspective) for the current player. # no player for NMT
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_fn
        self._value = value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        action_probs = self._policy(state)
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action) # assume state has do_move function

        # Evaluate the leaf using a policy network which outputs a list of (action, probability) tuples p 
        # and a value network that gives a score v in [0, 1]
        # action_probs = self._policy(state)
        leaf_value = self._value(state)

        # Check for end of game. ## end of translation
        end, bleu = state.translation_end()
        if not end:
            node.expand(action_probs)
        else:
            ## for end state，return the "true" leaf_value (BLEU score)
            leaf_value = bleu

        # Update value and visit count of nodes in this traversal
        node.update_recursive(leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current state (input and its translation so far)
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

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

    def __str__(self):
        return "MCTS"


class MCTSTranslator(object):
    """AI translator based on MCTS"""

    def __init__(self, policy_fn, value_fn,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_fn, value_fn, c_puct, n_playout)

    def get_action(self, translation, temp=1e-3, return_prob=0):
        available_words = translation.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(translation.size)
        if len(available_words) > 0:
            acts, probs = self.mcts.get_move_probs(translation, temp)
            move_probs[list(acts)] = probs

            # with the default temp=1e-3, it is almost equivalent
            # to choosing the move with the highest prob
            move = np.random.choice(acts, p=probs)
            # reset the root node
            self.mcts.update_with_move(-1)
            word = translation.select_next_word(move)
            print("system chose: %d\n" % (word))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: no word left to select")


    def __str__(self):
        return "MCTS {}".format(self.player)