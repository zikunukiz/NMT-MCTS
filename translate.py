# -*- coding: utf-8 -*-
"""
@author: Jerry Zikun Chen
"""

from __future__ import print_function
import numpy as np
from torch.autograd import Variable
import torch
import sacrebleu

global BOS_WORD_ID, EOS_WORD_ID
BOS_WORD_ID = 2
EOS_WORD_ID = 3


class Translation(object):
    def __init__(self, src, tgt, n_avlb, vocab, policy_value_fn, **kwargs,):
        """
        n_avlb: number of available word to choose at each step
        src: list of indices (SRC vocab) representing source sentence 
        vocab: target vocabulary (TGT.vocab.itos)
        """
        self.n_avlb = n_avlb
        self.vocab = np.array(vocab)
        self.src = np.array(src)  # list of indices
        self.tgt = np.array(tgt)  # list of indices
        self.output = np.array([2]) # index in the vocab
        self.policy_value_net = policy_value_fn
        self.device = policy_value_fn.device

    def init_state(self):
        # TO DO: policy_value input should be the current decoding state
        # pick highest 200 probability words
        # assume word_probs numpy array
        if self.policy_value_net.use_gpu == True:
            src_tensor = Variable(torch.from_numpy(
                np.array(self.src).reshape(-1, 1))).to(self.policy_value_net.device)
            dec_input = Variable(torch.from_numpy(
                np.array(self.output).reshape(-1, 1))).to(self.policy_value_net.device)
        else:
            src_tensor = Variable(torch.from_numpy(
                np.array(self.src).reshape(-1, 1)))
            dec_input = Variable(torch.from_numpy(
                np.array(self.output).reshape(-1, 1)))

        log_word_probs, value, encoder_output = self.policy_value_net.policy_value(src_tensor, dec_input)
        word_probs = np.exp(log_word_probs)
        top_ids = np.argpartition(word_probs, -self.n_avlb)[-self.n_avlb:]
        self.encoder_output = encoder_output.clone().detach() # store for later tranlstion output
        self.availables = top_ids
        self.last_word_id = BOS_WORD_ID

    def current_state(self):
        # Q There are two outputs for now, does it fit with data buffer?
        return self.src, self.output

    def select_next_word(self, word_id):
        return self.vocab[word_id]

    def do_move(self, word_id):  # can change name to choose_word
        # word_id is the index in the vocab
        # TO DO: normalize other probabilities?
        # add word as new input
        if self.policy_value_net.use_gpu == True:
            src_tensor = Variable(torch.from_numpy(
                    np.array(self.src).reshape(-1, 1))).to(self.policy_value_net.device)
            dec_input = Variable(torch.from_numpy(
                    np.array(self.output).reshape(-1, 1))).to(self.policy_value_net.device)
        else:
            src_tensor = Variable(torch.from_numpy(
                    np.array(self.src).reshape(-1, 1)))
            dec_input = Variable(torch.from_numpy(
                    np.array(self.output).reshape(-1, 1)))
        # encoder_output.requires_grad == True -> cannot create deepcopy?
        # should encoder_ouput be stored as numpy like src and output??
        log_word_probs, value, encoder_output = self.policy_value_net.policy_value(
                src_tensor, dec_input, self.encoder_output) # reusing encoder_output
        word_probs = np.exp(log_word_probs)[0]
        top_ids = np.argpartition(word_probs, -self.n_avlb)[-self.n_avlb:]
        next_id = np.argpartition(word_probs, -1)[-1:]
        self.availables = top_ids
        self.output = np.append(self.output, next_id)

    def translation_end(self):
        """Check whether the translation is ended or not"""
        if self.output[-1] == EOS_WORD_ID:
            prediction = self.fix_sentence(self.vocab[self.output].tolist()[0])
            reference = self.fix_sentence(self.vocab[self.tgt].tolist()[0])
            print("reference: {}".format(reference))
            print("prediction: {}".format(prediction))
            bleu = sacrebleu.corpus_bleu([prediction], [[reference]], smooth_method='exp')
            return True, bleu # TO DO return value output if end
        else:
            return False, -1

    def fix_sentence(self, sentence):
        """get original sentences from tokenizations"""
        sentence = sentence[1:-2]
        new_sentence = []
        cur_word = ''
        for p in sentence:
            if '@@' in p:
                cur_word += p[:-2]
            else:
                if cur_word != '':
                    new_sentence.append(cur_word+p)
                    cur_word = ''
                elif '&' in p and ';' in p:  # this means should be adding this onto last added word
                    if len(new_sentence) > 0:
                        new_sentence[-1] = new_sentence[-1] + \
                            "'"+p.split(';')[1]
                    # OTHERWISE NOT SURE WHAT TO DO
                    else:
                        pass  # NEED TO IMPLEMENT
                else:
                    new_sentence.append(p)
        return new_sentence

class Translate(object):
    """game server"""

    def __init__(self, translation, **kwargs):
        self.translation = translation

    def graphic(translation):
        pass

    def start_train_translate(self, translator, is_shown=0, temp=1e-3):
        """ start a translation using a MCTS player, reuse the search tree,
            and store the translation data: (state, mcts_probs, z) for training
            state: list of current states (src, output) of tranlations
            mcts_probs: list of probs
            bleus_z: list of bleu scores
        """
        self.translation.init_state()
        states, mcts_probs, bleus_z = [], [], []
        while True:
            word_id, word_probs = translator.get_action(self.translation,
                                                        temp=temp,
                                                        return_prob=1)
            # store the data
            # a tuple of (src, output)
            states.append((self.translation.current_state()))
            mcts_probs.append(word_probs)
            # choose a word (perform a move)
            self.translation.do_move(word_id)
            end, bleu = self.translation.translation_end()
            if end:
                bleus_z.append(bleu)
                # reset MCTS root node
                player.reset_player()
                return bleu, zip(states, mcts_probs, bleus_z)
