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
        self.vocab = vocab
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
        self.encoder_output = encoder_output # store for later tranlstion output
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
        word_probs = np.exp(log_word_probs)
        top_ids = np.argpartition(word_probs, -self.n_avlb)[-self.n_avlb:]
        next_id = np.argpartition(word_probs, -1)[-1:]
        self.availables = top_ids
        self.output = np.append(self.output, next_id)

    def translation_end(self):
        """Check whether the translation is ended or not"""
        prediction = self.output.tolist()
        reference = self.tgt.tolist()
        if self.output[-1] == EOS_WORD_ID:
            bleu = sacrebleu.sentence_bleu(
                reference, prediction, smooth_method='exp')
            return True, bleu # TO DO return value output if end
        else:
            return False, -1

    # def get_bleu_scores(trg_tensor, pred_tensor, vocab):
    #     bleus_per_sentence = torch.zeros(trg_tensor.shape[1],requires_grad=False)
    #     for col in range(trg_tensor.shape[1]): #each column contains sentence
           #      true_sentence = [vocab[i] for i in trg_tensor[:,col] if vocab[i] != BLANK_WORD][1:-1]
           #      pred_sentence = [vocab[i] for i in pred_tensor[:,col] if vocab[i] != BLANK_WORD]
           #      #now also need to stop pred_sentence after first EOS_WORD outputted
           #      #also don't want to use BOS chars
           #      ind_first_eos = 0
           #      for tok in pred_sentence:
           #        if tok == EOS_WORD:
           #          break
           #        ind_first_eos += 1
           #      if ind_first_eos != 0:
           #        pred_sentence = pred_sentence[1:ind_first_eos] #this gets rid of EOS_WORD
           #      #now undo some of the weird tokenization
           #      pred_sentence = fix_sentence(pred_sentence)
           #      true_sentence = fix_sentence(true_sentence)

           #      # sacrebleu to account for sentence length
           #      score = sacrebleu.sentence_bleu(pred_sentence, true_sentence, smooth_method='exp').score
           #      bleus_per_sentence[col] = score/100.0
    #     return bleus_per_sentence

    def fix_sentence(sentence):
        """get original sentences from tokenizations"""
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

    def start_translate(self, translator, is_shown=1):
        """start translation"""
        translator = translator
        if is_shown:
            self.graphic(self.vocab)
        while True:
            word_id = translator.get_action(self.vocab)
            self.vocab.do_move(word_id)
            if is_shown:
                self.graphic(self.translation)
            end = self.translation.translation_end()
            if end:
                if is_shown:
                    print("Translation end. The translation is {}".format())
                return translation

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
