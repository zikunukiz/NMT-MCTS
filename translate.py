# -*- coding: utf-8 -*-
"""
@author: Jerry Zikun Chen
"""

from __future__ import print_function
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

global BOS_WORD, EOS_WORD, BLANK_WORD,NUM_WORD
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
NUM_WORD = '<num>'

# vocab is a list of tokenized strings
# vocab = TGT.vocab.itos
class Translation(object):
    def __init__(self, **kwargs, vocab):
        self.size = len(vocab) # 200 is the default value
        self.output = []

    def init_state(self):
        # TO DO: top 200 words or sample 200 words from the vocab based on their log probabilities
        self.availables = list(range(self.size))
        self.states = {}
        self.last_move = BOS_WORD

    def current_state(self): 
    # TO DO
    # current_states go to states -> state_batch to train policy
    # in order to get log_prob from state, we need dec_input, which comes from target (need tgt_mask?)

        return square_state[:, ::-1, :]

    def select_next_word(self, move):
        h = move // self.width
        w = move % self.width
        return [h, w]

    def do_move(self, move):
        # TO DO: select new available moves

        self.last_move = move
        self.output.append(move)

    def translation_end(self, tgt_tensor):
        """Check whether the translation is ended or not"""
        if last_move == EOS_WORD:
            bleu = sentence_bleu(reference, self.output)     
            return True, bleu
        else:
            return False, -1

    def get_bleu_scores(trg_tensor, pred_tensor, vocab):
        bleus_per_sentence = torch.zeros(trg_tensor.shape[1],requires_grad=False) 
        for col in range(trg_tensor.shape[1]): #each column contains sentence
        true_sentence = [vocab[i] for i in trg_tensor[:,col] if vocab[i] != BLANK_WORD][1:-1]
        pred_sentence = [vocab[i] for i in pred_tensor[:,col] if vocab[i] != BLANK_WORD]
        #now also need to stop pred_sentence after first EOS_WORD outputted
        #also don't want to use BOS chars
        ind_first_eos = 0
        for tok in pred_sentence:
          if tok == EOS_WORD:
            break
          ind_first_eos += 1
        if ind_first_eos != 0:
          pred_sentence = pred_sentence[1:ind_first_eos] #this gets rid of EOS_WORD
        #now undo some of the weird tokenization
        pred_sentence = fix_sentence(pred_sentence)
        true_sentence = fix_sentence(true_sentence)
        #This bleu_score defaults to calculating BLEU-4 (not normal bleu) so change weights,
        #this change of weights gives BLEU based on 1-grams so normal bleu I believe
        score = nltk.translate.bleu_score.sentence_bleu([true_sentence], pred_sentence, weights=(1, 0, 0, 0))
        score = score*len(true_sentence)
        bleus_per_sentence[col] = score
        return bleus_per_sentence

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
            elif '&' in p and ';' in p: #this means should be adding this onto last added word 
              if len(new_sentence) >0:
                new_sentence[-1] = new_sentence[-1] + "'"+p.split(';')[1]
              #OTHERWISE NOT SURE WHAT TO DO
              else:
                pass #NEED TO IMPLEMENT
            else:
              new_sentence.append(p)  
        return new_sentence


class Translate(object):
    """game server"""

    def __init__(self, translation, **kwargs):
        self.translation = translation

    def start_translate(self, translator, is_shown=1):
        """start translation"""
        translator = translator
        if is_shown:
            self.graphic(self.vocab)
        while True:
            move = translator.get_action(self.vocab)
            self.vocab.do_move(move)
            if is_shown:
                self.graphic(self.translation)
            end = self.translation.translation_end()
            if end:
                if is_shown:
                    print("Translation end. The translation is {}".format())
                return translation

    def start_train_translate(self, translator, is_shown=0, temp=1e-3): 
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.translation.init_state()
        states, mcts_probs, bleus_z = [], [], []
        while True:
            move, move_probs = translator.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.translation.current_state())
            mcts_probs.append(move_probs)
            # perform a move
            self.translation.do_move(move)
            end, bleu = self.translation.translation_end()
            if end:
                bleus_z.append(bleu)
                # reset MCTS root node
                player.reset_player()
                return bleu, zip(states, mcts_probs, bleus_z)
