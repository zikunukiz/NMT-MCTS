# -*- coding: utf-8 -*-
"""
@author: Jerry Zikun Chen
"""

from __future__ import print_function
import numpy as np
import sacrebleu

global BOS_WORD, EOS_WORD, BLANK_WORD
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"

class Translation(object):
    def __init__(self, src, n_avlb, vocab, policy_fn, **kwargs,):
    	"""
		n_avlb: number of available word to choose at each step
    	src: list of indices (SRC vocab) representing source sentence 
    	vocab: target vocabulary (TGT.vocab.itos)
    	"""
        self.size = n_avlb
        self.src = src # list of indices
        self.output = []
        self.policy_net = policy_fn

    def init_state(self):
        # TO DO: policy_value input should be the current decoding state
        # pick highest 200 probability words
        # assume word_probs numpy array
        word_probs = self.policy_net.policy_value()
        top_ids = np.argpartition(word_probs, -n_avlb)[-n_avlb:]
        self.availables = top_ids
        self.last_word_id = BOS_WORD

    def current_state(self): 
    # TO DO:
    # current_states go to states -> state_batch to train policy
    # in order to get log_prob from state, we need dec_input, which comes from target (need tgt_mask?)
        return self.src, self.output

    def select_next_word(self, word_id):
        return vocab[word_id]

    def do_move(self, word_id): # can change name to choose_word
        # word_id is the index in the vocab
        # TO DO: normalize other probabilities
        # add word as new input 
        word_probs = self.policy_net.policy_value()
        top_ind = np.argpartition(word_probs, -n_avlb)[-n_avlb:]
        self.availables = vocab[top_ind]
        self.last_word_id = word_id
        self.output.append(word_id)

    def translation_end(self, tgt_tensor):
        """Check whether the translation is ended or not"""
        if vocab[last_word_id] == EOS_WORD:
            bleu = sacrebleu.sentence_bleu(reference, self.output, smooth_method='exp')     
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
	        
	        # sacrebleu to account for sentence length
	        score = sacrebleu.sentence_bleu(pred_sentence, true_sentence, smooth_method='exp').score
	        bleus_per_sentence[col] = score/100.0
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
        """ start a self-play game using a MCTS player, reuse the search tree,
        	and store the self-play data: (state, mcts_probs, z) for training
        """
        self.translation.init_state()
        states, mcts_probs, bleus_z = [], [], []
        while True:
            word_id, word_probs = translator.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.translation.current_state())
            mcts_probs.append(word_probs)
            # choose a word (perform a move)
            self.translation.do_move(word_id)
            end, bleu = self.translation.translation_end()
            if end:
                bleus_z.append(bleu)
                # reset MCTS root node
                player.reset_player()
                return bleu, zip(states, mcts_probs, bleus_z)
