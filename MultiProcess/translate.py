# -*- coding: utf-8 -*-


from __future__ import print_function
import numpy as np
from torch.autograd import Variable
import torch
import sacrebleu
import time
import nltk
import globalsFile


#holds current state of our MCTS translation
class Translation(object):
    def __init__(self, tgt, vocab, **kwargs,):
        
        self.vocab = np.array(vocab)
        self.tgt = np.array(tgt)  # list of indices
        self.output = np.array([2]) # index in the vocab
        self.last_word_id = globalsFile.BOS_WORD_ID

    def word_index_to_str(self, word_id):
        return self.vocab[word_id]

    def do_move(self, word_id):  # can change name to choose_word
        # word_id is the index in the vocab
        self.last_word_id = word_id
        self.output = np.append(self.output, word_id)

    def translation_end(self,forceGetBleu=False):
        """Check whether the translation is ended or not"""
        if self.last_word_id == globalsFile.EOS_WORD_ID or forceGetBleu:
            # revert tokenization back to strings
            predict_tokens = self.vocab[self.output].tolist()
            ref_tokens = self.vocab[self.tgt].tolist()

            #now assuming first token of each is BOS and last is EOS
            #and don't want BOS or EOS in our bleu calc
            predict_tokens = predict_tokens[1:-1]
            ref_tokens = ref_tokens[1:-1]

            bleu = None
            if globalsFile.USEBLEU1:
                prediction_list = self.fix_sentence(predict_tokens)
                reference_list = self.fix_sentence(ref_tokens)
                bleu = nltk.translate.bleu_score.sentence_bleu([reference_list], prediction_list, weights=(1, 0, 0, 0))
    
            else:
                pred_sentence = self.fix_sentence(predict_tokens,as_str=True)
                ref_sentence = self.fix_sentence(ref_tokens,as_str=True)
                bleu = sacrebleu.corpus_bleu([prediction], [[reference]], smooth_method='exp').score / 100
            
            # print("reference: {}".format(reference))
            # print("prediction: {}".format(prediction))
            # compute sacre BLEU score adjusted for length
            
            # print("BLEU: {}".format(bleu))
            return True, bleu # TO DO return value output if end
        else:
            return False, -1

    
    def fix_sentence(self, sentence, as_str=False):  
        #if as_str==True then join tokens by space unless comma or period
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
        
        str_to_ret= ''
        if as_str:
          for w in new_sentence:

            if w in [',','.','?'] and len(str_to_ret)!=0 and str_to_ret[-1]==' ': #remove last space added
              
              str_to_ret = str_to_ret[:-1] 
            elif  w in [',','.','?'] and len(str_to_ret)!=0:
              #print('Weird sentence: ', new_sentence)
              pass

            str_to_ret += w
            if not (w in ['.','?']):
              str_to_ret += ' '
          
          return str_to_ret
        
        return new_sentence



