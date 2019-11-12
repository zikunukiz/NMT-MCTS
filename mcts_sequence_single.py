import numpy as np
import copy
import torch
import torch.nn as nn
from onmt.modules.copy_generator import collapse_copy_scores

BATCH_SIZE = 1

def softmax(x):
    probs = np.exp(x-np.max(x))
    probs = probs / np.sum(probs)
    return probs

class TreeNode(object):
    
    def __init__(self, parent, prev_word, parent_Q, prior_p, log_prob, word, step=-1, hidden=None, input_feed=None): 
        #prior_p: log prob, parent_Q: sum of the log prob of the parent
        
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self._log_prob = log_prob + self._parent._log_prob if self._parent is not None else log_prob
        self.word = word
        self._step = step
        self._hidden = hidden
        self._input_feed = input_feed
        
        #if parent:
        #    if word == parent.word: self._log_prob = torch.Tensor([-1e20]).cuda()
        #else:
        #    if word == prev_word: self._log_prob = torch.Tensor([-1e20]).cuda()
        
    
    def select(self, c_puct):
        #print('select values: ')
        #print([act_node.get_value(c_puct) for k, act_node in self._children.items()] )
        #print('max value: ', max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct)))
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))
    
    def expand(self, top_k_word, top_k_word_probs, hidden=None, input_feed=None):
        for word, prob in zip(*(top_k_word, top_k_word_probs)):
            #if word not in self._children.items(): 
            prior_p = (prob - top_k_word_probs.sum()) / (-top_k_word_probs.sum()*(top_k_word_probs.size(0)-1))
            self._children[word] = TreeNode(self, self.word, self._Q, prior_p, prob, word, self._step+1, hidden, input_feed)
                
    
    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += (leaf_value - self._Q) / self._n_visits
        
    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)
        #print(self.word, leaf_value, self._n_visits)
        #print('update: ', (leaf_value - self._Q) / self._n_visits)
                
    def get_value(self, c_puct):
        visit_penalty = 2
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits)/(1e-8+(self._n_visits)**(visit_penalty))
        #print('Q value:', self._Q, 'u value: ', self._u, 'visit: ', self._n_visits, 'parent visit: ', self._parent._n_visits, 'p value: ', self._P, 'exp P: ', torch.exp(self._P))
        return self._Q + self._u
    
    def is_leaf(self):
        return self._children == {}
    
    def is_root(slef):
        return self._parent is None
    

    
class MCTS(object):
    def __init__(self, bos, prev_word, memory_bank, memory_lengths, model, c_puct=5, n_playout=20, hidden=None, input_feed=None):
     
        self._root = TreeNode(None, None, 0, 1.0, 0, bos, -1, hidden, input_feed) #parent, prev_word, parent_Q, prior_p, log_prob, word, step=-1, hidden=None): 
        self._prev_word = prev_word
        self.beam_size = 1
        self.copy_attn = False
        self.model = model
        self.memory_bank = memory_bank
        self.memory_lengths = memory_lengths
        self._policy = model
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._word2hidden = dict() #record the hidden states from the previous word
        
        
        
    def _decode_and_generate(
            self,
            decoder_in,
            batch,
            src_vocabs,
            src_map=None,
            step=None,
            batch_offset=None):
        
        if self.copy_attn:
            # Turn any copied words into UNKs.
            decoder_in = decoder_in.masked_fill(
                decoder_in.gt(self._tgt_vocab_len - 1), self._tgt_unk_idx
            )

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn = self.model.decoder(
            decoder_in, self.memory_bank, memory_lengths=self.memory_lengths, step=step
        )

        # Generator forward.
        if not self.copy_attn:
            if "std" in dec_attn:
                attn = dec_attn["std"]
            else:
                attn = None
            log_probs = self.model.generator(dec_out.squeeze(0))
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        else:
            attn = dec_attn["copy"]
            scores = self.model.generator(dec_out.view(-1, dec_out.size(2)),
                                          attn.view(-1, attn.size(2)),
                                          src_map)
            # here we have scores [tgt_lenxbatch, vocab] or [beamxbatch, vocab]
            if batch_offset is None:
                scores = scores.view(-1, batch.batch_size, scores.size(-1))
                scores = scores.transpose(0, 1).contiguous()
            else:
                scores = scores.view(-1, self.beam_size, scores.size(-1))
            scores = collapse_copy_scores(
                scores,
                batch,
                self._tgt_vocab,
                src_vocabs,
                batch_dim=0,
                batch_offset=batch_offset
            )
            scores = scores.view(decoder_in.size(0), -1, scores.size(-1))
            log_probs = scores.squeeze(0).log()
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence'''
        
        
        return log_probs, None
    
    def _playout(self, state, i_playout):
        
        prev_word, decoder_in, batch, src_vocabs, src_map, step = state
        node = self._root
        
        #print('playout %d' %i_playout, node._children)
        word = None
        while not node.is_leaf():
            word, node = node.select(self._c_puct)
        leaf_value = node._log_prob/(node._step+2)
        #print('leaf_value: ', leaf_value)
        if word is not None:
            decoder_in = word.view(1, -1, 1)
        #print('decoder_in:, ', word, decoder_in, node.word)
        assert decoder_in[0, 0, 0] == node.word
        self.model.decoder.state['hidden'] = node._hidden#self._word2hidden[(int(word.cpu().item()), node._step)]
        self.model.decoder.state['input_feed']= node._input_feed
        word_probs, attn = self._decode_and_generate(decoder_in, batch, src_vocabs, step=step) #word_probs: (Batch, vocab) 
        word = torch.argmax(word_probs, dim=1)[0]
        #leaf_value += word_probs[0, word]
        
        top_k_word = torch.argsort(word_probs[0], descending=True)[:5] # only allow 5 words to expand from a parent node
        top_k_word_probs = word_probs[0][top_k_word]
        node.expand(top_k_word, top_k_word_probs, self.model.decoder.state['hidden'], self.model.decoder.state['input_feed']) 
        node.update_recursive(leaf_value)
    
    def get_sent_probs(self, state):
        for n in range(self._n_playout):
            self._playout(state, n)
            word_visits = [(word, node._n_visits) for word, node in self._root._children.items()]
            #print('n: %d' % n, word_visits)
            
            
        
        word_visits = [(word, node._n_visits, node) for word, node in self._root._children.items()]
        word_Qs = [(word, node._Q.cpu().item(), node) for word, node in self._root._children.items()]
        #print([act_node[1].get_value(c_puct)])
        #print((self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct)))
        #print('word_visits', word_visits)
        words, visits, nodes = zip(*word_visits)
        words, Qs, nodes = zip(*word_Qs)
        words_probs_visits = softmax(np.array(visits))
        words_probs_Qs = softmax(np.array(Qs))
        words_probs = words_probs_Qs # words_probs_visits # + 
        
        return words, words_probs, nodes
    
    def update_with_move(self, last_move, prev_word, step, hidden, input_feed):
        
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
        else:
            #self._prev_word = prev_word
            #if prev_word == 15:
            #    print(hidden)
            self._root = TreeNode(None, prev_word, 0, 1.0, 0, prev_word, step, hidden, input_feed) #parent, prev_word, parent_Q, prior_p, log_prob, word, step=-1, hidden=None): 
    
    def __str__(self):
        return "MCTS"

    
class MCTSDecoder(object):
    
    def __init__(self, bos, device, prev_word,memory_bank, memory_lengths, model, c_puct=5, n_playout=20):
        self.mcts = MCTS(bos, prev_word, memory_bank, memory_lengths, model, c_puct, n_playout, model.decoder.state['hidden'], model.decoder.state['input_feed'])
        self.bos = bos
        self.alive_seq = torch.full((1, 1), bos, dtype=torch.long, device=device)
    
    def reset_mcts(self):
        self.update_with_move(-1)
    
    def get_action(self, decoder_in,
            batch,
            src_vocabs,
            src_map=None,
            step=None,
            batch_offset=None, temp=0, return_prob=0):
        prev_word = self.alive_seq[:, -1].view(-1)
        words, words_probs, nodes = self.mcts.get_sent_probs((prev_word, decoder_in, batch, src_vocabs, src_map, step))
        
        
        word = words[np.argmax(words_probs)]
        node = nodes[np.argmax(words_probs)]
        self.mcts.update_with_move(-1, word, step, node._hidden, node._input_feed)
        return word
    
class TestModel(object):
    
    def __init__(self):
        self.decoder = nn.LSTM(input_size=300, hidden_size=500)
        self.generator = nn.Linear(500, 1000)
    

def main():
    # Decoder forward, takes [tgt_len, batch, nfeats] as input
    # and [src_len, batch, hidden] as memory_bank
    # in case of inference tgt_len = 1, batch = beam times batch_size
    # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
    batch_size = 256
    inp = torch.rand((1, BATCH_SIZE, 1))
    memory_bank = torch.rand((30, BATCH_SIZE, 500))
    memory_lengths = 30
    max_length = 10
    model = TestModel()
    mcts_decoder = MCTSDecoder(memory_bank, memory_lengths, model, c_puct=0, n_playout=3)
    
    predict_sent = []
    batch = None
    src_vocabs = None
    src_map = None
    for i in range(max_length):
        #print('time step: %d' % i)
        word = mcts_decoder.get_action(inp, batch, src_vocabs, src_map=src_map, step=i)
        inp = word
        predict_sent.append(word)
        #if inp == 10000: # END Token ID
        #    break
        #out = out.view(batch_size, beam_size, -1)
        #beam_attn = beam_attn.view(batch_size, beam_size, -1)
    
   # print(predict_sent)
        
        

if __name__ == '__main__':
    main()
    
        
            
            
            
            
            
            
            
        
    
    
    
    
    
    
        
    
    
    
    

