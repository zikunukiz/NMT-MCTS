from torch.nn.modules import TransformerEncoderLayer,TransformerEncoder,TransformerDecoder,TransformerDecoderLayer
import math
import torch
from torch import nn
from torch.nn.modules import Transformer
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import xavier_uniform_
from torch.nn.modules import Linear
from torch.autograd import Variable
import torch.nn.functional as F

#Slightly extend the transformer model to embed encoder and decoder initial inputs
#and use positional encodings as well 
#some code here copied from https://pytorch.org/docs/master/_modules/torch/nn/modules/transformer.html#Transformer
class TransformerModel(nn.Module):

    def __init__(self,d_model,nhead,num_encoder_layers,num_decoder_layers,
                dim_feedforward,dropout,activation,src_vocab_size,tgt_vocab_size):
        super(TransformerModel,self).__init__()
        self.pos_encoder = PositionalEncoding(d_model=d_model,dropout=0.1) #, max_len=100)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        self.d_model = d_model
        self.nhead = nhead
        self.linear = Linear(d_model,tgt_vocab_size)
        self.transformer = Transformer(d_model=d_model,nhead=nhead,num_encoder_layers=num_encoder_layers,
                                        num_decoder_layers=num_decoder_layers,dim_feedforward=dim_feedforward,
                                        dropout=dropout,activation=activation)
        
        self.encoder_embedding = nn.Embedding(src_vocab_size,d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size,d_model)

        self._reset_parameters() #initialize all the parameters randomly


    def _reset_parameters(self):
        #Initiate parameters in the transformer model.
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
    
    def change_to_value_net(self):
        self.linear = Linear(self.d_model, 1) #now doing regression
        torch.nn.init.xavier_uniform(self.linear.weight) #initialize new weights
        
    
    '''
    args:
    src_key_padding_mask: mask out padded portion of src (is (N,S))
    tgt_mask: mask out future target words (I think usually just a square triangular matrix)
    tgt_key_padding_mask: mask out padded portion of tgt
    memory: (is the encoder output) in the case of testing or policy gradients, 
              we reuse this output so want to give option to give it here
    '''
    def forward(self, src, tgt, src_key_padding_mask=None,tgt_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None,memory=None,only_return_last_col=False):
        
        '''
        First embed src and tgt, then add positional encoding, then run encoder,
        then decoder.
        '''
        if memory is None:
            #here we reuse encoder output from previous decoding step
            memory = self.encoder_embedding(src).double() * math.sqrt(self.d_model)
            memory = self.pos_encoder(memory).double()
            memory = self.encoder(memory,src_key_padding_mask=src_key_padding_mask)

        tgt2 = self.decoder_embedding(tgt).double() * math.sqrt(self.d_model)
        tgt2 = self.pos_encoder(tgt2)

        output = self.decoder(tgt2, memory, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        #linear layer increases embedding dimension to size of vocab (then can feed through softmax)
        #If value_net, then linear layer will reduce embedding dim to size 1 since just regression
        output = self.linear(output)  #The cross entropy loss will take in the unnormalized outputs 
        return output,memory
        

def generate_square_subsequent_mask(sz):
  #Generate a square mask for the sequence. The masked positions are filled with float('-inf').
  #Unmasked positions are filled with float(0.0).
  mask = (torch.triu(torch.ones(sz, sz)) == 1).float().transpose(0, 1)
  mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
  return mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len,dtype=torch.float64).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).double() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x): 
        x = x + Variable(self.pe.transpose(0,1)[:x.size(0), :],requires_grad=False)
        return self.dropout(x)


#now our optimizer (uses adam but changes learning rate over time)
#is in paper and http://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0 
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
  


def get_std_opt(model):
    return NoamOpt(model.d_model, 1, 4000,
            torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-9))
    


#taken from https://nlp.seas.harvard.edu/2018/04/03/attention.html#label-smoothing
#CURRENTLY NOT USING THIS
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    #size is output vocab size

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum') #want to then divide by #tokens. 
        self.criterion2 = nn.KLDivLoss(reduction='none')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    #x is log(probs of predictions) and target are indices of true values 
    def forward(self, x, target):
        #(output of decoder has dim [T, vocab_size, batch_size] so this will be same and target will be (T,batch_size)
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2)) #just fills matrix with self.smoothing / (self.size - 2)
        #print('true dist shape: {}, unsqueezed shape: {}'.format(true_dist.shape,target.data.unsqueeze(1).shape))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) #places confidence in correct locations (so if confidence=0.9 then true indice has val 0.9
        true_dist[:, self.padding_idx] = 0 #Makes prob of getting blank space 0
        #mask = torch.nonzero(target.data == self.padding_idx) #gets indices of where target is padding
        
        #NEED TO CREATE MASK THAT MULTIPLIES
        mask = (~(target.data == self.padding_idx)).unsqueeze(1) # is now (T,1,N)
        #now mask is (T,N) now insert middle dimension then multiply
        true_dist = mask * true_dist
        
        self.true_dist = true_dist
        #want to return both the sum of kl div over all tokens and keep track of kl per token
                                                   
        return self.criterion(x, Variable(true_dist, requires_grad=False)), mask.sum()




