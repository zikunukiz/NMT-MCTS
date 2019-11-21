# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0
@author: Junxiao Song
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers,
                dim_feedforward, dropout, activation, src_vocab_size, tgt_vocab_size):
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

        self._reset_parameters()


    def _reset_parameters(self):
        #Initiate parameters in the transformer model.
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
    
    def change_to_value_net(self):
        self.linear = Linear(self.d_model, 1) #now doing regression
        torch.nn.init.xavier_uniform(self.linear.weight) #initialize new weights
        
    
    '''
    src_key_padding_mask: mask out padded portion of src (is (N,S))
    tgt_mask: mask out future target words (I think usually just a square triangular matrix)
    tgt_key_padding_mask: mask out padded portion of tgt
    memory: (is the encoder output) in the case of testing or policy gradients, 
              we reuse this output so want to give option to give it here
    '''
    def forward(self, src, tgt, src_key_padding_mask=None,tgt_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None,memory=None,only_return_last_col=False):
        #first embed src and tgt (here tgt )
        if memory is None:
          memory = self.encoder_embedding(src).double() * math.sqrt(self.d_model)
          memory = self.pos_encoder(memory).double()
          memory = self.encoder(memory, src_key_padding_mask=src_key_padding_mask)
        
        tgt2 = self.decoder_embedding(tgt).double() * math.sqrt(self.d_model)
        tgt2 = self.pos_encoder(tgt2)
        output = self.decoder(tgt2, memory, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        #linear layer increases embedding dimension to size of vocab (then can feed through softmax)
        if only_return_last_col:
          output = output[-1,:,:].unsqueeze(0)
        output = self.linear(output)  #T he cross entropy loss will take in the unnormalized outputs 
        return output, memory

class MainParams:
    def __init__(self, dropout, src_vocab_size, tgt_vocab_size, batch_size):
        use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_gpu else "cpu")
        print(self.device)
        self.model_params = dict(d_model=128,nhead=8,num_encoder_layers=4, num_decoder_layers=4,
                dim_feedforward=512, dropout=0.2,activation='relu',src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size)

        self.batch_size = batch_size
        self.num_decode_steps = 60 


class PolicyNet():
    """policy-value network """
    def __init__(self, main_params, path_to_policy=None):
        self.use_gpu = True if main_params.device == 'cuda:0' else False
        self.l2_const = 1e-4  # coef of l2 penalty
        
        # the policy value net module
        if self.use_gpu:
            self.policy_net = TransformerModel(**(main_params.model_params)).to(main_params.device)
        else:
            self.policy_net = TransformerModel(**(main_params.model_params))
        self.optimizer = optim.Adam(self.policy_net.parameters(), weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(path_to_policy)
            self.policy_net.load_state_dict(net_params)

    def get_log_act_prob(src_tensor, dec_input, src_key_padding_mask, src_key_padding_mask, encoder_output):
        output, _ = self.policy.forward(src_tensor, dec_input, src_key_padding_mask=src_key_padding_mask, 
                                                    tgt_mask=None, tgt_key_padding_mask=None, 
                                                    memory_key_padding_mask=src_key_padding_mask, memory=encoder_output)
        log_act_probs = F.log_softmax(output[-1,:,:], dim=1)
        return log_act_probs

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs = self.policy_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs = self.policy_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs

    def policy_fn(self, translation): 
        """
        input: translation
        output: a list of (action, probability) tuples for each available
        action and the score of the translation state
        """
        # TO DO
        legal_positions = translation.availables
        current_state = np.ascontiguousarray(translation.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        current_state = 
        if self.use_gpu:
            log_act_probs = self.policy_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        # else:
        #     log_act_probs = self.policy_value_net(
        #             Variable(torch.from_numpy(current_state)).float())
        #     act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs = self.policy_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        # backward and optimize
        policy_loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return policy_loss.data[0], entropy.data[0]
        #for pytorch version >= 0.5 please use the following line instead.
        #return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)