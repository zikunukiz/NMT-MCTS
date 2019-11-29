# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules import TransformerEncoderLayer,TransformerEncoder
from torch.nn.modules import TransformerDecoder,TransformerDecoderLayer
from torch.nn.modules import Linear, Transformer
from torch.nn.init import xavier_uniform_
from torch.nn.modules.normalization import LayerNorm
from torch.autograd import Variable
import numpy as np
import math

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float64).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).double() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe.transpose(0, 1)[:x.size(0), :], requires_grad=False)
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout, activation, src_vocab_size, tgt_vocab_size):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(
            d_model=d_model, dropout=0.1)  # , max_len=100)
        encoder_layer = TransformerEncoderLayer(
                        d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm)

        self.d_model = d_model
        self.nhead = nhead
        self.linear = Linear(d_model, tgt_vocab_size)
        self.transformer = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                       dropout=dropout, activation=activation)
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initiate parameters in the transformer model.
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def change_to_value_net(self):
        self.linear = Linear(self.d_model, 1)  # now doing regression
        torch.nn.init.xavier_uniform_(
            self.linear.weight)  # initialize new weights

    '''
    src_key_padding_mask: mask out padded portion of src (is (N,S))
    tgt_mask: mask out future target words (I think usually just a square triangular matrix)
    tgt_key_padding_mask: mask out padded portion of tgt
    memory: (is the encoder output) in the case of testing or policy gradients,
              we reuse this output so want to give option to give it here
    '''

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, memory=None, only_return_last_col=False):
        # first embed src and tgt (here tgt )
        if memory is None:
            memory = self.encoder_embedding(
                src).double() * math.sqrt(self.d_model)
            memory = self.pos_encoder(memory).double()
            memory = self.encoder(
                memory, src_key_padding_mask=src_key_padding_mask)

        tgt2 = self.decoder_embedding(tgt).double() * math.sqrt(self.d_model)
        tgt2 = self.pos_encoder(tgt2)
        output = self.decoder(tgt2, memory, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        # linear layer increases embedding dimension to size of vocab (then can feed through softmax)
        if only_return_last_col:
            output = output[-1, :, :].unsqueeze(0)
        # T he cross entropy loss will take in the unnormalized outputs
        output = self.linear(output)
        return output, memory


class MainParams:
    def __init__(self, dropout, src_vocab_size, tgt_vocab_size, 
                batch_size,l2_const,c_puct,num_sims,init_temperature):
        use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_gpu else "cpu")
        print(self.device)
        self.model_params = dict(d_model=128, nhead=8, num_encoder_layers=4, num_decoder_layers=4,
                                 dim_feedforward=512, dropout=0.2, activation='relu', 
                                 src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)

        self.batch_size = batch_size
        self.l2_const = l2_const
        self.c_puct = 

class PolicyValueNet():
    """policy-value network"""

    def __init__(self, main_params, path_to_policy=None, path_to_value=None):
        self.use_gpu = True if main_params.device == torch.device('cuda:0') else False
        self.l2_const = main_params.l2_const  # coef of l2 penalty
        self.device = main_params.device
        self.encoder_output = None #keep this when get for this batch for first time
        self.num_children = main_params.num_children
        print('USING GPU: ',self.use_gpu)
        # the policy net
        policy_net = TransformerModel(**(main_params.model_params))
        # the value net
        value_net = TransformerModel(**(main_params.model_params))
        value_net.change_to_value_net()
        
        if self.use_gpu:
            self.policy_net = policy_net.to(main_params.device).double()
            self.value_net = value_net.to(main_params.device).double()
        else:
            self.policy_net = policy_net
            self.value_net = value_net

        # load parameters if available
        if path_to_policy and path_to_value:
            policy_params = torch.load(path_to_policy)
            self.policy_net.load_state_dict(policy_params)
            value_params = torch.load(path_to_value)
            self.value_net.load_state_dict(value_params)

        # optimizer
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), weight_decay=self.l2_const)
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), weight_decay=self.l2_const)


    def forward(self, src_tensor, dec_input, sentence_lens,req_grad=False):
        """
        sentence_lens: is length of each sentence in dec_input when no padding,
                        these are the indices we'll be trying to get from dec_output.
                        Actually subtract 1 from these. 

        """
        if req_grad:
            self.policy_net.train()
            self.value_net.train()
        else:
            self.policy_net.eval()
            self.value_net.eval()

        src_key_padding_mask = (src_tensor == 1).transpose(0, 1)
        dec_key_padding_mask = (dec_input==1).transpose(0,1)
        batch_indices = np.arange(src_tensor.shape[1])
        with torch.set_grad_enabled(req_grad):
            policy_output, self.encoder_output = self.policy_net.forward(src_tensor, dec_input,
                                             src_key_padding_mask=src_key_padding_mask,
                                             tgt_mask=None, tgt_key_padding_mask=dec_key_padding_mask,
                                             memory_key_padding_mask=src_key_padding_mask,
                                             memory=self.encoder_output)
            
            log_act_probs = F.log_softmax(policy_output[sentence_lens-1, batch_indices, :], dim=1)
            print('SHAPE log_act_probs: ',log_act_probs.shape)

            #commenting out shared decoder_embedding for now since were 
            #trained with different ones which will throw off the algorithm initially
            #if now use the same
            #self.value_net.decoder_embedding.weight = nn.Parameter(
            #   self.policy_net.decoder_embedding.weight.clone())
            
            value_output, encoder_output = self.value_net.forward(src_tensor, dec_input,
                                          src_key_padding_mask=src_key_padding_mask,
                                          tgt_mask=None, tgt_key_padding_mask=None,
                                          memory_key_padding_mask=src_key_padding_mask,
                                          memory=encoder_output)

            value_output = torch.sigmoid(value_output[sentence_lens-1, batch_indices, 0])
            return log_act_probs, value

    
    def train_step(self, source, translation, mcts_probs, bleu_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            source = Variable(
                torch.from_numpy(source.reshape(-1, 1))).to(self.device)
            translation = Variable(
                torch.from_numpy(translation.reshape(-1, 1))).to(self.device)  
            mcts_probs = Variable(
                torch.from_numpy(mcts_probs)).to(self.device)
            bleu_batch = Variable(
                torch.from_numpy(bleu_batch)).to(self.device)

        # zero the parameter gradients
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.policy_optimizer, lr)
        set_learning_rate(self.value_optimizer, lr)

        # forward pass
        log_act_probs, value, encoder_output = self.policy_value_train(source, translation, req_grad=True)

        # if self.use_gpu:
        #     log_act_probs = Variable(
        #         torch.from_numpy(log_act_probs)).to(self.device)
        #     value = Variable(torch.from_numpy(value)).to(self.device)
        
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), bleu_batch)
        # TODO support batch dimensions (right now only 1 data point)

        # print(mcts_probs.shape, log_act_probs.shape)

        policy_loss = - torch.dot(mcts_probs.view(-1), log_act_probs.view(-1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = - torch.dot(torch.exp(log_act_probs.view(-1)), log_act_probs.view(-1))
        return loss.item(), entropy.item()

    def get_param(self):
        policy_net_params = self.policy_net.state_dict()
        value_net_params = self.value_net.state_dict()
        return policy_net_params, value_net_params

    def save_model(self, policy_model_file, value_model_file):
        """ save model params to file """
        policy_net_params, value_net_params = self.get_param()  # get model params
        torch.save(policy_net_params, policy_model_file)
        torch.save(value_net_params, value_model_file)
