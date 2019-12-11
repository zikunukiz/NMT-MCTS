# -*- coding: utf-8 -*-

#creates the iterators over our dataset

from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext import data

global BOS_WORD, EOS_WORD, BLANK_WORD
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"

def createIterators(batch_size, data_path):
    tokenize = lambda x: x.split(' ') #already have tokenized files so now just split

    SRC = Field(sequential=True, tokenize=tokenize, init_token = BOS_WORD, 
                                        eos_token = EOS_WORD, pad_token=BLANK_WORD)
    TGT = Field(sequential=True, tokenize=tokenize, init_token = BOS_WORD, 
                                        eos_token = EOS_WORD, pad_token=BLANK_WORD)

    train, valid, test = TabularDataset.splits(
               path=data_path, # the root directory where the data lies
               train='combined_train.tsv',
               validation='combined_valid.tsv',
               test = 'combined_test.tsv',
               fields=[('de', SRC),('en', TGT)],
               format='TSV',
               skip_header=False) 
 

    SRC.build_vocab(train.de)
    TGT.build_vocab(train.en)

    src_padding_ind = SRC.vocab.stoi[BLANK_WORD]
    tgt_padding_ind = TGT.vocab.stoi[BLANK_WORD]
    tgt_eos_ind = TGT.vocab.stoi[EOS_WORD]
    
    # for each token, check if tokenized version of that token is the same (if so then 
    # spacy contains that token
    # docs for Iterator: https://github.com/pytorch/text/blob/c839a7934930819be7e240ea972e4d600966afdc/torchtext/data/iterator.py
    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, valid, test), batch_size=batch_size,
                                        sort_key=lambda x: data.interleave_keys(len(x.en), len(x.de)), 
                    sort_within_batch=True, shuffle=True,repeat=False) #,device=0)

    # NOTE: each batch in train_iter has shape (max numtokens in batch, batch size) but for each batch
    # may have different max num tokens

    datasetDict = {'train_iter':train_iter, 'val_iter':val_iter, 'test_iter':test_iter, 'SRC':SRC, 'TGT':TGT, 
                   'src_padding_ind':src_padding_ind, 'tgt_padding_ind':tgt_padding_ind, 'tgt_eos_ind':tgt_eos_ind}
    
    return datasetDict


