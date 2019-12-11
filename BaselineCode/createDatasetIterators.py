
import settings
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext import data, datasets
import numpy as np
import pandas as pd


#combines train.en and train.de and so on so have src and tgt sentences in same file
def combine_src_tgt():
	for d_type in ['train','valid','test']:
		df1 = pd.read_csv(settings.DATA_PATH+'/'+d_type+'.en',header=None,names=['en'],sep='\t')
		df2 = pd.read_csv(settings.DATA_PATH+'/'+d_type+'.de',header=None,names=['de'],sep='\t')
		df2['en'] = df1['en'].values
		df2['en_len'] = df2['en'].apply(lambda x: len(x.split(' ')))
		df2 = df2.loc[df2['en_len'] <= 50]
		df2.loc[:,['de','en']].to_csv(settings.DATA_PATH+'/combined_'+d_type+'.tsv',sep='\t',header=False,index=False)
		print(df2.head())


'''
Create dataset iterators for train,validation,and test datasets
'''
def createIterators(batch_size):
	tokenize = lambda x: x.split(' ') #already have tokenized files so now just split

	SRC = Field(sequential=True, tokenize=tokenize, init_token = settings.BOS_WORD, 
										eos_token = settings.EOS_WORD, pad_token=settings.BLANK_WORD)
	TGT = Field(sequential=True, tokenize=tokenize, init_token = settings.BOS_WORD, 
										eos_token = settings.EOS_WORD, pad_token=settings.BLANK_WORD)

	train,valid,test = TabularDataset.splits(
               path=settings.DATA_PATH, # the root directory where the data lies
               train='combined_train.tsv',
               validation='combined_valid.tsv',
               test = 'combined_test.tsv',
               fields=[('de',SRC),('en',TGT)],
               format='TSV',
               skip_header=False) 
 

	SRC.build_vocab(train.de)
	TGT.build_vocab(train.en)

	src_padding_ind = SRC.vocab.stoi[settings.BLANK_WORD]
	tgt_padding_ind = TGT.vocab.stoi[settings.BLANK_WORD]
	tgt_eos_ind = TGT.vocab.stoi[settings.EOS_WORD]
	
	#docs for Iterator: https://github.com/pytorch/text/blob/c839a7934930819be7e240ea972e4d600966afdc/torchtext/data/iterator.py
	#buckets examples into batches where sentences are close to same size
	#which allows less padding and faster computation
	train_iter, val_iter, test_iter = data.BucketIterator.splits((train, valid, test), batch_size=batch_size,
										sort_key=lambda x: data.interleave_keys(len(x.en), len(x.de)), 
                    					sort_within_batch=True, shuffle=True,repeat=False) #,device=0)

	datasetDict = {'train_iter':train_iter,'val_iter':val_iter,'test_iter':test_iter,
					'SRC':SRC, 'TGT':TGT,'src_padding_ind':src_padding_ind, 'tgt_padding_ind':tgt_padding_ind,
					'tgt_eos_ind':tgt_eos_ind}
	
	return datasetDict


