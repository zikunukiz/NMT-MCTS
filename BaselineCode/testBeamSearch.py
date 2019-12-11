

import os
import math, copy, time
import settings
import torch
import numpy as np

from createDatasetIterators import *
from transformerModel import *
from lossAndTestClasses import *


if __name__=='__main__':
	stime = time.time()
	dataset_dict = createIterators()

	print('LOADING TOOK: ',time.time()-stime)

	src_vocab_size = len(dataset_dict['SRC'].vocab.itos)
	tgt_vocab_size = len(dataset_dict['TGT'].vocab.itos)

	main_params = MainParams(dropout=0.1,enc_embedding_mat=None,
	              dec_embedding_mat=None,src_vocab_size=src_vocab_size,
	              tgt_vocab_size=tgt_vocab_size,batch_size=settings.BATCH_SIZE)

	pathToModel = './policy_supervised.pt'
	policy = TransformerModel(**(main_params.model_params)).to(main_params.device).double()
	policy.load_state_dict(torch.load(pathToModel))

	get_policy_ave_bleu(policy,dataset_dict,'test',main_params.device,main_params.num_decode_steps)
