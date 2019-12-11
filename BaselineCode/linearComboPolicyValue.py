

'''
See if using linear combo of policy and value net during inference helps performance.

At each step of decoding, we have policy output and then 
for say top k outputs of policy, run value net and get 
corresponding values. Now we choose next word based on 
lambda*policy + (1-lambda)*value for each of these k words. 
Need to optimize these two hyperparams on the validation set 
(and show table of how BLEU changes for change in hyperparams.
'''


batch_size = 32
dataset_dict = createIterators(batch_size)
main_params = MainParams(dropout=0.2,src_vocab_size=len(dataset_dict['SRC'].vocab.itos),
              tgt_vocab_size=len(dataset_dict['TGT'].vocab.itos),batch_size=batch_size)

pathToPolicy = settings.SAVED_MODELS_PATH+'policy_supervised_RLTrained.pt'
pathToValue = settings.SAVED_MODELS_PATH+'value_supervised_RLTrained.pt'
value_net = TransformerModel(**(main_params.model_params)).to(main_params.device).double()
value_net.change_to_value_net()
value_net.load_state_dict(torch.load(pathToValue))

policy_net = TransformerModel(**(main_params.model_params)).to(main_params.device).double()
policy_net.load_state_dict(torch.load(pathToPolicy)) #need policy to simulate translations

policy_net.eval() #don't want dropout since not training policy anymore
value_net.eval()

TGT = dataset_dict['TGT']
SRC = dataset_dict['SRC']
  

resultsTable = []

#dataset_type is either val or test

for lam in [1.,0.,0.5,0.3,0.7,0.1,0.9]:
	k = 10 #tells us how many of the top policy words to look at. 
	predicted_sentences = []
	true_sentences = []
	with torch.set_grad_enabled(False):
		dataset_iterator = dataset_dict['val_iter']
		for batch in dataset_iterator:
			src_tensor = vars(batch)['de'].to(device) # is shape (S,N) where N batch size, S:largestnum tokens in batch
			trg_tensor = vars(batch)['en'].to(device) #transfer onto GPU

			#create masks (Here don't need target masks since outputting 1 word at a time.
			src_key_padding_mask = (src_tensor==dataset_dict['src_padding_ind']).transpose(0,1) #need it to be (N,S) and is (S,N)
			dec_input = trg_tensor[0,:].view(1,-1) #give BOS for each sentence in batch
			encoder_output_policy = None
			encoder_output_value = None
			num_decode_steps = trg_tensor.shape[0]+10

			for decoding_step in range(num_decode_steps):
				output_policy,encoder_output_policy = policy_net.forward(src_tensor,dec_input,src_key_padding_mask=src_key_padding_mask,
				                    tgt_mask=None,tgt_key_padding_mask=None,
				                    memory_key_padding_mask=src_key_padding_mask,memory=encoder_output_policy)

				#output is dim (T,batch_size,vocab_size) except last dim size is 1 for value_net
				word_rankings = F.log_softmax(output_policy[decoding_step,:,:],dim=1)

				#now get top k words from this and get values for them
				vals,indices = torch.sort(word_rankings,dim=1,descending=True) 
				tokens_to_add = indices[:,:k]
				#now for one sentence at a time, replicate the dec_input of that sentence across the 
				#batch_dim then stack on the new words and run these through value_net to get values
				next_words_chosen = torch.zeros(1,trg_tensor.shape[1])
				for sentence_ind in range(trg_tensor.shape[1]):
					mod_dec_input = dec_input[:,sentence_ind].view(-1,1).repeat(1,k) #modified decoder input
					#now stack on new words to get values for
					mod_dec_input = torch.cat([mod_dec_input,tokens_to_add[sentence_ind,:].view(1,k)],dim=0)

					output_value,encoder_output_value = value_net.forward(src_tensor,mod_dec_input,src_key_padding_mask=src_key_padding_mask,
					                tgt_mask=None,tgt_key_padding_mask=None,
					                memory_key_padding_mask=src_key_padding_mask,memory=encoder_output_value)

					#output should be dim (Len sentence, k, 1)
					value_net_vals = output_value[-1,:,0]
					policy_vals = vals[sentence_ind,:k]

					#now choose next best word for this sentence
					combo = lam*policy_vals + (1-lam)*value_net_vals
					print('COMBO: ', combo)
					max_val,max_ind = torch.max(combo)
					print('MAX_val: {}, max_ind: {}'.format(max_val,max_ind))
					next_words_chosen[0,sentence_ind] = tokens_to_add[sentence_ind,max_ind]

			
			dec_input = torch.cat([dec_input,next_words_chosen],dim=0)


		preds,targs = getPredAndTargSentences(trg_tensor,dec_input,TGT):   
		predicted_sentences += preds
		true_sentences += targs
  
  #here finally calculate BLEU (want to save file with the table of BLEUS
  bleu_score = sacrebleu.corpus_bleu(pred_sentences, [true_sentences],use_effective_order=True).score
  print('Lam: {}, k: {}, Bleu: {}'.format(lam,k,bleu_score))
  resultsTable.append([lam,k,bleu_score])

json.dump(resultsTable,open(path+'policy_value_table.json','w'))