
'''
Here we train value network by using policy network encoder/decoder params as starting point
then change the output linear layer so can be used for regression (output size of 1)

Training procedure:
1. Generate a translation using the trained policy
2. We'll randomly select a distinct state from a generated sequence to predict 
	our value from. We don't want to predict value from each state in the generated 
	sequence since these are strongly correlated (don't want a bunch of correlated updates in a row)
3. Now minimize least squares of true bleu score if follow our policy from current state vs our predicted
	bleu score.

Side Note: 
	PROBLEM: Tricky to jointly train policy and value in a single objective efficiently 
	because for each sentence, only want to do a single value update (as described above) 
	but then do policy updates at every step. One starting efficient way we can do this is
	to train policy network, clone encoder/decoder params and use in the value network and 
	train value network from that starting point. THIS IS HOW WE'LL START. 

'''


'''
TRAINING NOTES:
If use the scheduled adam from original transformer, we start off well then 
diverge quickly. So instead going to keep learnign rate a bit smaller 


'''

from torch.distributions.uniform import Uniform
from torch.nn import MSELoss 
from torch.distributions.categorical import Categorical
import warnings
import settings
from torch.distributions.uniform import Uniform
from torch.nn import MSELoss 
from torch import nn


warnings.filterwarnings("ignore") #THE NLTK library giving out mass amounts of warnings

batch_size = 64
#dataset_dict = createIterators(batch_size)
main_params = MainParams(dropout=0.2,src_vocab_size=len(dataset_dict['SRC'].vocab.itos),
			  tgt_vocab_size=len(dataset_dict['TGT'].vocab.itos),batch_size=batch_size)

pathToModel = settings.SAVED_MODELS_PATH+'policy_supervised.pt'
value_net = TransformerModel(**(main_params.model_params))
#value_net.load_state_dict(torch.load(pathToModel))
value_net.change_to_value_net()
value_net = value_net.to(main_params.device).double() #changes linear output layer and reinitializes those weights (keeps rest same as policy)
policy_net = TransformerModel(**(main_params.model_params)).to(main_params.device).double()
policy_net.load_state_dict(torch.load(pathToModel)) #need policy to simulate translations
policy_net.eval() #don't want dropout since not training policy anymore

loss_history = LossHistory()
opt = torch.optim.Adam(params=value_net.parameters(), lr=4e-4, amsgrad=False)
#opt = get_std_opt(value_net)
sigmoid = nn.Sigmoid()
loss_func = MSELoss()

for epoch in range(10000): #continue looping through epochs until convergence (based on validation loss) (WILL Want some restarts as well)
	data_iterator = None
	bleu_scores_train = [] #will use as our base for what we'd get on validation if just guessed average BLeu in train
	valid_scores_using_mean = []
	valid_bleu_scores = []
	for validate in [False,True]:
		if validate:
			value_net.eval()
			data_iterator = dataset_dict['val_iter']
		else:
			value_net.train()
			data_iterator = dataset_dict['train_iter']

		for batch in data_iterator:
			src_tensor = vars(batch)['de'].to(main_params.device) #will be (S,batch_size) where S is max num tokens of sentence in this batch
			trg_tensor = vars(batch)['en'].to(main_params.device) 
			if src_tensor.shape[1] != main_params.batch_size: #seems to be the last batch is of odd shape, can include if change code below though
				continue 
			src_key_padding_mask = (src_tensor==dataset_dict['src_padding_ind']).transpose(0,1) #need it to be (N,S) and is (S,N)
			
			#HOW TO CHOOSE WHEN RANDOM POINT TO GET VALUE FROM IS? Ideally different for each sentence in batch
			#important that random point is not after the EOS was put out. 

			dec_input = trg_tensor[0,:].view(1,-1) #each row has indices for words in each sentence at that level
			encoder_output = None #since after we get encoder output, should reuse since stays same for the sentence
			sentence_lens = torch.zeros(main_params.batch_size).to(main_params.device)
			eos_reached = torch.zeros(main_params.batch_size).type(torch.BoolTensor).to(main_params.device) 
			decode_s_time = time.time()
			
			#simulating output from policy (don't need to track computation on policy)
			num_decoding_steps = trg_tensor.shape[0]+5
			with torch.set_grad_enabled(False):
				for decoding_step in range(num_decoding_steps): 
					output,encoder_output = policy_net.forward(src_tensor,dec_input,
													  src_key_padding_mask=src_key_padding_mask,
													  tgt_mask=None,tgt_key_padding_mask=None,
													  memory_key_padding_mask=src_key_padding_mask,
													  memory=encoder_output)

					'''
					word_probs = F.softmax(output[decoding_step,:,:],dim=1) #output[decoding_step,:,:] has shape (batch_size,vocab_size)
					m = Categorical(word_probs)
					chosen_word = m.sample() #this generates along dim 1 of word_probs which is what we want since dim 1 contains distributions
					'''
					#instead going greedy search so less randomness that the value has to decipher
					word_rankings = F.log_softmax(output[decoding_step,:,:],dim=1)
					vals, indices = torch.max(word_rankings, 1) 
					chosen_word = indices
					#print('Chosen word shape: ',chosen_word.shape)
					if eos_reached.shape != chosen_word.shape:
						print('DIFFERENT sHAPES')
						print('src_tensor_shape: {}, tgttensor Shap: {}'.format(src_tensor.shape,trg_tensor.shape))

					eos_reached = (eos_reached | (chosen_word==dataset_dict['tgt_eos_ind'])) #set EOS if new word for sentence is EOS
					sentence_lens = sentence_lens + (~eos_reached) #so sentences which haven't reached EOS add 1 to length
					dec_input = torch.cat([dec_input,chosen_word.view(1,-1)],dim=0) #append chosen_word as row to end of decoder input for next iteration

			#time_to_decode = time.time()-decode_s_time
			#print('time to decode: ',time_to_decode)
			
			#now want to randomly choose numbers in  [0,sentence len]
			uniform_dist = Uniform(low=sentence_lens*0,high=sentence_lens)
			inds_to_use = uniform_dist.sample().round()

			#create tgt_key_padding_mask to mask out everything past inds_to_use (using a 1 means will be masked)
			tgt_key_padding_mask = torch.zeros((main_params.batch_size,num_decoding_steps+1)).type(torch.BoolTensor).to(main_params.device)
			for row in range(batch_size):
				tgt_key_padding_mask[row,(inds_to_use[row].int()+1):] = 1 

			bleu_cpu = get_bleu_scores(trg_tensor,dec_input,dataset_dict['TGT'])
			bleu_scores = bleu_cpu.to(main_params.device)
			
			if not validate: 
				bleu_scores_train.append(bleu_cpu.mean().item())
			else:
				#print('Bleu_csores: ',bleu_scores_train)
				#print('mean: ',np.mean(bleu_scores_train))
				#print('bleu_cpu
				valid_bleu_scores += [x for x in bleu_cpu]
				valid_scores_using_mean.append((np.square(bleu_cpu - np.mean(bleu_scores_train))).mean().item())
				#print('Valid scores: ',valid_scores_using_mean)
				#print(np.mean(valid_scores_using_mean))

			#print('SIZE decoder input: {}, size tgt_key_padding_mask: {}'.format(dec_input.shape,tgt_key_padding_mask.shape))
			#now run value net with mask
			set_grad = False if validate else True
			encoder_output=None
			with torch.set_grad_enabled(set_grad):
				output,encoder_output = value_net.forward(src_tensor,dec_input,src_key_padding_mask=src_key_padding_mask,
									tgt_mask=None,tgt_key_padding_mask=tgt_key_padding_mask,
									memory_key_padding_mask=src_key_padding_mask,memory=encoder_output)

				#Need to index differently for each sentnence(use index at inds_to_use[batch_num]
				batch_indices = [ind_1 for ind_1 in range(main_params.batch_size)]
				#print(output.shape)
				#print('first two outputs: {}, {}'.format(output[inds_to_use.long()[0],0,0],output[inds_to_use.long()[1],1,0]))
				#print(output[inds_to_use.long(),batch_indices,0][:2])
				output = sigmoid(output[inds_to_use.long(),batch_indices,0]) #want to squish to [0,1] since Bleu will be in [0,1]
				#print('predictions: ',output[:20])
				#print('true: ',bleu_scores[:20])
				#output = sigmoid(output[-1,:,0]) #want to squish to [0,1] since Bleu will be in [0,1]

				#print('OUTPUT Shape: {}, bleu_scores shpae: {}'.format(output.shape,bleu_scores.shape))
				loss = loss_func(output,bleu_scores.double())

				if not validate:
					opt.zero_grad()
					loss.backward()
					opt.step() 
					loss_history.add_batch_loss(loss,-1,'train')

					if loss_history.num_batches_processed % 500 == 0:
						print('num_batches_processed: ',loss_history.num_batches_processed)
						break
				else:
					loss_history.add_batch_loss(loss,-1,'valid')


	loss_history.add_epoch_loss(epoch)
	print('mse if used train mean: ',np.mean(valid_scores_using_mean))
	print('mean valid bleus: {}, variance: {}'.format(np.mean(valid_bleu_scores),np.var(valid_bleu_scores)))
	if loss_history.val_losses[-1] < np.min(loss_history.val_losses[:-1]):
		#SAVE MODEL (just saving parameters and specify hyper params in name)
		print('NEW BEST MODEL')
		model_path = pathToModel.replace('policy','value')
		torch.save(value_net.state_dict(),model_path)

	if len(loss_history.val_losses) > 80 and np.min(loss_history.val_losses[:-40]) < np.min(loss_history.val_losses[-40:]):
		print('\nCONVERGED AFTER: {} Minutes \n'.format((time.time()-start_time)/60))
		break

