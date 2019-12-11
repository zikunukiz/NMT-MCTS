

'''
Here we load in policy which was fully trained in supervised setting and now improve it 
using policy gradients (directly optimizing for BLEU)
'''

from torch.distributions.categorical import Categorical
import warnings
from createDatasetIterators import createIterators
from transformerModel import *
from lossAndTestClasses import *
import torch
import torch.nn.functional as F
import numpy as np
import settings

warnings.filterwarnings("ignore") #THE NLTK library giving out mass amounts of warnings


batch_size = 16
dataset_dict = createIterators(batch_size)
main_params = MainParams(dropout=0.2,src_vocab_size=len(dataset_dict['SRC'].vocab.itos),
              tgt_vocab_size=len(dataset_dict['TGT'].vocab.itos),batch_size=batch_size)

#here want to load a previous policy first (clone it) and then improve it
pathToModel = path+'policy_supervised.pt'
policy = TransformerModel(**(main_params.model_params)).to(main_params.device).double()
policy.load_state_dict(torch.load(pathToModel))

#going to want to loop over different learning rates for our Adam here
#look up if people do more complex things in practice
loss_history = LossHistory()
loss_history.val_losses[0] = 0 #since here larger number is better
opt = torch.optim.Adam(params=policy.parameters(), lr=1e-5, amsgrad=False) #is learning rate used in paper

#opt = get_std_opt(policy) #torch.optim.Adam(params=model.parameters(), amsgrad=amsgrad)

for epoch in range(10000): #continue looping through epochs until convergence (based on validation loss) (WILL Want some restarts as well)

  #now loop over training batches
  policy.train() #turning on train mode
  #every 500 batches processed, check validation score
  start_time = time.time()
  num_sentences_used = 0
  for batch_count, batch in enumerate(dataset_dict['train_iter']):
      #print('src shape: {}, tgt shape: {}'.format(vars(batch)['de'].shape,vars(batch)['en'].shape))
      src_tensor = vars(batch)['de'].to(main_params.device) #will be (S,batch_size) where S is max num tokens of sentence in this batch
      trg_tensor = vars(batch)['en'].to(main_params.device) 
      num_sentences_used += src_tensor.shape[1]

      src_key_padding_mask = (src_tensor==dataset_dict['src_padding_ind']).transpose(0,1) #need it to be (N,S) and is (S,N)
     
      #here have to get output one step at a time (will choose which word by sampling from the distribution
      #which will give unbiased estimate of our expected reward from our policy
      '''
      start by feeding decoder only start of sentence tokens for each
      sentence in batch
      '''                          
      dec_input = trg_tensor[0,:].view(1,-1) #each row has indices for words in each sentence at that level
      encoder_output = None #since after we get encoder output, should reuse since stays same for the sentence
      '''
      Maintaining two vectors, one has sum of log probs of words (eg first element in vector
              has sum of log probs of words in first sentence so far (not including log probs after output EOS token)
              Second vector is boolean that is true in element if we're still decoding the sentence and false if 
              have already hit EOS token for that sentence.
      '''
      eos_reached = torch.zeros(main_params.batch_size, dtype=torch.uint8).type(torch.BoolTensor).to(main_params.device)
      sum_log_probs = torch.zeros(main_params.batch_size).to(main_params.device) # (IS THIS KEEPING TRACK OF GRADIENT AS WE WANT?)
      num_decode_steps = trg_tensor.shape[0] + 5 #THIS wouldn't ideally be here but have strict memory size constraints
      #print(num_decode_steps)
      for decoding_step in range(num_decode_steps): 

        output,encoder_output = policy.forward(src_tensor,dec_input,src_key_padding_mask=src_key_padding_mask,
                          tgt_mask=None,tgt_key_padding_mask=None,
                          memory_key_padding_mask=src_key_padding_mask,memory=encoder_output)

        '''
        Now make prediction on next word
        we're only interested in the values of the embeddings of the current 
        step we're on which we can get by output[decoder_step,:,:]
        which has dim (batch_size,vocab_size)
        ''' 
        word_probs = F.softmax(output[decoding_step,:,:],dim=1) 

        #IMPLEMENTATION FOR REINFORCE VERY EASY https://pytorch.org/docs/stable/distributions.html
        #MAKE SURE THAT THIS BACKPROPS THROUGH EVERY ACTION WE CHOOSE
        m = Categorical(word_probs)
        chosen_word = m.sample() #this generates along dim 1 of word_probs which is what we want since dim 1 contains distributions
        log_prob_this_word = m.log_prob(chosen_word) #now decide if want to add these by if the sentence is done already
        #print('Chosen word shape: {}, vector: {}'.format(chosen_word.shape,chosen_word))

        eos_reached = (eos_reached | (chosen_word==dataset_dict['tgt_eos_ind'])) #set EOS if new word for sentence is EOS
        #now mask log probs which are part of sentences which have already completed
        #print('log_prob_this_word.shape: {}, eos_reached.shape: {}'.format(log_prob_this_word.shape,eos_reached.shape))
        log_prob_this_word = log_prob_this_word * (~eos_reached) #elementwise multiplication (acts as mask)
        sum_log_probs += log_prob_this_word
        dec_input = torch.cat([dec_input,chosen_word.view(1,-1)],dim=0) #append chosen_word as row to end of decoder input for next iteration
      

      reward = get_bleu_scores(trg_tensor,dec_input,dataset_dict['TGT']).to(main_params.device)
      loss = (-sum_log_probs * reward).sum() #.mean() 
      #print('shape1: {}, shape2: {},loss: {}'.format((-sum_log_probs * reward).shape, loss.shape,loss))
      
      loss.backward()

      #doing larger batch updates
      reached_4_batches = (batch_count != 0 and batch_count % 8 == 0)
      if reached_4_batches:
        for p in policy.parameters():
          #Maybe there is a part of the network that isn't involved in compuation here? 
          #This turned out to be transformer module inside that we don't use
          if not p.grad is None: 
            p.grad /= num_sentences_used  #this replicates taking gradient of all 4 batches at once. 
        
        opt.step() #now take gradient step (POSSIBLY SHOULD TAKE BIGGER BATCHES HERE FOR GRADIENT ACCURACY
        opt.zero_grad()
        num_sentences_used = 0

      loss_history.add_batch_loss(reward.mean(),-1,'train')
      if loss_history.num_batches_processed % 400 == 0:
        print('time for 400 batches: ',time.time()-start_time)
        break

  opt.zero_grad() #0 out for next iteration through

  #USE BEAM SEARCH INSTEAD HERE (DOESN'T really matter if not overly fast since only like 1500 sentences)
  #get validation bleu score 
  val_ave_bleu = get_policy_ave_bleu(policy,dataset_dict,'val',main_params.device,main_params.num_decode_steps)
  
  loss_history.add_batch_loss(val_ave_bleu,-1,'valid')
  
  loss_history.add_epoch_loss(epoch)

  if loss_history.val_losses[-1] > np.max(loss_history.val_losses[:-1]) and loss_history.val_losses[-1]>27.56:
    #SAVE MODEL (just saving parameters and specify hyper params in name)
    print('NEW BEST MODEL')
    model_path = pathToModel[:-3] + '_RLTrained.pt'
    torch.save(policy.state_dict(),model_path)

  if len(loss_history.val_losses) > 30 and np.min(loss_history.val_losses[:-15]) < np.min(loss_history.val_losses[-15:]):
    print('\nCONVERGED AFTER: {} Minutes \n'.format((time.time()-start_time)/60))
    break


get_policy_ave_bleu(policy,dataset_dict,'test',main_params.device,main_params.num_decode_steps) #GET TEST SCORES

