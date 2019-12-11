#here we want to find out if having the value network compliments the policy
#using actor critic where policy gradient same as in REINFORCE section except now
#subtracts baseline at each step (value net evaluation at that state)
#and value net is updated by sum((r-v(s_n))*gradient(v(s_n)))
'''
NOTE: This was meant to run in Google colab so may need several changes to run elsewhere.
      Mainly 'imports' 
'''


from torch.distributions.categorical import Categorical
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import time
warnings.filterwarnings("ignore") #THE NLTK library giving out mass amounts of warnings


batch_size = 16
dataset_dict = createIterators(batch_size)
main_params = MainParams(dropout=0.2,src_vocab_size=len(dataset_dict['SRC'].vocab.itos),
              tgt_vocab_size=len(dataset_dict['TGT'].vocab.itos),batch_size=batch_size)

#here want to load a previous policy first (clone it) and then improve it
pathToPolicy = path+'policy_supervised.pt'
pathToPolicy = pathToPolicy[:-3] + '_RLTrainedBatch256.pt'

pathToValue = path+'value_supervised_RLTrained.pt'

value_net = TransformerModel(**(main_params.model_params))
value_net.change_to_value_net()
value_net = value_net.to(main_params.device).double()
value_net.load_state_dict(torch.load(pathToValue))

policy_net = TransformerModel(**(main_params.model_params)).to(main_params.device).double()
policy_net.load_state_dict(torch.load(pathToPolicy)) #need policy to simulate translations



#going to want to loop over different learning rates for our Adam here
#look up if people do more complex things in practice
loss_history = LossHistory()
loss_history.val_losses[0] = 0 #since here larger number is better
opt_policy = torch.optim.Adam(params=policy_net.parameters(), lr=1e-4) #is learning rate used in paper
opt_value = torch.optim.Adam(params=value_net.parameters(), lr=1e-4)

#opt = get_std_opt(policy) #torch.optim.Adam(params=model.parameters(), amsgrad=amsgrad)
sigmoid = nn.Sigmoid() #need to squish value output

for epoch in range(10000): #continue looping through epochs until convergence (based on validation loss) (WILL Want some restarts as well)

  #now loop over training batches
  policy_net.train() #turning on train mode
  value_net.train() 

  start_time = time.time()
  num_sentences_used = 0
  for batch_count, batch in enumerate(dataset_dict['train_iter']):
      #print('src shape: {}, tgt shape: {}'.format(vars(batch)['de'].shape,vars(batch)['en'].shape))
      src_tensor = vars(batch)['de'].to(main_params.device) #will be (S,batch_size) where S is max num tokens of sentence in this batch
      trg_tensor = vars(batch)['en'].to(main_params.device) 
      num_sentences_used += src_tensor.shape[1]
      indexing_list = [x for x in range(trg_tensor.shape[1])]

      src_key_padding_mask = (src_tensor==dataset_dict['src_padding_ind']).transpose(0,1) #need it to be (N,S) and is (S,N)
     
      #here have to get output one step at a time (will choose which word by sampling from the distribution
      #which will give unbiased estimate of our expected reward from our policy
      '''
      start by feeding decoder only start of sentence tokens for each
      sentence in batch
      '''                          
      dec_input = trg_tensor[0,:].view(1,-1) #each row has indices for words in each sentence at that level
      encoder_output = None #since after we get encoder output, should reuse since stays same for the sentence
      
      num_decode_steps = trg_tensor.shape[0] + 5 #THIS wouldn't ideally be here but have strict memory size constraints
      #print(num_decode_steps)
      with torch.set_grad_enabled(False):
        policy_net.eval()
        for decoding_step in range(num_decode_steps): 

            output,encoder_output = policy_net.forward(src_tensor,dec_input,src_key_padding_mask=src_key_padding_mask,
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
            dec_input = torch.cat([dec_input,chosen_word.view(1,-1)],dim=0) #append chosen_word as row to end of decoder input for next iteration

      policy_net.train()
      reward = get_bleu_scores(trg_tensor,dec_input,dataset_dict['TGT'],BLEU1=False).to(main_params.device)
      
      #now that we have reward, go back through the path and get gradients
      eos_reached = torch.zeros(main_params.batch_size, dtype=torch.uint8).type(torch.BoolTensor).to(main_params.device)
      encoder_output_detached = encoder_output.clone().detach()
      #print('Num decode steps: ',dec_input.shape[0])
      #print('batch size: ',src_tensor.shape[1])
      starting_time = time.time()
      for decode_step in range(1,dec_input.shape[0]):
        #now reuse encoder output from above 
        mod_dec_input = dec_input[:decode_step,:]
        
        #print('timing 1: ')
        start_time = time.time()
        output_policy,encoder_output = policy_net.forward(src_tensor,mod_dec_input,src_key_padding_mask=src_key_padding_mask,
                          tgt_mask=None,tgt_key_padding_mask=None,
                          memory_key_padding_mask=src_key_padding_mask,memory=encoder_output)

        
        output_value,unused_output_value = value_net.forward(src_tensor,mod_dec_input,src_key_padding_mask=src_key_padding_mask,
                                tgt_mask=None,tgt_key_padding_mask=None,
                                memory_key_padding_mask=src_key_padding_mask,memory=encoder_output)#_detached)
            
        #total = time.time() - starting_time
        #print(total)
        '''
        print('time2: ')
        start_time= time.time()
        
        for count2 in range(src_tensor.shape[1]):
          encoder_output2 = encoder_output[:,count2,:].unsqueeze(1)
          src_tensor2 = src_tensor[:,count2].view(-1,1)
          dec_input2 = mod_dec_input[:,count2].view(-1,1)
          padding_s = src_key_padding_mask[count2,:].view(1,-1)
          output_policy2,encoder_output2 = policy_net.forward(src_tensor2,dec_input2,src_key_padding_mask=padding_s,
                          tgt_mask=None,tgt_key_padding_mask=None,
                          memory_key_padding_mask=padding_s,memory=encoder_output2)

        
          output_value2,unused_output_value = value_net.forward(src_tensor2,dec_input2,src_key_padding_mask=padding_s,
                                  tgt_mask=None,tgt_key_padding_mask=None,
                                  memory_key_padding_mask=padding_s,memory=encoder_output2)
              

        total = time.time()-start_time
        print('total: ',total)
        print('batchSize: ',src_tensor.shape[1])
        '''

        log_softmax_output = F.log_softmax(output_policy[-1,:,:],dim=1)
        #we already know for each sentence what chosen word was so just get probs of those
        policy_log_probs = log_softmax_output[indexing_list,dec_input[decode_step,:]]
        
        #for policy we want prob of next word, while for value want value of this current state
        #which gives us a sense of advantage function
        value_net_vals = sigmoid(output_value[-1,:,0])
        value_vals_detached = value_net_vals.clone().detach()
        
        eos_reached = (eos_reached | (dec_input[decode_step,:]==dataset_dict['tgt_eos_ind']))
        advantage = (reward - value_vals_detached)*(~eos_reached)
        
        #now get gradients (will add 0 to gradient if reached EOS)
        policy_func = (advantage*policy_log_probs).sum()
        policy_func.backward()
        
        value_func = (advantage*value_net_vals).sum()
        value_func.backward()
      
      #print(time.time()-starting_time)
      #print('time taken')
      #doing larger batch updates
      reached_8_batches = (batch_count != 0 and batch_count % 32 == 0)
      if reached_8_batches:
        for network in [policy_net, value_net]:
            for p in network.parameters():
              #Maybe there is a part of the network that isn't involved in compuation here? 
              #This turned out to be transformer module inside that we don't use
              if not p.grad is None: 
                p.grad /= num_sentences_used  #this replicates taking gradient of all 4 batches at once. 

        
        opt_policy.step() #now take gradient step (POSSIBLY SHOULD TAKE BIGGER BATCHES HERE FOR GRADIENT ACCURACY
        opt_policy.zero_grad()
        
        opt_value.step()
        opt_value.zero_grad()
        
        num_sentences_used = 0

        loss_history.add_batch_loss(reward.mean(),-1,'train')
        #if loss_history.num_batches_processed % 400 == 0:
        #print('time for 400 batches: ',time.time()-start_time)
        print('finished grad update')
        break

  opt_policy.zero_grad() #0 out for next iteration through
  opt_value.zero_grad()
    
  #get validation bleu score 
  val_ave_bleu = get_policy_ave_bleu(policy_net,dataset_dict,'val',main_params.device,main_params.num_decode_steps,useBLEU1=False)
  
  loss_history.add_batch_loss(val_ave_bleu,-1,'valid')
  
  loss_history.add_epoch_loss(epoch)

  if loss_history.val_losses[-1] > np.max(loss_history.val_losses[:-1]):
    #SAVE MODEL (just saving parameters and specify hyper params in name)
    print('NEW BEST MODELS')
    torch.save(policy_net.state_dict(),pathToPolicy[:-3] + '_AC.pt')
    torch.save(value_net.state_dict(),pathToValue[:-3] + '_AC.pt')
    

  if len(loss_history.val_losses) > 30 and np.min(loss_history.val_losses[:-15]) < np.min(loss_history.val_losses[-15:]):
    print('\nCONVERGED AFTER: {} Minutes \n'.format((time.time()-start_time)/60))
    break


get_policy_ave_bleu(policy,dataset_dict,'test',main_params.device,main_params.num_decode_steps) #GET TEST SCORES
