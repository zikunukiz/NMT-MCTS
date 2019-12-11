



import numpy as np
import torch
import math, copy, time
import os
import math, copy, time
from transformerModel import *
from lossAndTestClasses import *
import settings
from createDatasetIterators import createIterators

import torch.nn.functional as F


batch_size = 64
dataset_dict = createIterators(batch_size)

main_params = MainParams(dropout=0.2,src_vocab_size=len(dataset_dict['SRC'].vocab.itos),
              tgt_vocab_size=len(dataset_dict['TGT'].vocab.itos),batch_size=batch_size)

start_time = time.time()
model = TransformerModel(**(main_params.model_params)).double().to(main_params.device)
pathToModel = settings.SAVED_MODELS_PATH+'policy_supervised.pt'
#policy = TransformerModel(**(main_params.model_params)).to(main_params.device).double()
model.load_state_dict(torch.load(pathToModel))



#loss_func_old = nn.CrossEntropyLoss(ignore_index=dataset_dict['tgt_padding_ind']) #tries to maximize log of probability that correct word is chosen.
loss_func = nn.CrossEntropyLoss(reduction='none')
opt = get_std_opt(model) 
#opt = torch.optim.Adam(params=model.parameters(), amsgrad=False)
loss_history = LossHistory() #will keep track of our training and validation losses
#label_smoothing = LabelSmoothing(size=len(dataset_dict['TGT'].vocab.itos), padding_idx=dataset_dict['tgt_padding_ind'], smoothing=0.1)
    
for epoch in range(10000): #here just using name epoch as going through 500 batches
    data_iterator = None
    for validate in [False,True]:
        if validate:
            model.eval()
            data_iterator = dataset_dict['val_iter']
        else:
            model.train()
            data_iterator = dataset_dict['train_iter']

        enable_grad = False if validate else True
        with torch.set_grad_enabled(enable_grad):
            for batch in data_iterator:
                #print(vars(batch)['de'].is_cuda)
                src_tensor = vars(batch)['de'].to(main_params.device) #will be (S,batch_size) where S is max num tokens of sentence in this batch
                trg_tensor = vars(batch)['en'].to(main_params.device) #transfer onto GPU
                #print(src_tensor.shape)

                #src_tensor2 = src_tensor*(~(src_tensor==dataset_dict['src_padding_ind']))
                #print('src first sentence: ',src_tensor2[:,0])
                #print([dataset_dict['SRC'].vocab.itos[j2] for j2 in src_tensor[:,0]])
                #need to create two versions of target, 1. to input to decoder, 2. to use as true value
                trg_dec_input = trg_tensor[:-1,:] 
                trg_dec_true = trg_tensor[1:,:] #shift right one token so dec has to predict the next unseen token.

                #create masks
                tgt_mask = generate_square_subsequent_mask(sz=trg_dec_input.shape[0]).to(main_params.device)
                tgt_key_padding_mask = (trg_dec_input==dataset_dict['tgt_padding_ind']).transpose(0,1) #true values will be masked
                src_key_padding_mask = (src_tensor==dataset_dict['src_padding_ind']).transpose(0,1) #need it to be (N,S) and is (S,N)
                
                #print("Mask shapes: tgt_mask: {}, src_key_padding_mask: {}, tgt_key_padding_mask: {}".format(tgt_mask.shape,
                #                                                                                            src_key_padding_mask.shape,
                #                                                                                            tgt_key_padding_mask.shape)) 
          
                output,memory_garbage = model.forward(src_tensor,trg_dec_input,src_key_padding_mask=src_key_padding_mask,
                              tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=src_key_padding_mask)

                #print('output shape from transformer: ',output.shape)
                output = output.transpose(1,2) #need to switch 2 and 3rd dims so is right size for loss
                
                
                #output = masked_cross_entropy(trg_dec_true,output,loss_func) #this ignores padding tokens from our crossentropy calculation
                #masked_cross_entropy(real,pred,loss_func)
                if validate: 
                    output,num_tokens = masked_cross_entropy(trg_dec_true,output,loss_func)  #THIS IS MORE INTERPREBALE FOR VALIDATION
                    loss_history.add_batch_loss(output,num_tokens,'valid')
                else:
                    #log_probs = F.log_softmax(output,dim=1)
                    #print('shape log_probs: {}, shape output: {}'.format(log_probs.shape,output.shape))
                    #output,num_tokens = label_smoothing(log_probs,trg_dec_true) #MAKE SURE THIS IS WORKING PROPERLY WITH THESE ADDED DIMS
                    output,num_tokens = masked_cross_entropy(trg_dec_true,output,loss_func) 
                    #print('Shape output from label_smoothing: {}'.format(output.shape))
                    #output = output.mean() 
                
                    opt.optimizer.zero_grad()
                    output.backward()
                    opt.step() #now take gradient step
                    loss_history.add_batch_loss(output,num_tokens,'train')
                    if loss_history.num_batches_processed % 500 == 0:
                      print('Processed #batches: ',loss_history.num_batches_processed)
                    #    break #go and evaluate on validation set

  
    loss_history.add_epoch_loss(epoch)
    if loss_history.val_losses[-1] < 1.8952 and loss_history.val_losses[-1] < np.min(loss_history.val_losses[:-1]):
        print('NEW BEST MODEL')
        if epoch > 0:
          model_name = 'policy_supervised.pt'
          torch.save(model.state_dict(),settings.SAVED_MODELS_PATH+model_name) #SAVE MODEL (just saving parameters)

    if len(loss_history.val_losses) > 30 and np.min(loss_history.val_losses[:-15]) < np.min(loss_history.val_losses[-15:]):
        print('STOPPING TRAINING')
        print('CONVERGED AFTER: {} Minutes '.format((time.time()-start_time)/60))
        break


#GET TEST SCORES
get_policy_ave_bleu(model,dataset_dict,'test',main_params.device,main_params.num_decode_steps)


