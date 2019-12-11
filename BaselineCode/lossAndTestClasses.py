

import torch
import settings
import torch.nn.functional as F
import numpy as np
import sacrebleu
import time

'''
Some functions here are for testing the model (getting test set BLEU score).
We also have the loss class and a parameter holding class here. 

Here we don't teacher force so start by feeding decoder with <BOS> token, output a new token, and continue
until we've output say 50 tokens (now look at each sentence and compare the sentence up until it output <EOS> 
which the true sentence in terms of BLEU. THIS MEANS WANT TO TRAIN OUR MODEL TO KNOW WHEN TO OUTPUT EOS IN TRAINING
DOES INPUT SENTENCE HAVE EOS on it?

'''

#now need to make these into true sentences again (many parts are split with @@ at end)
#this is mainly undoing the stuff done by our tokenizer
#two weird cases are they &apos;re which is they're, and others end in @@ which means to concat
#and there's this: &quot; which I think is supposed to be quotations
def fix_sentence(sentence,as_str=False):  
    #if as_str==True then join tokens by space unless comma or period
    new_sentence = []
    cur_word = ''
    for p in sentence:
      if '@@' in p:
        cur_word += p[:-2]
      else:
        if cur_word != '':
          new_sentence.append(cur_word+p)
          cur_word = ''
        elif '&' in p and ';' in p: #this means should be adding this onto last added word 
          if len(new_sentence) >0:
            new_sentence[-1] = new_sentence[-1] + "'"+p.split(';')[1]
          #OTHERWISE NOT SURE WHAT TO DO
          else:
            pass #NEED TO IMPLEMENT
        else:
          new_sentence.append(p)  
    
    str_to_ret= ''
    if as_str:
      for w in new_sentence:

        if w in [',','.','?'] and len(str_to_ret)!=0 and str_to_ret[-1]==' ': #remove last space added
          
          str_to_ret = str_to_ret[:-1] 
        elif  w in [',','.','?'] and len(str_to_ret)!=0:
          #print('Weird sentence: ', new_sentence)
          pass

        str_to_ret += w
        if not (w in ['.','?']):
          str_to_ret += ' '
      
      return str_to_ret
    
    return new_sentence

#returns average bleu score of batch
def get_bleu_scores(trg_tensor,pred_tensor,TGT):
  bleus_per_sentence = torch.zeros(trg_tensor.shape[1],requires_grad=False) 
  for col in range(trg_tensor.shape[1]): #each column contains sentence
    true_sentence = [TGT.vocab.itos[i] for i in trg_tensor[:,col] if TGT.vocab.itos[i] != settings.BLANK_WORD][1:-1]
    pred_sentence = [TGT.vocab.itos[i] for i in pred_tensor[:,col] if TGT.vocab.itos[i] != settings.BLANK_WORD]
    #print('Before: ')
    #print(true_sentence)
    #print(pred_sentence)
    #now also need to stop pred_sentence after first EOS_WORD outputted
    #also don't want to use BOS chars
    ind_first_eos = 0
    for tok in pred_sentence:
      if tok == settings.EOS_WORD:
        break
      ind_first_eos += 1
    
    if ind_first_eos != 0:
      pred_sentence = pred_sentence[1:ind_first_eos] #this gets rid of EOS_WORD

    
    #now undo some of the weird tokenization
    
    pred_sentence = fix_sentence(pred_sentence,as_str=True)
    true_sentence = fix_sentence(true_sentence,as_str=True)

    #This bleu_score defaults to calculating BLEU-4 (not normal bleu) so change weights,
    #this change of weights gives BLEU based on 1-grams so normal bleu I believe
    #score = nltk.translate.bleu_score.sentence_bleu([true_sentence], pred_sentence, weights=(1, 0, 0, 0))
    score = sacrebleu.sentence_bleu(pred_sentence, true_sentence,smooth_method='exp').score
    #score = score*len(true_sentence)
    
    
    #print(true_sentence)
    #print(pred_sentence)
    #print(score)
    #print()
    bleus_per_sentence[col] = score/100.0 
  return bleus_per_sentence

def getPredAndTargSentences(trg_tensor,pred_tensor,TGT):

  preds = []
  targs = []
  for col in range(trg_tensor.shape[1]): #each column contains sentence
    true_sentence = [TGT.vocab.itos[i] for i in trg_tensor[:,col] if TGT.vocab.itos[i] != settings.BLANK_WORD][1:-1]
    pred_sentence = [TGT.vocab.itos[i] for i in pred_tensor[:,col] if TGT.vocab.itos[i] != settings.BLANK_WORD]
    #print('Before: ')
    #print(true_sentence)
    #print(pred_sentence)
    #now also need to stop pred_sentence after first EOS_WORD outputted
    #also don't want to use BOS chars
    ind_first_eos = 0
    for tok in pred_sentence:
      if tok == settings.EOS_WORD:
        break
      ind_first_eos += 1
    
    if ind_first_eos != 0:
      pred_sentence = pred_sentence[1:ind_first_eos] #this gets rid of EOS_WORD

    
    #now undo some of the weird tokenization
    #print('pred before: ',pred_sentence)
    #print('true before: ',true_sentence)
    pred_sentence = fix_sentence(pred_sentence,as_str=True)
    true_sentence = fix_sentence(true_sentence, as_str=True)
    preds.append(pred_sentence)
    targs.append(true_sentence)
    #print('pred after: ',pred_sentence)
    #print('true after: ',true_sentence)
    
    #This bleu_score defaults to calculating BLEU-4 (not normal bleu) so change weights,
    #this change of weights gives BLEU based on 1-grams so normal bleu I believe
    #score = nltk.translate.bleu_score.sentence_bleu([true_sentence], pred_sentence, weights=(1, 0, 0, 0))
    
    #score = score*len(true_sentence)
    
  return preds,targs 




def print_test_summary(bleu_scores):
  mean_bleu = np.mean(bleu_scores) 
  std_mean = np.std(bleu_scores,ddof=1)/np.sqrt(len(bleu_scores))   
  print('\nTEST SUMMARY:\nAVERAGE BLEU SCORE: {}, STD of mean: {}'.format(mean_bleu,std_mean))
  print('95% CI: [{}, {}]'.format(mean_bleu-1.96*std_mean, mean_bleu+1.96*std_mean))


#speed this up by batching (check how slow is for us to do one at a time
'''
some modifications: Each time one of our beams ends up using a EOS, we store this
  sentence and then average log prob per token in that sentence. We then copy one of the other
  beams to replace this one and continue on. 

NEED BEAM OF AT LEAST 2 OR THIS WILL BREAK

'''
def beam_search(src_tensor,trg_tensor,num_decode_steps,
                model_to_test,SRC,TGT,eos_ind,beam_size=5):

  eos_ind = dataset_dict['tgt_eos_ind']
  tgt_padding_ind = dataset_dict['tgt_padding_ind']
  pred_tensor = []
  
  for s_num in range(trg_tensor.shape[1]): #loop over one sentence at a time
    src_tensor_input = src_tensor[:,s_num].view(-1,1).repeat(1,5).long().to(main_params.device)
    src_key_padding_mask = (src_tensor_input==dataset_dict['src_padding_ind']).transpose(0,1) #need it to be (N,S) and is (S,N)
                
    finished_beams = [] #is [[beam_tokens,ave log prob per token], ..], once output EOS then sentence added to this
    dec_input = (torch.ones(beam_size).view(1,-1)*trg_tensor[0,0].item()).long().to(main_params.device)
    cur_sum_log_probs = torch.zeros(beam_size).view(-1,1)
    encoder_output = None #since after we get encoder output, should reuse since stays same for the sentence
    
    
    for decoding_step in range(num_decode_steps): 
      print('src shape: {}, tar_shape: {}, src_mask: {}'.format(src_tensor_input.shape,dec_input.shape,src_key_padding_mask.shape))
      #print('src: {}, tar: {}, src_mask: {}'.format(src_tensor_input.cuda(),dec_input.cuda(),src_key_padding_mask.cuda()))
      if not encoder_output is None:
        print('encoder_output: ',encoder_output.cuda())
      output,encoder_output = model_to_test.forward(src_tensor_input,dec_input,
                                              src_key_padding_mask=src_key_padding_mask,
                                              tgt_mask=None,tgt_key_padding_mask=None,
                                              memory_key_padding_mask=src_key_padding_mask,memory=encoder_output)
      
      output = output.cpu()
      #Now make prediction on next word, note: output[decoder_step,:,:] has dim (batch_size,vocab_size)
      word_rankings = F.log_softmax(output[decoding_step,:,:],dim=1)
      #now for each, want to add cur_sum_log_probs to word_rankings
      word_rankings += cur_sum_log_probs #this gets sum of log probs of each sentence so far (so now find top ones)
      
      #now sort word_rankings then find top beams
      vals2,indices2 = torch.sort(word_rankings,dim=1,descending=True) 

      #now can just iterate over these tensors since only need to find usually like the top 5
      cur_inds = [0]*beam_size #keeps track of index we're at in each of the word_rankings vectors
      new_dec_input = []
      for b in range(beam_size):
        max_val = -1 #max sum log probs in this iteration
        beam_ind = 0 #the beam number which contains the max val this iter
        word_ind = -1 #word that gives max sum log probs this iter
        for b_num in range(beam_size):
          if vals2[b_num,cur_inds[b_num]] > max_val:
            max_val = vals2[b_num,cur_inds[b_num]]
            beam_ind = b_num
            word_ind = indices2[b_num,cur_inds[b_num]]
        
        #now have found best beam this iteration so inc cur_inds
        cur_inds[beam_ind] += 1
        new_dec_input.append([x for x in dec_input[:,beam_ind]] + [word_ind])
        cur_sum_log_probs[b,0] = max_val
      
      #if any beams have finished, then put them in finished_beams and replace
      #find one which hasn't finished
      ind_not_finished = -1
      for b in range(len(new_dec_input)):
        if new_dec_input[b][-1] != eos_ind:
          ind_not_finished = b
          break
      
      for b in range(len(new_dec_input)):
        if new_dec_input[b][-1] == eos_ind:
          finished_beams.append([[x for x in new_dec_input[b]],cur_sum_log_probs[b,0]/len(new_dec_input[b])])
          if ind_not_finished != -1:
            new_dec_input[b] = [x for x in new_dec_input[ind_not_finished]]
            cur_sum_log_probs[b,0] = cur_sum_log_probs[ind_not_finished,0]
          
      if ind_not_finished == -1:
        #add all to finished_beams
        break

      dec_input = torch.tensor(new_dec_input).transpose(0,1).long().to(main_params.device)
    
    finished_beams.sort(key=lambda x: x[1], descending=True)
    pred_tensor.append([x for x in finished_beams[0][0]]) #NEED TO ADD PADDING IN HERE

  max_len = max([len(x) for x in pred_tensor])
  for i in range(len(pred_tensor)):
    pred_tensor[i] += [tgt_padding_ind for x in range(max_len-len(pred_tensor[i]))]
  
  return torch.tensor(pred_tensor).tranpose(0,1)



def greedy_search(src_tensor,trg_tensor,num_decode_steps,src_key_padding_mask,SRC,TGT,model_to_test):
  dec_input = trg_tensor[0,:].view(1,-1)
  encoder_output = None #since after we get encoder output, should reuse since stays same for the sentence
  num_decode_steps = trg_tensor.shape[0]+10 #EVENTUALLY REMOVE THIS
  for decoding_step in range(num_decode_steps): 
    
    #TESTING REMOVE WHEN NOT TESTING
    #if decoding_step >= trg_tensor.shape[0]-1:
    #  break
    #dec_input = trg_tensor[:(decoding_step+1),:] #TEACHER FORCING to see how good our outputs are

    output,encoder_output = model_to_test.forward(src_tensor,dec_input,src_key_padding_mask=src_key_padding_mask,
                      tgt_mask=None,tgt_key_padding_mask=None,
                      memory_key_padding_mask=src_key_padding_mask,memory=encoder_output)
    #print('encoder output: ',encoder_output[:2,:3,:3])
    #print('shape output: ',output[decoder_step,:,:].shape)
    '''
    Now make prediction on next word
    we're only interested in the values of the embeddings of the current 
    step we're on which we can get by output[decoder_step,:,:]
    which has dim (batch_size,vocab_size) and now get 
    Using log_softmax since better numerical properties
    '''
    word_rankings = F.log_softmax(output[decoding_step,:,:],dim=1)
    #IS THIS WORD_RANKINGS CALC CORRECT?
    #print('word_rankings.shape: ',word_rankings.shape)
    #NOW SUMS OVER DIM 1 SHOULD BE 1
    #sums_dim1 = torch.exp(word_rankings).sum(dim=1)
    #print('sums_dim1 shpae: {}, first 5 els: {}'.format(sums_dim1.shape,sums_dim1[:5]))

    #now just go greedy and choose the highest for now
    #get indices along dimension 1 which have highest value (highest probability word)
    vals, indices = torch.max(word_rankings, 1) 
    vals2,indices2 = torch.sort(word_rankings,dim=1,descending=True) 
    #print('True input so far: ',[SRC.vocab.itos[ind2] for ind2 in src_tensor[:(decoding_step+1),0]])
    #print('True output so far: ',[TGT.vocab.itos[ind2] for ind2 in trg_tensor[:(decoding_step+1),0]])
      
    #print('True output word: ',TGT.vocab.itos[trg_tensor[decoding_step+1,0]])
    #print('Our prob of true output: ',torch.exp(torch.tensor([word_rankings[0,trg_tensor[decoding_step+1,0]]])))
    top10 = [TGT.vocab.itos[indices2[0,ind2]] for ind2 in range(10)]
    #print('Top 10 preds: ',top10)
    #print('probs of preds: ',torch.exp(vals2[0,:10]))
    #print()
    #print('shape vals: {}, first 5 vals: {}'.format(vals.shape,torch.exp(vals[:5])))
    #print('shape word_rankings: {}, shape indices: {}'.format(word_rankings.shape,indices.shape))
    #now indices contain our next predicted word for each sentence so add this as row to our dec_input
    #this isn't most efficient way but test set small so doesn't matter
    dec_input = torch.cat([dec_input,indices.view(1,-1)],dim=0)
  
  return dec_input

#dataset_type is either val or test
def get_policy_ave_bleu(model_to_test,dataset_dict,dataset_type,device,num_decode_steps):
  #print('IN GET_POLIC AVE')
  #print('weigths: ',model_to_test.linear.weight[:5])   
  model_to_test.eval()
  bleu_scores = []
  TGT = dataset_dict['TGT']
  SRC = dataset_dict['SRC']
  
  #need to get a list of predicted untokenized sentences and list of reference sentences
  pred_sentences = []
  true_sentences = []
  with torch.set_grad_enabled(False):
      dataset_iterator = dataset_dict[dataset_type+'_iter']
      #print('num valid batches: ',len(dataset_iterator))
      for batch in dataset_iterator:
          start_time = time.time()
          src_tensor = vars(batch)['de'].to(device) # is shape (S,N) where N batch size, S:largestnum tokens in batch
          trg_tensor = vars(batch)['en'].to(device) #transfer onto GPU
          
          #create masks (Here don't need target masks since outputting 1 word at a time.
          src_key_padding_mask = (src_tensor==dataset_dict['src_padding_ind']).transpose(0,1) #need it to be (N,S) and is (S,N)
          
          '''
          start by feeding decoder only start of sentence tokens for each
          sentence in batch
          '''                          
          dec_input = greedy_search(src_tensor=src_tensor,trg_tensor=trg_tensor,num_decode_steps=num_decode_steps,
                                    src_key_padding_mask=src_key_padding_mask,SRC=SRC,TGT=TGT,
                                    model_to_test=model_to_test)
          
          #dec_input = beam_search(src_tensor,trg_tensor,num_decode_steps,
          #      model_to_test,SRC,TGT,dataset_dict['tgt_eos_ind'])
          
          
          #now convert dec_input to sentences and then compare with trg_tensor when also converted to sentences
          #now go column by column getting BLEU scores: 
          #bleu_scores.append(get_bleu_scores(trg_tensor,dec_input,dataset_dict['TGT']).mean())
          #print('time for 1 batch: ',time.time()-start_time)
          preds,targs = getPredAndTargSentences(trg_tensor,dec_input,dataset_dict['TGT'])
          pred_sentences += preds
          true_sentences += targs


  return sacrebleu.corpus_bleu(pred_sentences, [true_sentences],use_effective_order=True).score
  
  #if dataset_type == 'test':
  #  print_test_summary(bleu_scores)
  
  #return np.mean(bleu_scores)


class LossHistory:
    def __init__(self):
      self.val_losses_this_epoch = [] #every 500 batches we will print our results in that last 500
      self.train_losses_this_epoch = []
      self.test_losses_this_epoch = []
      self.train_num_toks_this_epoch = 0
      self.val_num_toks_this_epoch = 0
      self.train_losses = [1000000]
      self.val_losses = [1000000] #starting point
      self.num_batches_processed = 0

    def add_batch_loss(self,this_loss,num_tokens, loss_type): #loss_type is in ['train','valid','test']
      np_ver_of_loss = this_loss if (isinstance(this_loss,np.float32) or isinstance(this_loss,np.float64) or isinstance(this_loss,float)) else this_loss.cpu().detach().numpy()
      #have to bring to cpu then detach from computation graph
      num_tokens = num_tokens.cpu().numpy() if num_tokens >0 else -1
      if loss_type == 'train':
        self.train_losses_this_epoch.append(np_ver_of_loss)
        self.num_batches_processed += 1
        self.train_num_toks_this_epoch += num_tokens
      elif loss_type == 'valid':
        self.val_losses_this_epoch.append(np_ver_of_loss)
        self.val_num_toks_this_epoch += num_tokens
      elif loss_type == 'test':
        self.test_losses_this_epoch.append(np_ver_of_loss)
      else:
        print('Unknown loss_type: ',loss_type)
        assert(0)
      
    def add_epoch_loss(self,epoch):
      if self.train_num_toks_this_epoch < 0:
        self.train_losses.append(np.mean(self.train_losses_this_epoch))
        self.val_losses.append(np.mean(self.val_losses_this_epoch))
      else:
        self.train_losses.append(np.sum(self.train_losses_this_epoch)/self.train_num_toks_this_epoch)
        self.val_losses.append(np.sum(self.val_losses_this_epoch)/self.val_num_toks_this_epoch)
      self.val_losses_this_epoch = [] #every 500 batches we will print our results in that last 500
      self.train_losses_this_epoch = []
      self.num_batches_processed = 0
      self.train_num_toks_this_epoch = 0
      self.val_num_toks_this_epoch = 0
      print('Epoch: {}, train_loss: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(
                              epoch, self.train_losses[-1], self.val_losses[-1],np.exp(-1*self.val_losses[-1])))

#want to return average loss per token
def masked_cross_entropy(real,pred,loss_func):
  mask = ~((real==dataset_dict['tgt_padding_ind'])) # | (real==dataset_dict['tgt_eos_ind'])) #I THINK WE ACTUALLY DO WANT TO PREDICT EOS
  
  #now 

  loss_ = loss_func(pred,real) #loss func does no reduction, so want to mask entire thing afterwards
  
  #print('pred Shape: {}, real shape: {}, lossShape: {}, lossmean: {}'.format(pred.shape,real.shape,loss_.shape,loss_.mean().shape))
  #pred Shape: torch.Size([16, 7787, 64]), real shape: torch.Size([16, 64]), lossShape: torch.Size([16, 64])
  
  loss_ *= mask

  return loss_.sum(), mask.sum() #just gives us average loss in the whole matrix



class MainParams:
    def __init__(self,dropout,src_vocab_size,tgt_vocab_size,batch_size):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        print(self.device)
        self.model_params = dict(d_model=128,nhead=8,num_encoder_layers=4, num_decoder_layers=4,
                dim_feedforward=512, dropout=0.2,activation='relu',src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size)

        self.batch_size = batch_size
        self.num_decode_steps = 60 #MAX NUMBER OF DECODING STEPS WE'LL do (can increase this)
