
import torch
import settings
import torch.nn.functional as F
import numpy as np
import sacrebleu
import time
import globalsFile
import nltk

'''
Some functions here are for testing the model (getting test set BLEU score).
We also have the loss class and a parameter holding class here. 

'''

def fix_sentence(sentence, as_str=False):  
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
def get_bleu_scores(trg_tensor,pred_tensor,TGT,BLEU1=False):
  bleus_per_sentence = torch.zeros(trg_tensor.shape[1],requires_grad=False) 
  for col in range(trg_tensor.shape[1]): #each column contains sentence
    true_sentence = [TGT.vocab.itos[i] for i in trg_tensor[:,col] if TGT.vocab.itos[i] != globalsFile.BLANK_WORD][1:-1]
    pred_sentence = [TGT.vocab.itos[i] for i in pred_tensor[:,col] if TGT.vocab.itos[i] != globalsFile.BLANK_WORD]
    
    #print('Before: ')
    #print(true_sentence)
    #print(pred_sentence)
    #now also need to stop pred_sentence after first EOS_WORD outputted
    #also don't want to use BOS chars
    ind_first_eos = 0
    for tok in pred_sentence:
      if tok == globalsFile.EOS_WORD:
        break
      ind_first_eos += 1
    
    if ind_first_eos != 0:
      pred_sentence = pred_sentence[1:ind_first_eos] #this gets rid of EOS_WORD

    
    #now undo some of the weird tokenization
    if BLEU1:
      pred_sentence = fix_sentence(pred_sentence,as_str=False)
      true_sentence = fix_sentence(true_sentence,as_str=False)
      score = nltk.translate.bleu_score.sentence_bleu([true_sentence], pred_sentence, weights=(1, 0, 0, 0))
    
    else:
      pred_sentence = fix_sentence(pred_sentence,as_str=True)
      true_sentence = fix_sentence(true_sentence,as_str=True)
      #This bleu_score defaults to calculating BLEU-4 (not normal bleu) so change weights,
      #this change of weights gives BLEU based on 1-grams so normal bleu I believe
    
      score = sacrebleu.sentence_bleu(pred_sentence, true_sentence,smooth_method='exp').score
      score /= 100.0
    #score = score*len(true_sentence)
    
    
    #print(true_sentence)
    #print(pred_sentence)
    #print(score)
    #print()
    bleus_per_sentence[col] = score
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
      if tok == globalsFile.EOS_WORD:
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



def greedy_search(src_tensor,trg_tensor,num_decode_steps,src_key_padding_mask,SRC,TGT,model_to_test):
  dec_input = trg_tensor[0,:].view(1,-1)
  encoder_output = None #since after we get encoder output, should reuse since stays same for the sentence
  num_decode_steps = trg_tensor.shape[0]+10 #EVENTUALLY REMOVE THIS
  for decoding_step in range(num_decode_steps): 
    
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
    
    #now just go greedy and choose the highest for now
    #get indices along dimension 1 which have highest value (highest probability word)
    vals, indices = torch.max(word_rankings, 1) 
    vals2,indices2 = torch.sort(word_rankings,dim=1,descending=True) 
    
    dec_input = torch.cat([dec_input,indices.view(1,-1)],dim=0)
  
  return dec_input


#dataset_type is either val or test
def get_policy_ave_bleu(model_to_test,dataset_dict,dataset_type,device,num_decode_steps,useBLEU1=False):
  #print('IN GET_POLIC AVE')
  #print('weigths: ',model_to_test.linear.weight[:5])   
  model_to_test.eval()
  bleu_scores = []
  TGT = dataset_dict['TGT']
  SRC = dataset_dict['SRC']
  
  #need to get a list of predicted untokenized sentences and list of reference sentences
  pred_sentences = []
  true_sentences = []
  bleu_scores = []
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
          if useBLEU1:
            bleu_scores.append(get_bleu_scores(trg_tensor,dec_input,dataset_dict['TGT'],BLEU1=True).mean())
          
          else:
            #print('time for 1 batch: ',time.time()-start_time)
            preds,targs = getPredAndTargSentences(trg_tensor,dec_input,dataset_dict['TGT'])
            pred_sentences += preds
            true_sentences += targs

  if useBLEU1:
    return np.mean(bleu_scores)

  return sacrebleu.corpus_bleu(pred_sentences, [true_sentences],use_effective_order=True).score
  

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





