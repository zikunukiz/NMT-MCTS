

'''

Here we have disjoint architecture for policy and value network to see difference in performance.

Here the multi processing strategy is to have a Queue which the 
subprocesses send their information to, then the subprocess uses an
irecv to wait for a response from main, and main does a isend
to each subprocess that it has something to send to. 
Note: The main process doesn't have to wait for the send to be 
received. 
'''

import pyximport
pyximport.install()

import torch.distributed as dist
from torch.multiprocessing import Process
from torch.multiprocessing import Queue
import torch
import os
import globalsFile
from load_data import createIterators
from policy_net_disjoint import *
from mcts_translator import MCTS
import time
import json
from dataBuffer import DataBuffer
from lossAndTestClasses import *


def run(rank,maxlen,main_params,trg_tensor,queue,queue_exit):
	
	mcts = MCTS(tgt_tensor=trg_tensor,group=None,rankInGroup=rank,
				max_len=maxlen,main_params=main_params,queue=queue)


	#here actions is list of actions corresponding to the 
	#200 probabilities in mcts_probs
	bleu, output_states, mcts_probs,actions = mcts.translate_sentence()
	#write to file
	fileName = globalsFile.CODEPATH+'MCTSFiles/rank'+str(rank)+'.json'
	with open(fileName,'w') as f:
		json.dump([bleu,output_states,mcts_probs,actions],f)

	
	queue_exit.put([rank])
	#print('rank: ',rank, ' is EXITING')
	exit(0) #check that this doesn't shut down others
	


#this is where the main process runs
def main_func(numProcesses,src_tensor,maxlen,model,queue,queue_exit):

    starting_time = time.time()

    #RUN MODEL WITH ENTIRE SOURCE TENSOR ONCE TO GET 
    #ENCODER OUTPUT:
    #from then on we'll be using slices of it
    model.policy_encoder_output = None #want new output corresponding to this source tensor
    model.value_encoder_output = None
    
    dec_input = (torch.ones(numProcesses-1)*globalsFile.BOS_WORD_ID).long().view(1,-1)
    processes = torch.tensor(np.arange(1,numProcesses)).long()
    dec_lengths = torch.ones(numProcesses-1).long()
    log_probs, values = model.forward(src_tensor,dec_input,processes=processes,
                                        sentence_lens=dec_lengths,req_grad=False)

    #print('done setting encoder output from main')

    exited_processes = []
    while(True):

        #first receive tensors from subprocesses in queue
        recv_tensors = []
        start_time=time.time()
        while(True):
            if not queue.empty():
                rec_tens = queue.get()
                recv_tensors.append(rec_tens)
            
            #SHOULD HAVE time waiting dependent on number of exited processes
            if len(exited_processes) > (numProcesses-1)-3: #means less than or equal to 2 processes left so just always run
                break
            elif time.time()-start_time>0.1:
                break

        if len(recv_tensors) > 0:
            #first get list of lengths and processes these came from
            
            dec_lengths = torch.tensor([x[-2] for x in recv_tensors]).long()
            processes = torch.tensor([x[-1].item() for x in recv_tensors]).long()

            #trim them down to the maximum length of all gathered when remove padding
            max_gathered_len = int(dec_lengths.max().item())
            assert(max_gathered_len > 0)
            #print('RECEIVED TENSORS: ',recv_tensors[0])
            dec_input = torch.cat([x[:max_gathered_len].view(-1,1) for x in recv_tensors],1).long()
            #print('RECEIVED TENSORS: ',recv_tensors[0][:max_gathered_len])
            src_slice = src_tensor[:,(processes-1)] #take processes-1 since subprocesses start at rank=1
            #need to rearrange columns of src_tensor to work with 
            #this input
            #now encoder output, should do one initial run with all the sentences
            #then use slices thereafter.  

            #mask for decoder_input happens within this function
            act_probs, values = model.forward(src_slice,dec_input,processes=processes,
                                            sentence_lens=dec_lengths,req_grad=False)

            
            #need to get top model.num_children probs and their corresponding actions
            #which are the indices
            #print('log probs shape: ',log_probs.shape)
            act_probs = act_probs.to('cpu')
            values= values.to('cpu')
            sorted_probs,inds = torch.sort(act_probs,dim=1,descending=True)
            inds = inds.double() #so that concat with probs and values

            #now stack sorted_probs under inds then put value underneath, (this is what 
            #the other processes are expecting as format)
            to_scatter = torch.cat([inds[:,:model.num_children].transpose(0,1),
                                sorted_probs[:,:model.num_children].transpose(0,1),
                                values.unsqueeze(0)], dim=0)

            to_scatter = list(np.split(to_scatter,to_scatter.shape[1],axis=1))

            #need to clone compoennets so that they don't share memory
            to_scatter = [t.clone().squeeze(1) for t in to_scatter]

            #now send each of these to processes
            for i,process in enumerate(processes):
                dist.isend(tensor=to_scatter[i].clone(),dst=int(process))


        #now check for processes having exited
        if not queue_exit.empty():
            exited_processes.append(queue_exit.get())

        #print(len(exited_processes))
        if len(exited_processes) == numProcesses-1:
            #print('everyones exited')
            #print(time.time()-starting_time2)
            model.value_net = model.value_net.to('cpu')
            model.policy_net = model.policy_net.to('cpu')
            exit(0)


def init_processes(rank,numProcesses,src_tensor,trg_tensor,modelToPass,main_params,queue,queue_exit):
    #using spawn_processes passes process index as first param

    #create the processes here then have our workers call one function
    #and our main process (rank=0) call another
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29504'
    #print('rank: {}, numProcesses: {}'.format(rank,numProcesses))
    dist.init_process_group('gloo', rank=rank, world_size=numProcesses)
    #groupMCTS = dist.new_group([i for i in range(numProcesses)])

    maxlen = trg_tensor.shape[0] + 5 #max number of decode steps/ length in MCTS
    if rank == 0:
        main_func(numProcesses,src_tensor,maxlen,modelToPass,queue,queue_exit)
    else:
        run(rank,maxlen,main_params,trg_tensor[:,rank-1],queue,queue_exit)




if __name__=='__main__':

    torch.multiprocessing.set_start_method('spawn')
    #now create data_set iterators
    dataset_dict = createIterators(globalsFile.BATCHSIZE,globalsFile.DATAPATH)
        
    dataset_dict_for_val = createIterators(128, globalsFile.DATAPATH) #this will allow us to be faster on validation 
    eng_vocab = dataset_dict['TGT'].vocab.itos

    src_vocab_size = len(dataset_dict['SRC'].vocab.itos)
    
    hyperParamCombos = []
    for temperature in [1e0, 10**0.2, 1e-1]: #1e-1 exagerates all the percentages(high N because almost prob 1)
        for adamlr in [1e-4,3e-5]:
            for c_puct in [0.1,1,0.5,0.1]:
                hyperParamCombos.append([temperature,adamlr,c_puct])
    
    
    for hyper_params in hyperParamCombos: 
        temperature,adamlr,c_puct = hyper_params
        
        main_params = MainParams(dropout=0.2, src_vocab_size=src_vocab_size,
                      tgt_vocab_size=len(eng_vocab), batch_size=256,l2_const=0,
                      c_puct=c_puct,num_sims=100,temperature=temperature,
                      tgt_vocab_itos=dataset_dict['TGT'].vocab.itos,
                      num_children=50,is_training=False,
                      num_grad_steps_per_epoch=8,
                      adamlr=adamlr)


        policy_path = globalsFile.MODELPATH + 'policy_supervised_RLTrained.pt'
        value_path = globalsFile.MODELPATH + 'value_supervised_RLTrained.pt'
        #policy_path = None
        #value_path = None #want to use cpu for now
        network = PolicyValueNet(main_params=main_params,path_to_policy=policy_path,
                                path_to_value=value_path,adamlr=main_params.adamlr)

        
        '''
        bleu= get_policy_ave_bleu(network.policy_net,dataset_dict_for_val,'test',network.device,num_decode_steps=-1,useBLEU1=False)
        print('Test BLEU:',bleu)
        exit()
        '''
        '''
        bleu= get_policy_ave_bleu(network.policy_net,dataset_dict_for_val,'val',network.device,num_decode_steps=-1,useBLEU1=False)
        print('validation BLEU:',bleu)
        
        exit()
        '''
        
        print('loaded network')
        loss_history = LossHistory()
        loss_history.val_losses = [0] 
        
        for epoch in range(1000): #number iterations over dataset
            data_buffer = DataBuffer() #reset this since will now hopefully get more accurate traversals
            
            data_iter = 'train_iter'
            sentences_processed = 0
            for batch_count, batch in enumerate(dataset_dict[data_iter]):
                #print('src shape: {}, tgt shape: {}'.format(vars(batch)['de'].shape,vars(batch)['en'].shape))

                src_tensor = vars(batch)['de'] #will be (S,batch_size) where S is max num tokens of sentence in this batch
                trg_tensor = vars(batch)['en'] 
                
                #REMOVE THIS
                #if src_tensor.shape[0] > 8:
                #    continue
                if src_tensor.shape[0] > 50:
                    continue
                
                sentences_processed += src_tensor.shape[1]

                
                main_params.is_training = True if 'train' in data_iter else False

                #now create processes to use this
                size = src_tensor.shape[1]+1 #one process per sentence and one to do computation
                #print('sentence len: ',src_tensor.shape[0])
                starting_time = time.time()

                queue = Queue()
                queue_exit = Queue()
                ps = []
                for rank in range(size):
                    model = network if rank==0 else None
                    args =(rank,size,src_tensor,trg_tensor,model,main_params,queue,queue_exit)
                    p = Process(target=init_processes,args=args)
                    p.start()
                    ps.append(p)

                ps[0].join()

                #print('sentence len: ',src_tensor.shape[0])
                print('sentence_len: {}, totalTime: {}'.format(src_tensor.shape[0],time.time()-starting_time))

                #print('Finished BATCH')

                for rankNum in range(1,size):
                    fileName = globalsFile.CODEPATH+'/MCTSFiles/rank'+str(rankNum)+'.json'
                    with open(fileName) as f:
                        data = json.load(f)
                        #[bleu,output_states,mcts_probs,actions]
                        src_ = src_tensor[:,rankNum-1]
                        data_buffer.add_examples(src_,data)

                #print('sent all to files')
                #exit()
                #break

                if sentences_processed >= 256:
                    sentences_processed = 0
                    break


            #now do some updates to the model
            #HOW MANY TRAINING STEPS/SIZE TO DO? 
            data_buffer.set_iterator(batch_size=main_params.batch_size,shuffle=True)
            for grad_step in range(main_params.num_grad_steps_per_epoch):
                src_,dec_,actions,probs,bleus = data_buffer.next()	
                if src_ is None:
                    break

                loss = network.train_step(src_,dec_,probs,actions,bleus)
                loss_history.add_batch_loss(loss,-1, 'train')
            
            network.policy_encoder_output = None
            network.value_encoder_output = None
            
            #get validation loss
            print('Getting validation loss')
            bleu = get_policy_ave_bleu(network.policy_net,dataset_dict_for_val,'val',network.device,num_decode_steps=-1,useBLEU1=False)
            #print('BLEU: ',bleu)
            loss_history.add_batch_loss(bleu,-1, 'valid')
            loss_history.add_epoch_loss(epoch)
            
            if loss_history.val_losses[-1] >= np.max(loss_history.val_losses):
                #SAVE MODEL
                print('best model saving')
                policy_model_file = globalsFile.MODELPATH + 'policyMCTSDisjoint.pt'
                value_model_file = globalsFile.MODELPATH + 'valueMCTSDisjoint.pt'
                
                network.save_model(policy_model_file, value_model_file)
            
            #else: #didn't improve this iteration
            #    main_params.num_grad_steps_per_epoch = 4
            #    main_params.num_sims = 200
            
            if np.max(loss_history.val_losses) > np.max(loss_history.val_losses[-5:]):
                print('FINISHING: best bleu: {}, hyper params: {}'.format(np.max(loss_history.val_losses), hyper_params))
                break
            
            
            
            
            
            
            
