# -*- coding: utf-8 -*-
"""
@author: Jerry Zikun Chen
"""
from __future__ import print_function
from translate import Translate, Translation
from mcts_translator import MCTSTranslator
from policy_net import PolicyValueNet, MainParams
from load_data import createIterators
from collections import deque
import torch
import numpy as np
import random
import time


class TrainPipeline():
    def __init__(self, dataset_dict, vocab, init_params,
                  init_policy_model=None, init_value_model=None):

        self.n_playout = 100  # num of simulations for each move
        self.batch_size = 1  # mini-batch size for training
        self.play_batch_size = 1
        # self.translation_batch_num = 2 # number of translation to do
        self.n_avlb = 200 # number of nodes to search at each layer
        self.epochs = 1  # num of train_steps for each update

        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.c_puct = 5
        self.buffer_size = 10000
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.kl_targ = 0.02
        self.check_freq = 50
        self.device = init_params.device
        # self.best_win_ratio = 0.0
        
        if init_policy_model and init_value_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(main_params=init_params,
                                                   path_to_policy=init_policy_model,
                                                   path_to_value=init_value_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(main_params=init_params)
        self.mcts_translator = MCTSTranslator(self.policy_value_net,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_train=1)
        self.dataset_dict = dataset_dict
        self.vocab = vocab

    def run(self):
        """run the training pipeline"""
        # training
        dataset_type = "train"
        dataset_iterator = self.dataset_dict[dataset_type + '_iter']
        
        try:
            i = 0
            for batch in dataset_iterator:
                src = vars(batch)['de'].view(-1).tolist()
                tgt = vars(batch)['en'].view(-1).tolist()
                self.src = src
                self.tgt = tgt
                self.translation = Translation(self.src, self.tgt, self.n_avlb, self.vocab, self.device)
                self.translate = Translate(self.translation)
                
                start_time = time.time()
                self.collect_translation_data(self.play_batch_size)
                i += 1
                print("batch i: {}, episode_len: {}".format(
                        i, self.episode_len))
                print("total simulation time: {}".format(time.time() - start_time))
                if len(self.data_buffer) > self.batch_size:
                    print("=========== train network ===========")
                    policy_loss, entropy = self.policy_update()
                # TO DO check the performance of the current model,
                # and save the model params
                # if (i+1) % self.check_freq == 0:
                #     print("current self-play batch: {}".format(i+1))
                #     win_ratio = self.policy_evaluate()
                #     self.policy_value_net.save_model('./current_policy.model')
                #     if win_ratio > self.best_win_ratio:
                #         print("New best policy!!!!!!!!")
                #         self.best_win_ratio = win_ratio
                #         # update the best_policy
                #         self.policy_value_net.save_model('./best_policy.model')
                #         if (self.best_win_ratio == 1.0 and
                #                 self.pure_mcts_playout_num < 5000):
                #             self.pure_mcts_playout_num += 1000
                #             self.best_win_ratio = 0.0    
        except KeyboardInterrupt:
            print('\n\rquit')


    def collect_translation_data(self, n_translations=1): # collect_train_translation_data
        """collect self-play data for training"""
        for i in range(n_translations):
            bleus, translation_data = self.translate.start_train_translate(self.mcts_translator,
                                                          temp=self.temp)
            translation_data = list(translation_data)[:]
            self.episode_len = len(translation_data)
            # print(translation_data)            
            self.data_buffer.extend(translation_data)
            # break

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch] # list of tuple (src, output)
        mcts_probs_batch = [data[1] for data in mini_batch]
        bleu_batch = [data[2] for data in mini_batch]

        # print(mini_batch)
        # print(state_batch, mcts_probs_batch, bleu_batch)

        # state_batch = zip(*state_batch)
        # print(state_batch)
        src = state_batch[0][0]
        output = state_batch[0][1]
        src_tensor = torch.from_numpy(
                np.array(src).reshape(-1, 1)).to(self.device)
        output_tensor = torch.from_numpy(
                np.array(output).reshape(-1, 1)).to(self.device)
        old_probs, old_v, encoder_output = self.policy_value_net.policy_value(src_tensor, output_tensor)
        for i in range(self.epochs):
            print("train step")
            loss, entropy = self.policy_value_net.train_step(
                    src, output,
                    mcts_probs_batch,
                    bleu_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(src_tensor, output_tensor)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(bleu_batch) - old_v.flatten()) /
                             np.var(np.array(bleu_batch)))
        explained_var_new = (1 -
                             np.var(np.array(bleu_batch) - new_v.flatten()) /
                             np.var(np.array(bleu_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

if __name__ == '__main__':
    working_path = ''
    policy_pt = 'policy_supervised_RLTrained_BLEU1.pt'
    value_pt = 'value_supervised_RLTrained_BLEU1.pt'

    batch_size = 1
    dataset_dict = createIterators(batch_size, working_path + '')

    # English vocabulary
    eng_vocab = dataset_dict['TGT'].vocab.itos
    src_vocab_size = len(dataset_dict['SRC'].vocab.itos)
    tgt_vocab_size = len(eng_vocab)
    main_params = MainParams(dropout=0.2, src_vocab_size=src_vocab_size,
                  tgt_vocab_size=tgt_vocab_size, batch_size=batch_size)


    policy_file = working_path + policy_pt
    value_file = working_path + value_pt

    training_pipeline = TrainPipeline(dataset_dict, eng_vocab, init_params = main_params, 
                            init_policy_model=policy_file, init_value_model=value_file)
    training_pipeline.run()


