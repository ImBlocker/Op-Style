from data_process import state_get, state_get_merge, highaction_get_merge, middleaction_get, lowaction_get, \
    get_datasets_worker
from networks import MiddleActionNetwork, LowActionNetwork_simple, LowActionNetwork_complex, HighActionNetwork, \
    LowActionNetwork_simplest
from trainer import MiddleActionModelTrainer, LowActionModelTrainer, HighActionModelTrainer
import torch
import torch.nn as nn

"""
worker: produce harvest attack
input: state(27) + pos(1)
high_action: [a1 a2 a3 a4], a1=0
produce(high a2=1):
middle_action: [a1 a2],
low_action: when middle action a1=1, use these middle action a4 to train a low_action_network
            when middle action a2=1, use these middle action a4 to train a low_action_network

harvest(high a3=1):
middle_action: [a1 a2 a3 a4],
low_action: when middle action a1=1, use these middle action a4 to train a low_action_network
            when middle action a2=1, use these middle action a4 to train a low_action_network
            when middle action a3=1, use these middle action a4 to train a low_action_network
            when middle action a4=1, use these middle action a4 to train a low_action_network

attack(high a4=1):
middle_action: [a1 a2 a3 a4], a1 a2 a3 is 0/1; a4 in {0,1,2,3,4,5,6}
low_action: when middle action a1=1, use these middle action a4 to train a low_action_network
            when middle action a2=1, use these middle action a4 to train a low_action_network
            when middle action a3=1, use these middle action a4 to train a low_action_network
"""


lr = 0.0001
lr2 = 0.0002
channels = 28 # 27 state + 1 pos
dim = 16

high_action_num = 3
middle_action_num = [2, 4, 10]
low_action_num = 4

test_size = 0.2

epoch_num = 500


produce_state_file_path = '../data/worker/produce/worker_state.csv'
produce_pos_file_path = '../data/worker/produce/worker_pos.csv'
produce_highaction_file_path = '../data/worker/produce/worker_high_action.csv'
produce_middleaction_file_path = '../data/worker/produce/worker_middle_action.csv'
produce_lowaction_file_path = '../data/worker/produce/worker_low_action.csv'

harvest_state_file_path = '../data/worker/harvest/worker_state.csv'
harvest_pos_file_path = '../data/worker/harvest/worker_pos.csv'
harvest_highaction_file_path = '../data/worker/harvest/worker_high_action.csv'
harvest_middleaction_file_path = '../data/worker/harvest/worker_middle_action.csv'
harvest_lowaction_file_path = '../data/worker/harvest/worker_low_action.csv'

attack_state_file_path = '../data/worker/attack/worker_state.csv'
attack_pos_file_path = '../data/worker/attack/worker_pos.csv'
attack_highaction_file_path = '../data/worker/attack/worker_high_action.csv'
attack_middleaction_file_path = '../data/worker/attack/worker_middle_action.csv'
attack_lowaction_file_path = '../data/worker/attack/worker_low_action.csv'

train_high_action_model = False
train_middle_action_model = False
train_low_action_model = True

state_numpy, produce_state, harvest_state, attack_state = state_get_merge(produce_state_file_path,
                                                                          harvest_state_file_path,
                                                                          attack_state_file_path,
                                                                          produce_pos_file_path,
                                                                          harvest_pos_file_path,
                                                                          attack_pos_file_path)

if (train_low_action_model == True):
    def data_load(state_numpy, train_batch, test_batch, middleaction_file_path, lowaction_file_path, action_TF, data_TF, y):
        middle_action_numpy = middleaction_get(middleaction_file_path, a4_info=False, worker=data_TF)
        low_action_numpy = lowaction_get(lowaction_file_path)
        train_loader, test_loader = get_datasets_worker(state_numpy,
                                                        low_action_numpy,
                                                        test_size=test_size,
                                                        train_batch=train_batch,
                                                        test_batch=test_batch,
                                                        actions=action_TF,
                                                        middle_action_numpy=middle_action_numpy,
                                                        y=y)
        return train_loader, test_loader

    def complex_model_build_33(train_loader, test_loader, network, y):
        LowAction_model = LowActionNetwork_simple(n_channels=channels,
                                                   in_dim=dim,
                                                   intermediate_size=1024,
                                                   action_num=low_action_num)

        action_optimizer = torch.optim.Adam(LowAction_model.parameters(), lr=0.001, weight_decay=0.001)
        LowActionTrainer = LowActionModelTrainer(LowAction_model, train_loader, test_loader, action_optimizer, dim,
                                                 channels, low_action_num, y=y)
        LowActionTrainer.train(100, network, epoch_record=1)

    def low_action_network_attack(state_numpy, middleaction_file_path, lowaction_file_path, y, network_num, data_TF):

        i = network_num - 1
        action_TF = [False, False, False, False]
        action_TF[i] = True
        train_loader, test_loader = data_load(state_numpy, 64, 64, middleaction_file_path, lowaction_file_path,
                                                  action_TF, data_TF, y)
        complex_model_build_33(train_loader, test_loader, network=str(i + 1), y=y)
        print('end')

    low_action_network_attack(attack_state, attack_middleaction_file_path, attack_lowaction_file_path, y=True,
                       network_num=3, data_TF=[False, False, True])