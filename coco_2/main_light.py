from data_process import state_get, middleaction_get, lowaction_get, get_datasets
from networks import MiddleActionNetwork, LowActionNetwork_simple, LowActionNetwork_complex
from trainer import MiddleActionModelTrainer, LowActionModelTrainer
import torch
import torch.nn as nn

"""
light
input: state(27) + pos(1)
high_action: no need 
middle_action: [a1 a2 a3 a4], a1 a2 a3 is 0/1; a4 in {0,1,2,3,4,5,6}
low_action: when middle action a1=1, use these middle action a4 to train a low_action_network
            when middle action a2=1, use these middle action a4 to train a low_action_network
            when middle action a3=1, use these middle action a4 to train a low_action_network

"""


lr = 0.0001
lr2 = 0.0002
channels = 28 # 27 state + 1 pos
dim = 16

middle_action_num = 10 # a1 a2 a3 (3) and one-hot a4(7)
low_action_num = 4

test_size = 0.2

epoch_num = 500

state_file_path = '../data/light/light_state.csv'
pos_file_path = '../data/light/light_pos.csv'
middleaction_file_path = '../data/light/light_middle_action.csv'
lowaction_file_path = '../data/light/light_low_action.csv'

train_middle_action_model = True
train_low_action_model = True

state_numpy = state_get(state_file_path, pos_file_path)

if (train_middle_action_model == True):
    train_batch = 64
    test_batch = 64
    # 数据集
    middle_action_numpy = middleaction_get(middleaction_file_path, a4_info=True)
    train_loader, test_loader = get_datasets(state_numpy,
                                             middle_action_numpy,
                                             test_size=test_size,
                                             train_batch=train_batch,
                                             test_batch=test_batch)

    # middleaction网络
    MiddleAction_model = MiddleActionNetwork(n_channels=channels,
                                             in_dim=dim,
                                             intermediate_size=256,
                                             action_num=middle_action_num)

    # 训练
    action_optimizer = torch.optim.Adam(MiddleAction_model.parameters(), lr=lr)
    MiddleActionTrainer = MiddleActionModelTrainer(MiddleAction_model, train_loader, test_loader, action_optimizer, dim,
                                                   channels, middle_action_num, double=True)
    MiddleActionTrainer.train(epoch_num)
    print('end')

if (train_low_action_model == True):
    def data_load(train_batch, test_batch, action_TF):
        middle_action_numpy = middleaction_get(middleaction_file_path, a4_info=True)
        low_action_numpy = lowaction_get(lowaction_file_path)
        train_loader, test_loader = get_datasets(state_numpy,
                                                 low_action_numpy,
                                                 test_size=test_size,
                                                 train_batch=train_batch,
                                                 test_batch=test_batch,
                                                 action123=action_TF,
                                                 middle_action_numpy=middle_action_numpy)
        return train_loader, test_loader


    def simple_model_build(train_loader, test_loader, network):
        LowAction_model = LowActionNetwork_simple(n_channels=channels,
                                                  in_dim=dim,
                                                  intermediate_size=128,
                                                  action_num=low_action_num)

        action_optimizer = torch.optim.Adam(LowAction_model.parameters(), lr=lr)
        LowActionTrainer = LowActionModelTrainer(LowAction_model, train_loader, test_loader, action_optimizer, dim,
                                                 channels, low_action_num, y=True)
        LowActionTrainer.train(epoch_num, network)


    def complex_model_build(train_loader, test_loader, network):
        LowAction_model = LowActionNetwork_complex(n_channels=channels,
                                                   in_dim=dim,
                                                   intermediate_size=256,
                                                   action_num=low_action_num)

        action_optimizer = torch.optim.Adam(LowAction_model.parameters(), lr=lr)
        LowActionTrainer = LowActionModelTrainer(LowAction_model, train_loader, test_loader, action_optimizer, dim,
                                                 channels, low_action_num, y=True)
        LowActionTrainer.train(epoch_num, network, epoch_record=1)


    train_loader, test_loader = data_load(1, 1, [True, False, False])
    simple_model_build(train_loader, test_loader, network='1')
    print('end')

    train_loader, test_loader = data_load(1, 1, [False, True, False])
    simple_model_build(train_loader, test_loader, network='2')
    print('end')

    train_loader, test_loader = data_load(64, 64, [False, False, True])
    complex_model_build(train_loader, test_loader, network='3')
    print('end')
