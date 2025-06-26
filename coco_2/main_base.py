from data_process import state_get, highaction_get, lowaction_get, get_datasets_base
from networks import HighActionNetwork, LowActionNetwork_simple, LowActionNetwork_complex, LowActionNetwork_simplest
from trainer import HighActionModelTrainer, LowActionModelTrainer
import torch
import torch.nn as nn

"""
base
input: state(27)
high_action: [a1, a2, a3, a4]; a3=0, a4 =0
middle_action: no need
low_action: [a1, a2, a3, a4]
            when high action a2=1, use these high action to train a low_action_network

"""


lr = 0.0001
lr2 = 0.0002
channels = 27
dim = 16

high_action_num = 2
low_action_num = 4

test_size = 0.2

epoch_num1 = 5
epoch_num = 200

state_file_path = '../data/base/base_state.csv'
highaction_file_path = '../data/base/base_high_action.csv'
lowaction_file_path = '../data/base/base_low_action.csv'

train_high_action_model = False
train_low_action_model = True

state_numpy = state_get(state_file_path, pos_file_path=None) # no pos information

if (train_high_action_model == True):
    train_batch = 64
    test_batch = 64
    # 数据集
    high_action_numpy = highaction_get(highaction_file_path)
    train_loader, test_loader = get_datasets_base(state_numpy,
                                                  high_action_numpy,
                                                  test_size=test_size,
                                                  train_batch=train_batch,
                                                  test_batch=test_batch)

    # highaction网络
    HighAction_model = HighActionNetwork(n_channels=channels,
                                         in_dim=dim,
                                         intermediate_size=256,
                                         action_num=high_action_num)

    # 训练
    action_optimizer = torch.optim.Adam(HighAction_model.parameters(), lr=lr)
    HighActionTrainer = HighActionModelTrainer(HighAction_model, train_loader, test_loader, action_optimizer, dim,
                                               channels, high_action_num)
    HighActionTrainer.train(epoch_num1, epoch_record=1)
    print('end')

if (train_low_action_model == True):
    train_batch = 64
    test_batch = 64
    high_action_numpy = highaction_get(highaction_file_path)
    low_action_numpy = lowaction_get(lowaction_file_path)
    train_loader, test_loader = get_datasets_base(state_numpy,
                                                  low_action_numpy,
                                                  test_size=test_size,
                                                  train_batch=train_batch,
                                                  test_batch=test_batch,
                                                  action2=True,
                                                  high_action_numpy=high_action_numpy)
    LowAction_model = LowActionNetwork_complex(n_channels=channels,
                                               in_dim=dim,
                                               intermediate_size=256,
                                               action_num=low_action_num)
    action_optimizer = torch.optim.Adam(LowAction_model.parameters(), lr=lr)#, weight_decay=0.001)
    LowActionTrainer = LowActionModelTrainer(LowAction_model, train_loader, test_loader, action_optimizer, dim,
                                             channels, low_action_num, y=False)

    LowActionTrainer.train(epoch_num, network='0', epoch_record=1)
    print('end')


