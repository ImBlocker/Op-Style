from data_process import state_get, highaction_get, lowaction_get, middleaction_get, get_datasets_barrack
from networks import HighActionNetwork, LowActionNetwork_simple, LowActionNetwork_complex, LowActionNetwork_simplest, \
    MiddleActionNetwork
from trainer import HighActionModelTrainer, LowActionModelTrainer, MiddleActionModelTrainer
import torch
import torch.nn as nn

"""
barrack
input: state(27) + pos(1)
high_action: [a1, a2, a3, a4]; a3=0, a4 =0
middle_action:  [a1, a2, a3, a4, a5]; a1=0, a2 =0
                when high_action a2=1, use these middle action to train a middle_action_network
low_action: [a1, a2, a3, a4]
            when middle action a3=1, use these middle action to train a low_action_network
            when middle action a4=1, use these middle action to train a low_action_network
            when middle action a5=1, use these middle action to train a low_action_network

"""

lr = 0.0001
lr2 = 0.0002
channels = 28 # 27 state + 1 pos
dim = 16

high_action_num = 2
middle_action_num = 3
low_action_num = 4

test_size = 0.2

epoch_num1 = 5
epoch_num = 100

state_file_path = '../data/barrack/barrack_state.csv'
pos_file_path = '../data/barrack/barrack_pos.csv'
highaction_file_path = '../data/barrack/barrack_high_action.csv'
middleaction_file_path = '../data/barrack/barrack_middle_action.csv'
lowaction_file_path = '../data/barrack/barrack_low_action.csv'

train_high_action_model = False
train_middle_action_model = True
train_low_action_model = False

state_numpy = state_get(state_file_path, pos_file_path=pos_file_path)

if (train_high_action_model == True):
    train_batch = 64
    test_batch = 64
    # 数据集
    high_action_numpy = highaction_get(highaction_file_path)
    train_loader, test_loader = get_datasets_barrack(state_numpy,
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

if (train_middle_action_model == True):
    train_batch = 256
    test_batch = 256
    high_action_numpy = highaction_get(highaction_file_path)
    middle_action_numpy = middleaction_get(middleaction_file_path, a4_info=False)
    train_loader, test_loader = get_datasets_barrack(state_numpy,
                                                     middle_action_numpy,
                                                     test_size=test_size,
                                                     train_batch=train_batch,
                                                     test_batch=test_batch,
                                                     action2=True,
                                                     high_action_numpy=high_action_numpy)
    MiddleAction_model = MiddleActionNetwork(n_channels=channels,
                                             in_dim=dim,
                                             intermediate_size=128,
                                             action_num=middle_action_num)
    action_optimizer = torch.optim.Adam(MiddleAction_model.parameters(), lr=0.002)#, weight_decay=0.001)
    MiddleActionTrainer = MiddleActionModelTrainer(MiddleAction_model, train_loader, test_loader, action_optimizer, dim,
                                                   channels, middle_action_num, double=False)

    MiddleActionTrainer.train(epoch_num, epoch_record=1)
    print('end')


def low_model_build(train_batch, test_batch, middle_action_numpy, low_action_numpy, action345, network):
    train_loader, test_loader = get_datasets_barrack(state_numpy,
                                                     low_action_numpy,
                                                     test_size=test_size,
                                                     train_batch=train_batch,
                                                     test_batch=test_batch,
                                                     action345=action345,
                                                     middle_action_numpy=middle_action_numpy)
    LowAction_model = LowActionNetwork_simplest(n_channels=channels,
                                                in_dim=dim,
                                                intermediate_size=128,
                                                action_num=low_action_num)
    action_optimizer = torch.optim.Adam(LowAction_model.parameters(), lr=lr, weight_decay=0.0001)
    LowActionTrainer = LowActionModelTrainer(LowAction_model, train_loader, test_loader, action_optimizer, dim,
                                             channels, low_action_num, y=False)

    LowActionTrainer.train(epoch_num, network=network)

def low_model_build_complex(train_batch, test_batch, middle_action_numpy, low_action_numpy, action345, network):
    train_loader, test_loader = get_datasets_barrack(state_numpy,
                                                     low_action_numpy,
                                                     test_size=test_size,
                                                     train_batch=train_batch,
                                                     test_batch=test_batch,
                                                     action345=action345,
                                                     middle_action_numpy=middle_action_numpy)
    LowAction_model = LowActionNetwork_simple(n_channels=channels,
                                                in_dim=dim,
                                                intermediate_size=128,
                                                action_num=low_action_num)
    action_optimizer = torch.optim.Adam(LowAction_model.parameters(), lr=0.0005)#, weight_decay=0.0001)
    LowActionTrainer = LowActionModelTrainer(LowAction_model, train_loader, test_loader, action_optimizer, dim,
                                             channels, low_action_num, y=False)

    LowActionTrainer.train(100, network=network, epoch_record=1)

if (train_low_action_model == True):
    train_batch = 64
    test_batch = 64
    middle_action_numpy = middleaction_get(middleaction_file_path, a4_info=False)
    low_action_numpy = lowaction_get(lowaction_file_path)
    low_model_build(train_batch, test_batch, middle_action_numpy, low_action_numpy,
                    action345=[True, False, False], network='1')
    low_model_build_complex(train_batch, test_batch, middle_action_numpy, low_action_numpy,
                            action345=[False, True, False], network='2')
    low_model_build_complex(train_batch, test_batch, middle_action_numpy, low_action_numpy,
                    action345=[False, False, True], network='3')



