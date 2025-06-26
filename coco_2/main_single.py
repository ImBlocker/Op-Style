from data_process import state_get, highaction_get, lowaction_get, middleaction_get, get_datasets_single, data_get
from networks import HighActionNetwork, LowActionNetwork_simple, LowActionNetwork_complex, LowActionNetwork_simplest, \
    MiddleActionNetwork
from trainer import HighActionModelTrainer, LowActionModelTrainer, MiddleActionModelTrainer
import torch
import torch.nn as nn
import numpy
import copy
"""
barrack
input: state(27) + pos(1)
action1: [a1, a2, a3, a4, a5, a6]; 
action2:  [a1, a2, a3, a4]; 
"""

lr = 0.0001
lr2 = 0.0002
channels = 28 # 27 state + 1 pos
dim = 16

type_num = 6
turn_num = 4

test_size = 0.2

epoch_num1 = 5
epoch_num = 500

state_file_path = '../data_single/state_single_demo.csv'
pos_file_path = '../data_single/pos_single_demo.csv'
turn_file_path = '../data_single/turn_single_demo.csv'
type_file_path = '../data_single/type_single_demo.csv'

train_type_action_model = False
train_turn_action_model = True
def train():
    state_numpy = state_get(state_file_path, pos_file_path=pos_file_path)

    if (train_type_action_model == True):
        train_batch = 64
        test_batch = 64
        # 数据集
        type_action_numpy = data_get(type_file_path)
        train_loader, test_loader = get_datasets_single(state_numpy,
                                                         type_action_numpy,
                                                         test_size=test_size,
                                                         train_batch=train_batch,
                                                         test_batch=test_batch)

        # typeaction网络
        TypeAction_model = HighActionNetwork(n_channels=channels,
                                             in_dim=dim,
                                             intermediate_size=256,
                                             action_num=type_num)

        # 训练
        action_optimizer = torch.optim.Adam(TypeAction_model.parameters(), lr=lr)
        TypeActionTrainer = HighActionModelTrainer(TypeAction_model, train_loader, test_loader, action_optimizer, dim,
                                                   channels, type_num)
        TypeActionTrainer.train(epoch_num, epoch_record=1)
        print('end')

    if (train_turn_action_model == True):
        train_batch = 64
        test_batch = 64
        # 数据集
        turn_action_numpy = data_get(turn_file_path)
        train_loader, test_loader = get_datasets_single(state_numpy,
                                                        turn_action_numpy,
                                                        test_size=test_size,
                                                        train_batch=train_batch,
                                                        test_batch=test_batch)

        # typeaction网络
        TurnAction_model = HighActionNetwork(n_channels=channels,
                                             in_dim=dim,
                                             intermediate_size=256,
                                             action_num=turn_num)

        # 训练
        action_optimizer = torch.optim.Adam(TurnAction_model.parameters(), lr=lr)
        TurnActionTrainer = HighActionModelTrainer(TurnAction_model, train_loader, test_loader, action_optimizer, dim,
                                                   channels, turn_num)
        TurnActionTrainer.train(epoch_num, epoch_record=1)
        print('end')

def type_net_build(path):
    TypeAction_model = HighActionNetwork(n_channels=channels,
                                         in_dim=dim,
                                         intermediate_size=256,
                                         action_num=type_num)
    TypeAction_model.load_state_dict(torch.load(path))
    return TypeAction_model

def turn_net_build(path):
    TypeAction_model = HighActionNetwork(n_channels=channels,
                                         in_dim=dim,
                                         intermediate_size=256,
                                         action_num=turn_num)
    TypeAction_model.load_state_dict(torch.load(path))
    return TypeAction_model

def net_forward(output_row):
    output_argmax = torch.softmax(output_row, dim=0)
    output_class = torch.argmax(output_row).item()

    return output_argmax, output_class

def forward_compute(state_merge_pos_input):
    TypeAction_model = type_net_build("results_type_model/type_action_model_99.pth")
    type_output_row = TypeAction_model.forward(state_merge_pos_input).squeeze()
    type_output_argmax, type_output_class = net_forward(type_output_row)

    type_action = type_output_argmax

    TurnAction_model = turn_net_build("results_turn_model/turn_action_model_99.pth")
    turn_output_row = TurnAction_model.forward(state_merge_pos_input).squeeze()
    turn_output_argmax, turn_output_class = net_forward(turn_output_row)

    turn_action = turn_output_argmax

    return type_action, turn_action



def main(state_, pos):

    state = copy.deepcopy(state_)
    #print("state=",state.shape)
    #print("pos=", pos)
    """
    state: numpy [16,16,27]
    pos: int
    """
    state1 = numpy.swapaxes(state, 1, 2)
    state2 = numpy.swapaxes(state1, 0, 1)
    state_input = torch.tensor(state2).unsqueeze(dim=0) # 1, 27, 16, 16

    pos_input = pos
    i_2D = pos_input // dim
    j_2D = pos_input % dim
    pos_2D = torch.zeros((1, 1, dim, dim))
    pos_2D[0][0][i_2D][j_2D] = 1

    state_merge_pos_input = torch.cat((state_input, pos_2D), dim=1) # 1, 28, 16, 16
    # print('state_merge_pos_input_shape:', state_merge_pos_input.shape)
    owner_type_tensor = state_input[0][:, i_2D, j_2D] # 27
    # print('owner_type_tensor_shape:', owner_type_tensor.shape)
    type_action, turn_action = forward_compute(state_merge_pos_input)
    if torch.is_tensor(type_action):
        type_action = type_action.tolist()
    if torch.is_tensor(turn_action):
        turn_action = turn_action.tolist()

    return type_action, turn_action
train()