from coco_2.networks import HighActionNetwork, LowActionNetwork_simple, LowActionNetwork_complex, LowActionNetwork_simplest, \
    MiddleActionNetwork
import torch
import numpy
import copy

from coco_2.data_process import state_get

"""
15: base
16: barrack
17: worker
18: light
19: heavy
20:ranged
"""

dim = 16

def high_net_build(path, channels, intermediate_size, high_action_num):
    HighAction_model = HighActionNetwork(n_channels=channels,
                                         in_dim=dim,
                                         intermediate_size=intermediate_size,
                                         action_num=high_action_num)
    HighAction_model.load_state_dict(torch.load(path))
    return HighAction_model


def middle_net_build(path, channels, intermediate_size, middle_action_num):
    MiddleAction_model = MiddleActionNetwork(n_channels=channels,
                                             in_dim=dim,
                                             intermediate_size=intermediate_size,
                                             action_num=middle_action_num)
    MiddleAction_model.load_state_dict(torch.load(path))
    return MiddleAction_model


def low_net_simplest_build(path, channels, intermediate_size, low_action_num):
    LowAction_model = LowActionNetwork_simplest(n_channels=channels,
                                                in_dim=dim,
                                                intermediate_size=intermediate_size,
                                                action_num=low_action_num)
    LowAction_model.load_state_dict(torch.load(path))
    return LowAction_model


def low_net_simple_build(path, channels, intermediate_size, low_action_num):
    LowAction_model = LowActionNetwork_simple(n_channels=channels,
                                              in_dim=dim,
                                              intermediate_size=intermediate_size,
                                              action_num=low_action_num)
    LowAction_model.load_state_dict(torch.load(path))
    return LowAction_model


def low_net_complex_build(path, channels, intermediate_size, low_action_num):
    LowAction_model = LowActionNetwork_complex(n_channels=channels,
                                               in_dim=dim,
                                               intermediate_size=intermediate_size,
                                               action_num=low_action_num)
    LowAction_model.load_state_dict(torch.load(path))
    return LowAction_model


def net_forward(output_row):
    output_argmax = torch.softmax(output_row, dim=0)
    output_class = torch.argmax(output_row).item()

    return output_argmax, output_class

def forward_compute(owner_type_tensor, state_input, state_merge_pos_input):
    if (owner_type_tensor[15] == 1):  # base
        #print('type: base')
        channels = 27

        high_action_num = 2
        low_action_num = 4

        HighAction_model = high_net_build("coco_2/results_base/high_action_model_4.pth",
                                          channels,
                                          256,
                                          high_action_num)
        high_output_row = HighAction_model.forward(state_input).squeeze()
        high_output_argmax, high_output_class = net_forward(high_output_row)
        #print('high action: ', high_output_argmax, high_output_class)

        high_action = high_output_argmax
        middle_action = [0,0,0,0]


        if (high_output_class == 1):
            LowAction_model = low_net_complex_build("coco_2/results_base/low_action0_model_199.pth",
                                                    channels,
                                                    256, low_action_num)
            low_output_row = LowAction_model.forward_without_y(state_input).squeeze()
            low_output_argmax, low_output_class = net_forward(low_output_row)
            #print('low action: ', low_output_argmax, low_output_class)
            low_action = low_output_argmax
        else:
            pass
            low_action = [0,0,0,0]
            #print('no low action.')

    elif (owner_type_tensor[16] == 1):  # barrack
        #print('type: barrack')
        channels = 28  # 27 state + 1 pos

        high_action_num = 2
        middle_action_num = 3
        low_action_num = 4

        HighAction_model = high_net_build("coco_2/results_barrack/high_action_model_4.pth",
                                          channels,
                                          256,
                                          high_action_num)
        high_output_row = HighAction_model(state_merge_pos_input).squeeze()
        high_output_argmax, high_output_class = net_forward(high_output_row)
        #print('high action: ', high_output_argmax, high_output_class)
        high_action = high_output_argmax

        if (high_output_class == 1):
            MiddleAction_model = middle_net_build("coco_2/results_barrack/middle_action_model_99.pth",
                                                  channels,
                                                  128,
                                                  middle_action_num)
            middle_output_row = MiddleAction_model.forward(state_merge_pos_input).squeeze()
            middle_output_argmax, middle_output_class = net_forward(middle_output_row)
            #print('middle action: ', middle_output_argmax, middle_output_class)
            middle_action = middle_output_argmax

            if (middle_output_class == 0):
                LowAction_model = low_net_simplest_build("coco_2/results_barrack/low_action1_model_99.pth",
                                                         channels,
                                                         128,
                                                         low_action_num)
                low_output_row = LowAction_model.forward_without_y(state_merge_pos_input).squeeze()
                low_output_argmax, low_output_class = net_forward(low_output_row)
                #print('low action: ', low_output_argmax, low_output_class)
                low_action = low_output_argmax

            else:
                LowAction_model = low_net_simple_build(
                    "coco_2/results_barrack/low_action" + str(int(middle_output_class) + 1) + "_model_99.pth",
                    channels,
                    128,
                    low_action_num)
                low_output_row = LowAction_model.forward_without_y(state_merge_pos_input).squeeze()
                low_output_argmax, low_output_class = net_forward(low_output_row)
                #print('low action: ', low_output_argmax, low_output_class)
                low_action = low_output_argmax
        else:
            #print('no middle action.')
            middle_action = [0,0,0,0]
            low_action = [0,0,0,0]
    elif (owner_type_tensor[17] == 1):  # worker
        #print('type: worker')
        channels = 28  # 27 state + 1 pos

        high_action_num = 3
        middle_action_num = [2, 4, 10]
        low_action_num = 4

        HighAction_model = high_net_build("coco_2/results_worker/high_action_model_9.pth",
                                          channels,
                                          256,
                                          high_action_num)
        high_output_row = HighAction_model(state_merge_pos_input).squeeze()
        high_output_argmax, high_output_class = net_forward(high_output_row)
        #print('high action: ', high_output_argmax, high_output_class)
        high_action = high_output_argmax
        if (high_output_class == 0):
            MiddleAction_model = middle_net_build(
                "coco_2/results_worker/middle_action_model_" + str(int(high_output_class)) + "99.pth",
                channels,
                128,
                middle_action_num[high_output_class])

            middle_output_row = MiddleAction_model(state_merge_pos_input).squeeze()
            middle_output_argmax, middle_output_class = net_forward(middle_output_row)
            #print('middle action: ', middle_output_argmax, middle_output_class)
            middle_action = middle_output_argmax
            LowAction_model = low_net_simplest_build(
                "coco_2/results_worker/low_action0_" + str(int(middle_output_class) + 1) + "_model_499.pth",
                channels,
                128,
                low_action_num)
            low_output_row = LowAction_model.forward_without_y(state_merge_pos_input).squeeze()
            low_output_argmax, low_output_class = net_forward(low_output_row)
            #print('low action: ', low_output_argmax, low_output_class)
            low_action = low_output_argmax
        elif (high_output_class == 1):
            MiddleAction_model = middle_net_build(
                "coco_2/results_worker/middle_action_model_" + str(int(high_output_class)) + "99.pth",
                channels,
                128,
                middle_action_num[high_output_class])

            middle_output_row = MiddleAction_model(state_merge_pos_input).squeeze()
            middle_output_argmax, middle_output_class = net_forward(middle_output_row)
            #print('middle action: ', middle_output_argmax, middle_output_class)
            middle_action = middle_output_argmax
            LowAction_model = low_net_simplest_build(
                "coco_2/results_worker/low_action1_" + str(int(middle_output_class) + 1) + "_model_499.pth",
                channels,
                128,
                low_action_num)
            low_output_row = LowAction_model.forward_without_y(state_merge_pos_input).squeeze()
            low_output_argmax, low_output_class = net_forward(low_output_row)
            #print('low action: ', low_output_argmax, low_output_class)
            low_action = low_output_argmax
        else:
            MiddleAction_model = middle_net_build(
                "coco_2/results_worker/middle_action_model_" + str(int(high_output_class)) + "99.pth",
                channels,
                128,
                middle_action_num[high_output_class])

            middle_output_row = MiddleAction_model(state_merge_pos_input).squeeze()
            middle_output_argmax, middle_output_class = net_forward(middle_output_row[:3])
            #print('middle action: ', middle_output_argmax, middle_output_class)
            middle_action = middle_output_argmax
            y_middle = middle_output_row[:7].reshape(1, -1)

            if(middle_output_class != 2):
                LowAction_model = low_net_simplest_build(
                    "coco_2/results_worker/low_action2_" + str(int(middle_output_class) + 1) + "_model_499.pth",
                    channels,
                    128,
                    low_action_num)
            else:
                LowAction_model = low_net_simple_build(
                    "coco_2/results_worker/low_action2_" + str(int(middle_output_class) + 1) + "_model_499.pth",
                    channels,
                    128,
                    low_action_num)
            low_output_row = LowAction_model(state_merge_pos_input, y_middle).squeeze()
            low_output_argmax, low_output_class = net_forward(low_output_row)
            #print('low action: ', low_output_argmax, low_output_class)
            low_action = low_output_argmax

    elif (owner_type_tensor[18] == 1):  # light
        #print('type: light')
        channels = 28  # 27 state + 1 pos
        middle_action_num = 10  # a1 a2 a3 (3) and one-hot a4(7)
        low_action_num = 4

        MiddleAction_model = middle_net_build("coco_2/results_light/middle_action_model_499.pth",
                                              channels,
                                              256,
                                              middle_action_num)
        middle_output_row = MiddleAction_model(state_merge_pos_input).squeeze()
        middle_output_argmax, middle_output_class = net_forward(middle_output_row[:3])
        #print('middle action: ', middle_output_argmax, middle_output_class)
        high_action = [0,0,0,0]
        middle_action = middle_output_argmax
        y_middle = middle_output_row[:7].reshape(1, -1)

        if (middle_output_class == 0 or middle_output_class == 1):
            LowAction_model = low_net_simple_build(
                "coco_2/results_light/low_action" + str(int(middle_output_class) + 1) + "_model_499.pth",
                channels,
                128,
                low_action_num)
            low_output_row = LowAction_model(state_merge_pos_input, y_middle).squeeze()
            low_output_argmax, low_output_class = net_forward(low_output_row)
            #print('low action: ', low_output_argmax, low_output_class)
            low_action = low_output_argmax

        else:
            LowAction_model = low_net_complex_build("coco_2/results_light/low_action3_model_499.pth",
                                                    channels,
                                                    256,
                                                    low_action_num)
            low_output_row = LowAction_model(state_merge_pos_input, y_middle).squeeze()
            low_output_argmax, low_output_class = net_forward(low_output_row)
            #print('low action: ', low_output_argmax, low_output_class)
            low_action = low_output_argmax

    elif (owner_type_tensor[19] == 1):  # heavy
        #print('type: heavy')
        channels = 28  # 27 state + 1 pos
        middle_action_num = 10  # a1 a2 a3 (3) and one-hot a4(7)
        low_action_num = 4

        MiddleAction_model = middle_net_build("coco_2/results_heavy/middle_action_model_499.pth",
                                              channels,
                                              256,
                                              middle_action_num)
        middle_output_row = MiddleAction_model(state_merge_pos_input).squeeze()
        middle_output_argmax, middle_output_class = net_forward(middle_output_row[:3])
        #print('middle action: ', middle_output_argmax, middle_output_class)
        high_action = [0,0,0,0]
        middle_action = middle_output_argmax
        y_middle = middle_output_row[:7].reshape(1, -1)

        if (middle_output_class == 0):
            LowAction_model = low_net_simplest_build("coco_2/results_heavy/low_action1_model_499.pth",
                                                     channels,
                                                     28,
                                                     low_action_num)
            low_output_row = LowAction_model(state_merge_pos_input, y_middle).squeeze()
            low_output_argmax, low_output_class = net_forward(low_output_row)
            #print('low action: ', low_output_argmax, low_output_class)
            low_action = low_output_argmax
        elif (middle_output_class == 1):
            LowAction_model = low_net_simple_build("coco_2/results_heavy/low_action2_model_499.pth",
                                                   channels,
                                                   128,
                                                   low_action_num)
            low_output_row = LowAction_model(state_merge_pos_input, y_middle).squeeze()
            low_output_argmax, low_output_class = net_forward(low_output_row)
            #print('low action: ', low_output_argmax, low_output_class)
            low_action = low_output_argmax
        else:
            LowAction_model = low_net_complex_build("coco_2/results_heavy/low_action3_model_499.pth",
                                                    channels,
                                                    256,
                                                    low_action_num)
            low_output_row = LowAction_model(state_merge_pos_input, y_middle).squeeze()
            low_output_argmax, low_output_class = net_forward(low_output_row)
            #print('low action: ', low_output_argmax, low_output_class)
            low_action = low_output_argmax

    elif (owner_type_tensor[20] == 1):  # ranged
        #print('type: ranged')
        channels = 28  # 27 state + 1 pos
        middle_action_num = 10  # a1 a2 a3 (3) and one-hot a4(7)
        low_action_num = 4

        MiddleAction_model = middle_net_build("coco_2/results_ranged/middle_action_model_499.pth",
                                              channels,
                                              256,
                                              middle_action_num)
        middle_output_row = MiddleAction_model(state_merge_pos_input).squeeze()
        middle_output_argmax, middle_output_class = net_forward(middle_output_row[:3])
        #print('middle action: ', middle_output_argmax, middle_output_class)
        high_action = [0,0,0,0]
        middle_action = middle_output_argmax
        y_middle = middle_output_row[:7].reshape(1, -1)

        if (middle_output_class == 0):
            LowAction_model = low_net_simplest_build("coco_2/results_ranged/low_action1_model_499.pth",
                                                     channels,
                                                     28,
                                                     low_action_num)
            low_output_row = LowAction_model(state_merge_pos_input, y_middle).squeeze()
            low_output_argmax, low_output_class = net_forward(low_output_row)
            #print('low action: ', low_output_argmax, low_output_class)
            low_action = low_output_argmax
        elif (middle_output_class == 1):
            LowAction_model = low_net_simple_build("coco_2/results_ranged/low_action2_model_499.pth",
                                                   channels,
                                                   128,
                                                   low_action_num)
            low_output_row = LowAction_model(state_merge_pos_input, y_middle).squeeze()
            low_output_argmax, low_output_class = net_forward(low_output_row)
            #print('low action: ', low_output_argmax, low_output_class)
            low_action = low_output_argmax
        else:
            LowAction_model = low_net_complex_build("coco_2/results_ranged/low_action3_model_499.pth",
                                                    channels,
                                                    256,
                                                    low_action_num)
            low_output_row = LowAction_model(state_merge_pos_input, y_middle).squeeze()
            low_output_argmax, low_output_class = net_forward(low_output_row)
            #print('low action: ', low_output_argmax, low_output_class)
            low_action = low_output_argmax
    else:
        #print('no existing owners.')
        high_action = [0,0,0,0]
        middle_action = [0,0,0,0]
        low_action = [0,0,0,0]
    return high_action,middle_action,low_action


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
    high_action, middle_action,low_action = forward_compute(owner_type_tensor, state_input, state_merge_pos_input)
    if torch.is_tensor(high_action):
        high_action = high_action.tolist()
    if torch.is_tensor(middle_action):
        middle_action = middle_action.tolist()
    if torch.is_tensor(low_action):
        low_action = low_action.tolist()

    return high_action, middle_action,low_action

