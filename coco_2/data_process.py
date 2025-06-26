import pandas
import numpy
from sklearn.model_selection import train_test_split
import torch

def state_get_merge(state_file_path1, state_file_path2, state_file_path3,
                    pos_file_path1, pos_file_path2, pos_file_path3):
    dim = 16
    channels = 27

    def state_pos_merge(state, pos):
        data_num = pos.shape[0]
        # 将 1d 的pos转化为16 * 16 * 1
        pos_2D = numpy.zeros((data_num, dim, dim), dtype=int)
        i_2D = pos // dim
        j_2D = pos % dim
        for i in range(data_num):
            pos_2D[i][i_2D[i]][j_2D[i]] = 1

        pos_2D = pos_2D.reshape(data_num, dim, dim, 1)

        state_new0 = numpy.append(state, pos_2D, axis=-1)
        state_new1 = numpy.swapaxes(state_new0, 2, 3)
        state_new = numpy.swapaxes(state_new1, 1, 2)

        return state_new

    state1 = pandas.read_csv(state_file_path1).values.reshape(-1, dim, dim, channels)
    state2 = pandas.read_csv(state_file_path2).values.reshape(-1, dim, dim, channels)
    state3 = pandas.read_csv(state_file_path3).values.reshape(-1, dim, dim, channels)

    pos1 = pandas.read_csv(pos_file_path1).values.reshape(-1)
    pos2 = pandas.read_csv(pos_file_path2).values.reshape(-1)
    pos3 = pandas.read_csv(pos_file_path3).values.reshape(-1)

    state_new1 = state_pos_merge(state1, pos1)
    state_new2 = state_pos_merge(state2, pos2)
    state_new3 = state_pos_merge(state3, pos3)

    state_new = numpy.append(numpy.append(state_new1, state_new2, axis=0), state_new3, axis=0)

    return state_new, state_new1, state_new2, state_new3


def state_get(state_file_path='data/light/light_state.csv', pos_file_path=None):
    dim = 16
    channels = 27

    state = pandas.read_csv(state_file_path).values.reshape(-1, dim, dim, channels)
    if (pos_file_path != None):
        pos = pandas.read_csv(pos_file_path).values.reshape(-1)

        data_num = pos.shape[0]
        # 将 1d 的pos转化为16 * 16 * 1
        pos_2D = numpy.zeros((data_num, dim, dim), dtype=int)
        i_2D = pos // dim
        j_2D = pos % dim
        for i in range(data_num):
            pos_2D[i][i_2D[i]][j_2D[i]] = 1

        pos_2D = pos_2D.reshape(data_num, dim, dim, 1)

        state_new0 = numpy.append(state, pos_2D, axis=-1)
        state_new1 = numpy.swapaxes(state_new0, 2, 3)
        state_new = numpy.swapaxes(state_new1, 1, 2)
    else:
        state_new1 = numpy.swapaxes(state, 2, 3)
        state_new = numpy.swapaxes(state_new1, 1, 2)
    return state_new  # 24199, 28, 16, 16

def middleaction_get(action_file_path, a4_info=True, worker=[False, False, False]):
    """
    "a4_info" is whether to use the one-hot a4 information
    """
    action = pandas.read_csv(action_file_path).values
    if(a4_info == True):
        dim_action_4 = 7
        action_4_onthot = numpy.array(numpy.eye(dim_action_4)[action[:, -1].reshape(1, -1) - 1][0].tolist(), dtype=int)
        middle_action_numpy = numpy.append(action[:, :-1], action_4_onthot, axis=1)
    elif(worker[0] == True or worker[1] == True): # produce or harvest
        middle_action_numpy = action
    elif(worker[2] == True): # attack
        dim_action_4 = 7
        action_4_onthot = numpy.array(numpy.eye(dim_action_4)[action[:, -1].reshape(1, -1) - 1][0].tolist(), dtype=int)
        middle_action_numpy = numpy.append(action[:, :-1], action_4_onthot, axis=1)
    else:
        middle_action_numpy = action[:, 2:]

    return middle_action_numpy

def highaction_get(action_file_path):
    action = pandas.read_csv(action_file_path).values

    return action[:, :2]

def data_get(file_path):
    return pandas.read_csv(file_path).values

def highaction_get_merge(action_file_path1, action_file_path2, action_file_path3):
    action1 = pandas.read_csv(action_file_path1).values
    action2 = pandas.read_csv(action_file_path2).values
    action3 = pandas.read_csv(action_file_path3).values

    action = numpy.append(numpy.append(action1, action2, axis=0), action3, axis=0)
    return action[:, 1:]

def lowaction_get(action_file_path):
    action = pandas.read_csv(action_file_path).values

    return action

def get_datasets(state_numpy, action_numpy, test_size=0.2, random_state=0, train_batch=64, test_batch=1,
                 action123=[False,False,False], middle_action_numpy=None):
    if (action123[0] == True):
        state_numpy_new = state_numpy[numpy.where(middle_action_numpy[:, 0] == 1)]
        action_numpy_new0 = action_numpy[numpy.where(middle_action_numpy[:, 0] == 1)]
        middle_numpy_new = middle_action_numpy[numpy.where(middle_action_numpy[:, 0] == 1)][:, 3:]
        action_numpy_new = numpy.append(action_numpy_new0, middle_numpy_new, axis=1)

    elif (action123[1] == True):
        state_numpy_new = state_numpy[numpy.where(middle_action_numpy[:, 1] == 1)]
        action_numpy_new0 = action_numpy[numpy.where(middle_action_numpy[:, 1] == 1)]
        middle_numpy_new = middle_action_numpy[numpy.where(middle_action_numpy[:, 1] == 1)][:, 3:]
        action_numpy_new = numpy.append(action_numpy_new0, middle_numpy_new, axis=1)

    elif (action123[2] == True):
        state_numpy_new = state_numpy[numpy.where(middle_action_numpy[:, 2] == 1)]
        action_numpy_new0 = action_numpy[numpy.where(middle_action_numpy[:, 2] == 1)]
        middle_numpy_new = middle_action_numpy[numpy.where(middle_action_numpy[:, 2] == 1)][:, 3:]
        action_numpy_new = numpy.append(action_numpy_new0, middle_numpy_new, axis=1)
    else:
        state_numpy_new = state_numpy
        action_numpy_new = action_numpy
    train_data, test_data, train_label, test_label = train_test_split(state_numpy_new,
                                                                      action_numpy_new,
                                                                      train_size=1 - test_size,
                                                                      random_state=random_state)

    traindata_size = train_data.shape[0]
    testdata_size = test_data.shape[0]
    train_set = numpy.append(train_data.reshape(traindata_size, -1), train_label, axis=-1)
    test_set = numpy.append(test_data.reshape(testdata_size, -1), test_label, axis=-1)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=test_batch, shuffle=False)

    return train_loader, test_loader

def get_datasets_worker(state_numpy, action_numpy, test_size=0.2, random_state=0, train_batch=64, test_batch=1,
                        actions=[False,False,False, False], middle_action_numpy=None, y=False):
    if (actions[0] == True):
        state_numpy_new = state_numpy[numpy.where(middle_action_numpy[:, 0] == 1)]
        action_numpy_new = action_numpy[numpy.where(middle_action_numpy[:, 0] == 1)]
        if(y == True):
            middle_numpy_new = middle_action_numpy[numpy.where(middle_action_numpy[:, 0] == 1)][:, 3:]
            action_numpy_new = numpy.append(action_numpy_new, middle_numpy_new, axis=1)

    elif (actions[1] == True):
        state_numpy_new = state_numpy[numpy.where(middle_action_numpy[:, 1] == 1)]
        action_numpy_new = action_numpy[numpy.where(middle_action_numpy[:, 1] == 1)]
        if (y == True):
            middle_numpy_new = middle_action_numpy[numpy.where(middle_action_numpy[:, 1] == 1)][:, 3:]
            action_numpy_new = numpy.append(action_numpy_new, middle_numpy_new, axis=1)

    elif (actions[2] == True):
        state_numpy_new = state_numpy[numpy.where(middle_action_numpy[:, 2] == 1)]
        action_numpy_new = action_numpy[numpy.where(middle_action_numpy[:, 2] == 1)]
        if (y == True):
            middle_numpy_new = middle_action_numpy[numpy.where(middle_action_numpy[:, 2] == 1)][:, 3:]
            action_numpy_new = numpy.append(action_numpy_new, middle_numpy_new, axis=1)
    elif (actions[3] == True):
        state_numpy_new = state_numpy[numpy.where(middle_action_numpy[:, 3] == 1)]
        action_numpy_new = action_numpy[numpy.where(middle_action_numpy[:, 3] == 1)]
        if (y == True):
            middle_numpy_new = middle_action_numpy[numpy.where(middle_action_numpy[:, 3] == 1)][:, 3:]
            action_numpy_new = numpy.append(action_numpy_new, middle_numpy_new, axis=1)
    else:
        state_numpy_new = state_numpy
        action_numpy_new = action_numpy
    train_data, test_data, train_label, test_label = train_test_split(state_numpy_new,
                                                                      action_numpy_new,
                                                                      train_size=1 - test_size,
                                                                      random_state=random_state)

    traindata_size = train_data.shape[0]
    testdata_size = test_data.shape[0]
    train_set = numpy.append(train_data.reshape(traindata_size, -1), train_label, axis=-1)
    test_set = numpy.append(test_data.reshape(testdata_size, -1), test_label, axis=-1)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=test_batch, shuffle=False)

    return train_loader, test_loader

def get_datasets_base(state_numpy, action_numpy, test_size=0.2, random_state=0, train_batch=64, test_batch=1,
                      action2=False, high_action_numpy=None):
    if (action2 == True):
        state_numpy_new = state_numpy[numpy.where(high_action_numpy[:, 1] == 1)]
        action_numpy_new = action_numpy[numpy.where(high_action_numpy[:, 1] == 1)]
    else:
        state_numpy_new = state_numpy
        action_numpy_new = action_numpy
    train_data, test_data, train_label, test_label = train_test_split(state_numpy_new,
                                                                      action_numpy_new,
                                                                      train_size=1 - test_size,
                                                                      random_state=random_state)

    traindata_size = train_data.shape[0]
    testdata_size = test_data.shape[0]
    train_set = numpy.append(train_data.reshape(traindata_size, -1), train_label, axis=-1)
    test_set = numpy.append(test_data.reshape(testdata_size, -1), test_label, axis=-1)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=test_batch, shuffle=False)

    return train_loader, test_loader


def get_datasets_barrack(state_numpy, action_numpy, test_size=0.2, random_state=0, train_batch=64, test_batch=1,
                         action2=False, high_action_numpy=None, action345=[False, False, False], middle_action_numpy=None):
    if (action2 == True):
        state_numpy_new = state_numpy[numpy.where(high_action_numpy[:, 1] == 1)]
        action_numpy_new = action_numpy[numpy.where(high_action_numpy[:, 1] == 1)]
    elif (action345[0] == True):
        state_numpy_new = state_numpy[numpy.where(middle_action_numpy[:, 0] == 1)]
        action_numpy_new = action_numpy[numpy.where(middle_action_numpy[:, 0] == 1)]
    elif (action345[1] == True):
        state_numpy_new = state_numpy[numpy.where(middle_action_numpy[:, 1] == 1)]
        action_numpy_new = action_numpy[numpy.where(middle_action_numpy[:, 1] == 1)]
    elif (action345[2] == True):
        state_numpy_new = state_numpy[numpy.where(middle_action_numpy[:, 2] == 1)]
        action_numpy_new = action_numpy[numpy.where(middle_action_numpy[:, 2] == 1)]
    else:
        state_numpy_new = state_numpy
        action_numpy_new = action_numpy
    train_data, test_data, train_label, test_label = train_test_split(state_numpy_new,
                                                                      action_numpy_new,
                                                                      train_size=1 - test_size,
                                                                      random_state=random_state)

    traindata_size = train_data.shape[0]
    testdata_size = test_data.shape[0]
    train_set = numpy.append(train_data.reshape(traindata_size, -1), train_label, axis=-1)
    test_set = numpy.append(test_data.reshape(testdata_size, -1), test_label, axis=-1)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=test_batch, shuffle=False)

    return train_loader, test_loader


def get_datasets_single(state_numpy, action_numpy, test_size=0.2, random_state=0, train_batch=64, test_batch=1):
    state_numpy_new = state_numpy
    action_numpy_new = action_numpy
    train_data, test_data, train_label, test_label = train_test_split(state_numpy_new,
                                                                      action_numpy_new,
                                                                      train_size=1 - test_size,
                                                                      random_state=random_state)

    traindata_size = train_data.shape[0]
    testdata_size = test_data.shape[0]
    train_set = numpy.append(train_data.reshape(traindata_size, -1), train_label, axis=-1)
    test_set = numpy.append(test_data.reshape(testdata_size, -1), test_label, axis=-1)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=test_batch, shuffle=False)

    return train_loader, test_loader