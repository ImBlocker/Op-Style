import os
import csv
import random

import time
import numpy as np
import pandas as pd
import torch
import copy
from coco_2 import main

class GLOBAL_VALUE:
    def __init__(self):
        self.heavy_fix = []
        self.heavy_follow = []





def get_action_type1(state_,global_value):
    state = copy.deepcopy(state_[0])
    #state is 16*16*27
    basepos = [34]
    barrackpos =[]
    workerpos = []
    lightpos = []
    heavypos = []
    rangepos = []
    actions = np.zeros([256,7])
    for i in range(16):
        for j in range(16):
            pos = 16*i + j
            if state[i][j][11] == 1 and state[i][j][16] == 1 and state[i][j][21] == 1:
                barrackpos.append(pos)
            if state[i][j][11] == 1 and state[i][j][17] == 1 and state[i][j][21] == 1:
                workerpos.append(pos)
            if state[i][j][11] == 1 and state[i][j][18] == 1 and state[i][j][21] == 1:
                lightpos.append(pos)
            if state[i][j][11] == 1 and state[i][j][19] == 1 and state[i][j][21] == 1:
                heavypos.append(pos)
            if state[i][j][11] == 1 and state[i][j][20] == 1 and state[i][j][21] == 1:
                rangepos.append(pos)
    for pos in barrackpos:
        high_action,middle_action,low_action = usenetwork(state,pos,1)
        if high_action[1] == 1:  #build
            if middle_action[0] == 1:
                action = [4, 0, 0, 0, 0, 4, 0]
            elif middle_action[1] == 1:
                action = [4, 0, 0, 0, 0, 5, 0]
            else:
                action = [4, 0, 0, 0, 0, 6, 0]
            if low_action[0] == 1:
                action[4] = 0
            elif low_action[1] == 1:
                action[4] = 1
            elif low_action[2] == 1:
                action[4] = 2
            else:
                action[4] = 3
            if len(heavypos) < 3:
                action[5] = 5
            actions[pos] = action
            state = updatestate(state, action, pos)
        else:
            action = [4, 0, 0, 0, 0, 5, 0]
            actions[pos] = action
            state = updatestate(state, action, pos)

    for pos in basepos:
        high_action,middle_action,low_action = usenetwork(state,pos,1)
        if high_action[1] == 1: #build worker
            if low_action[0] == 1:
                action = [4, 0, 0, 0, 0, 3, 0]
            elif low_action[1] == 1:
                action = [4, 0, 0, 0, 1, 3, 0]
            elif low_action[2] == 1:
                action = [4, 0, 0, 0, 2, 3, 0]
            else:
                action = [4, 0, 0, 0, 3, 3, 0]
            if len(workerpos) > 4:
                action = [0,0,0,0,0,0,0]
            actions[pos] = action
            state = updatestate(state,action,pos)

    for pos in rangepos:
        high_action,middle_action,low_action = usenetwork(state,pos,1)

        if middle_action[0] == 1:   #can attack
            targetpos = get_targetpos(pos,state,low_action)
            if targetpos != -1:
                action = [5, 0, 0, 0, 0, 0, targetpos]
            else:
                action = [0, 0, 0, 0, 0, 0, 0]
        else:

            if low_action[0] == 1:
                p = random.random()
                if p < 0.5:
                    action = [1, 1, 0, 0, 0, 0, 0]

                else:
                    action = [1, 2, 0, 0, 0, 0, 0]
                    if pos // 16 == 15:
                        action = [1, 0, 0, 0, 0, 0, 0]
            elif low_action[1] == 1:
                action = [1, 1, 0, 0, 0, 0, 0]
            elif low_action[2] == 1:
                action = [1, 2, 0, 0, 0, 0, 0]
            else:
                action = [1, 3, 0, 0, 0, 0, 0]
        actions[pos] = action
        state = updatestate(state,action,pos)

    for pos in lightpos:
        high_action,middle_action,low_action = usenetwork(state,pos,1)
        if middle_action[0] == 1:
            if low_action[0] == 1:
                action = [5, 0, 0, 0, 0, 0, 17]
            elif low_action[1] == 1:
                action = [5, 0, 0, 0, 0, 0, 25]
            elif low_action[2] == 1:
                action = [5, 0, 0, 0, 0, 0, 31]
            else :
                action = [5, 0, 0, 0, 0, 0, 23]
        else:
            if low_action[0] == 1:
                action = [1, 0, 0, 0, 0, 0, 0]
            elif low_action[1] == 1:
                action = [1, 1, 0, 0, 0, 0, 0]
            elif low_action[2] == 1:
                action = [1, 2, 0, 0, 0, 0, 0]
            else:
                action = [1, 3, 0, 0, 0, 0, 0]
        actions[pos] = action
        state = updatestate(state,action,pos)

    for pos in heavypos:
        high_action,middle_action,low_action = usenetwork(state,pos,1)
        if middle_action[0] == 1:  #can attack
            if low_action[0] == 1:
                action = [5, 0, 0, 0, 0, 0, 17]
            elif low_action[1] == 1:
                action = [5, 0, 0, 0, 0, 0, 25]
            elif low_action[2] == 1:
                action = [5, 0, 0, 0, 0, 0, 31]
            else :
                action = [5, 0, 0, 0, 0, 0, 23]
        else:  #move
            if low_action[0] == 1:
                action = [1, 0, 0, 0, 0, 0, 0]
            elif low_action[1] == 1:
                action = [1, 1, 0, 0, 0, 0, 0]
            elif low_action[2] == 1:
                action = [1, 2, 0, 0, 0, 0, 0]
            else:
                action = [1, 3, 0, 0, 0, 0, 0]
        actions[pos] = action
        state = updatestate(state,action,pos)

    for pos in workerpos:
        high_action,middle_action,low_action = usenetwork(state,pos,1)
        if high_action[0] == 1:   #build barrack
            if middle_action[0] == 1:  #move
                action = [1, 0, 0, 0, 0, 0, 0]
                if low_action[0] == 1:
                    action[1] = 0
                elif low_action[1] == 1:
                    action[1] = 1
                elif low_action[2] == 1:
                    action[1] = 2
                else:
                    action[1] = 3
            else:   #produce
                action = [4, 0, 0, 0, 0, 2, 0]
                if low_action[0] == 1:
                    action[4] = 0
                elif low_action[1] == 1:
                    action[4] = 1
                elif low_action[2] == 1:
                    action[4] = 2
                else:
                    action[4] = 3

        elif high_action[1] == 1:  #mine
            if middle_action[0] == 1 or middle_action[1] == 1:  #move
                action = [1, 0, 0, 0, 0, 0, 0]
                if low_action[0] == 1:
                    action[1] = 0
                elif low_action[1] == 1:
                    action[1] = 1
                elif low_action[2] == 1:
                    action[1] = 2
                else:
                    action[1] = 3
            elif middle_action[2] == 1:  #harvest
                action = [2, 0, 0, 0, 0, 0, 0]
                if low_action[0] == 1:
                    action[2] = 0
                elif low_action[1] == 1:
                    action[2] = 1
                elif low_action[2] == 1:
                    action[2] = 2
                else:
                    action[2] = 3
            else:  #return
                action = [3, 0, 0, 0, 0, 0, 0]
                if low_action[0] == 1:
                    action[3] = 0
                elif low_action[1] == 1:
                    action[3] = 1
                elif low_action[2] == 1:
                    action[3] = 2
                else:
                    action[3] = 3

        else:   #attack
            if middle_action[0] == 1: #can attack
                action = [5, 0, 0, 0, 0, 0, 0]
                if low_action[0] == 1:
                    action[6] = 17
                elif low_action[1] == 1:
                    action[6] = 25
                elif low_action[2] == 1:
                    action[6] = 31
                else:
                    action[6] = 23
            else:
                action = [1, 0, 0, 0, 0, 0, 0]
                if low_action[0] == 1:
                    action[1] = 0
                elif low_action[1] == 1:
                    action[1] = 1
                elif low_action[2] == 1:
                    action[1] = 2
                else:
                    action[1] = 3

        actions[pos] = action
        state = updatestate(state,action,pos)

    # for i in range(256):
    #     if sum(actions[i] != 0):
    #         print('action_before_mask', i, '=', actions[i])

    action_mask(state_,actions,basepos,barrackpos,workerpos,lightpos,heavypos,rangepos,global_value)

    # for i in range(256):
    #     if sum(actions[i] != 0):
    #         print('action_after_mask', i, '=', actions[i])



    return actions









def usenetwork(state,pos,type):
    high_action, middle_action, low_action = main.main(state,pos)
    if type == 2:
        return high_action, middle_action, low_action
    else:
        high_max_index = high_action.index(max(high_action))
        len1 = len(high_action)
        for i in range(len1):
            if i == high_max_index:
                high_action[i] = 1
            else:
                high_action[i] = 0
        middle_max_index = middle_action.index(max(middle_action))
        len2 = len(middle_action)
        for i in range(len2):
            if i == middle_max_index:
                middle_action[i] = 1
            else:
                middle_action[i] = 0
        low_max_index = low_action.index(max(low_action))
        len3 = len(low_action)
        for i in range(len3):
            if i == low_max_index:
                low_action[i] = 1
            else:
                low_action[i] = 0
        return high_action, middle_action, low_action

def get_targetpos(pos,state,low_action):
    if low_action[0] == 1:
        for i in range(17,21):
            targetpos = pos + i - 33
            if targetpos < 0 or targetpos > 255:
                continue
            x = targetpos // 16
            y = targetpos % 16
            if state[x][y][12] == 1:
                return i
        for i in range(10,14):
            targetpos = pos + i - 42
            if targetpos < 0 or targetpos > 255:
                continue
            x = targetpos // 16
            y = targetpos % 16
            if state[x][y][12] == 1:
                return i
        for i in range(3,7):
            targetpos = pos + i - 51
            if targetpos < 0 or targetpos > 255:
                continue
            x = targetpos // 16
            y = targetpos % 16
            if state[x][y][12] == 1:
                return i
    elif low_action[1] == 1:
        for i in range(25,28):
            targetpos = pos + i - 24
            if targetpos < 0 or targetpos > 255:
                continue
            x = targetpos // 16
            y = targetpos % 16
            if state[x][y][12] == 1:
                return i
        for i in range(32,35):
            targetpos = pos + i - 15
            if targetpos < 0 or targetpos > 255:
                continue
            x = targetpos // 16
            y = targetpos % 16
            if state[x][y][12] == 1:
                return i
        for i in range(39,42):
            targetpos = pos + i - 6
            if targetpos < 0 or targetpos > 255:
                continue
            x = targetpos // 16
            y = targetpos % 16
            if state[x][y][12] == 1:
                return i
        for i in range(46,49):
            targetpos = pos + i + 3
            if targetpos < 0 or targetpos > 255:
                continue
            x = targetpos // 16
            y = targetpos % 16
            if state[x][y][12] == 1:
                return i
    elif low_action[2] == 1:
        for i in range(28,32):
            targetpos = pos + i - 15
            if targetpos < 0 or targetpos > 255:
                continue
            x = targetpos // 16
            y = targetpos % 16
            if state[x][y][12] == 1:
                return i
        for i in range(35,39):
            targetpos = pos + i - 6
            if targetpos < 0 or targetpos > 255:
                continue
            x = targetpos // 16
            y = targetpos % 16
            if state[x][y][12] == 1:
                return i
        for i in range(32,46):
            targetpos = pos + i + 3
            if targetpos < 0 or targetpos > 255:
                continue
            x = targetpos // 16
            y = targetpos % 16
            if state[x][y][12] == 1:
                return i
    else:
        for i in range(21,24):
            targetpos = pos + i - 24
            if targetpos < 0 or targetpos > 255:
                continue
            x = targetpos // 16
            y = targetpos % 16
            if state[x][y][12] == 1:
                return i
        for i in range(14,17):
            targetpos = pos + i - 33
            if targetpos < 0 or targetpos > 255:
                continue
            x = targetpos // 16
            y = targetpos % 16
            if state[x][y][12] == 1:
                return i
        for i in range(7,10):
            targetpos = pos + i - 42
            if targetpos < 0 or targetpos > 255:
                continue
            x = targetpos // 16
            y = targetpos % 16
            if state[x][y][12] == 1:
                return i
        for i in range(0,3):
            targetpos = pos + i - 51
            if targetpos < 0 or targetpos > 255:
                continue
            x = targetpos // 16
            y = targetpos % 16
            if state[x][y][12] == 1:
                return i
    return  -1

def updatestate(state,action,pos): #add action without consider turn
    x = pos // 16
    y = pos % 16
    action_type = action[0]
    if action_type == 1: #move
        state[x][y][21] = 0
        state[x][y][22] = 1
    elif action_type == 2: #harvest
        state[x][y][21] = 0
        state[x][y][23] = 1
    elif action_type == 3: #return
        state[x][y][21] = 0
        state[x][y][24] = 1
    elif action_type == 4: #produce
        state[x][y][21] = 0
        state[x][y][25] = 1
    elif action_type == 5: #attack
        state[x][y][21] = 0
        state[x][y][26] = 1
    return state

def action_mask(state_,actions,basepos,barrackpos,workerpos,lightpos,heavypos,rangepos,global_value):
    state = copy.deepcopy(state_[0])
    have_our_unit_pos = []
    have_enemy_unit_pos = []
    have_buildormove_unit_pos = []
    for i in range(16):
        for j in range(16):
            if  state[i][j][11] == 1:
                pos_ = 16*i+j
                have_our_unit_pos.append(pos_)
            if  state[i][j][12] == 1:
                pos_ = 16*i+j
                have_enemy_unit_pos.append(pos_)
    for pos in barrackpos:
        action = actions[pos]  #must be produce
        turn = action[4]
        target_pos = get_targetpos_inmask(pos,turn)
        if (target_pos not in have_our_unit_pos) and (target_pos not in have_enemy_unit_pos) and (target_pos not in have_buildormove_unit_pos):
            have_buildormove_unit_pos.append(target_pos)
        else:
            new_turn,new_targetpos = choose_new_turn(pos,turn,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos)
            if new_turn != -1:
                have_buildormove_unit_pos.append(new_targetpos)
                action[4] = new_turn
                actions[pos] = action
            else:
                actions[pos] = [0,0,0,0,0,0,0]

    for pos in basepos:
        need_idle = base_need_idle(state)
        if need_idle :
            actions[pos] = [0,0,0,0,0,0,0]
        else:
            action = actions[pos]  #must be produce
            turn = action[4]
            target_pos = get_targetpos_inmask(pos,turn)
            if (target_pos not in have_our_unit_pos) and (target_pos not in have_enemy_unit_pos) and (target_pos not in have_buildormove_unit_pos):
                have_buildormove_unit_pos.append(target_pos)
            else:
                new_turn,new_targetpos = choose_new_turn(pos,turn,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos)
                if new_turn != -1:
                    have_buildormove_unit_pos.append(new_targetpos)
                    action[4] = new_turn
                    actions[pos] = action
                else:
                    actions[pos] = [0,0,0,0,0,0,0]

    for pos in rangepos:
        action = actions[pos]  #move or attack
        if action[0] == 5:  #attack   no need to do
            continue
        if action[0] == 1: #move
            target_pos = unit_can_attack(pos,have_enemy_unit_pos,0)  #0:ranged
            if target_pos != -1:  #can attack
                attack_pos = build_attack_pos(pos,target_pos)
                action = [5,0,0,0,0,0,attack_pos]
                actions[pos] = action
            else:   #only move
                turn = action[1]
                target_pos = get_targetpos_inmask(pos, turn)
                if (target_pos not in have_our_unit_pos) and (target_pos not in have_enemy_unit_pos) and (
                        target_pos not in have_buildormove_unit_pos) and(inside(target_pos)):
                    have_buildormove_unit_pos.append(target_pos)
                else:
                    new_turn, new_targetpos = choose_new_turn(pos, turn, have_our_unit_pos, have_enemy_unit_pos,
                                                              have_buildormove_unit_pos)
                    if new_turn != -1:
                        have_buildormove_unit_pos.append(new_targetpos)
                        action[1] = new_turn
                        actions[pos] = action
                    else:
                        actions[pos] = [0, 0, 0, 0, 0, 0, 0]

    for pos in lightpos:
        action = actions[pos]  #move or attack
        if action[0] == 5:  #attack   no need to do
            continue
        if action[0] == 1: #move
            target_pos = unit_can_attack(pos, have_enemy_unit_pos, 1)  # 0:light
            if target_pos != -1:  # can attack
                attack_pos = build_attack_pos(pos,target_pos)
                action = [5, 0, 0, 0, 0, 0, attack_pos]
                actions[pos] = action
            else:   #only move
                turn = action[1]
                xiuzheng_turn = change_turn(pos,turn)
                target_pos = get_targetpos_inmask(pos, xiuzheng_turn)
                if (target_pos not in have_our_unit_pos) and (target_pos not in have_enemy_unit_pos) and (
                        target_pos not in have_buildormove_unit_pos) and (inside(target_pos)):
                    have_buildormove_unit_pos.append(target_pos)
                else:
                    new_turn, new_targetpos = choose_new_turn(pos, xiuzheng_turn, have_our_unit_pos, have_enemy_unit_pos,
                                                              have_buildormove_unit_pos)
                    if new_turn != -1:
                        have_buildormove_unit_pos.append(new_targetpos)
                        action[1] = new_turn
                        actions[pos] = action
                    else:
                        actions[pos] = [0, 0, 0, 0, 0, 0, 0]

    for pos in heavypos:
        target_pos = unit_can_attack(pos, have_enemy_unit_pos, 1)  # attack
        if target_pos != -1:  # can attack
            attack_pos = build_attack_pos(pos, target_pos)
            action = [5, 0, 0, 0, 0, 0, attack_pos]
            actions[pos] = action
        else:  #can not attack,move
            xiuzheng_turn = heavy_turn_inmask(pos, state, global_value,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos)
            print('xiuzheng_turn=',xiuzheng_turn)
            if xiuzheng_turn != -1:
                action = [1, xiuzheng_turn, 0, 0, 0, 0, 0]
                if xiuzheng_turn == 0:
                    next_pos = pos - 16
                elif xiuzheng_turn == 1:
                    next_pos = pos + 1
                elif xiuzheng_turn == 2:
                    next_pos = pos + 16
                else:
                    next_pos = pos - 1
                have_buildormove_unit_pos.append(next_pos)
            else:
                action = [0, 0, 0, 0, 0, 0, 0]
            actions[pos] = action





    # for pos in heavypos:
    #     action = actions[pos]  #move or attack
    #     if action[0] == 5:  #attack   no need to do
    #         continue
    #     if action[0] == 1: #move
    #         target_pos = unit_can_attack(pos,have_enemy_unit_pos,1)  # 1:light or heavy
    #         if target_pos != -1:  #can attack
    #             attack_pos = build_attack_pos(pos,target_pos)
    #             action = [5,0,0,0,0,0,attack_pos]
    #             actions[pos] = action
    #         else:   #only move
    #             turn = action[1]
    #             #xiuzheng_turn = heavy_turn_inmask(pos,state,global_value)
    #             xiuzheng_turn = change_turn(pos, turn)
    #             target_pos = get_targetpos_inmask(pos, xiuzheng_turn)
    #             if (target_pos not in have_our_unit_pos) and (target_pos not in have_enemy_unit_pos) and (
    #                     target_pos not in have_buildormove_unit_pos) and(inside(target_pos)):
    #                 have_buildormove_unit_pos.append(target_pos)
    #             else:
    #                 new_turn, new_targetpos = choose_new_turn(pos, xiuzheng_turn, have_our_unit_pos, have_enemy_unit_pos,
    #                                                           have_buildormove_unit_pos)
    #                 if new_turn != -1:
    #                     have_buildormove_unit_pos.append(new_targetpos)
    #                     action[1] = new_turn
    #                     actions[pos] = action
    #                 else:
    #                     actions[pos] = [0, 0, 0, 0, 0, 0, 0]

    for pos in workerpos:
        build_barrack = need_to_build(state)
        if build_barrack and pos >= 67:
            new_turn, new_targetpos = choose_new_turn(pos, -1, have_our_unit_pos, have_enemy_unit_pos,
                                                      have_buildormove_unit_pos)
            action = [4, 0, 0, 0, new_turn, 2, 0]
            actions[pos] = action
        else:
            action = actions[pos]  #move or harvest  or return or produce or attack
            if action[0] == 4:  #produce
                turn = action[4]
                target_pos = get_targetpos_inmask(pos, turn)
                if (target_pos not in have_our_unit_pos) and (target_pos not in have_enemy_unit_pos) and (
                        target_pos not in have_buildormove_unit_pos) and(inside(target_pos)):
                    have_buildormove_unit_pos.append(target_pos)
                else:
                    new_turn, new_targetpos = choose_new_turn(pos, turn, have_our_unit_pos, have_enemy_unit_pos,
                                                              have_buildormove_unit_pos)
                    if new_turn != -1:
                        have_buildormove_unit_pos.append(new_targetpos)
                        action[4] = new_turn
                        actions[pos] = action
                    else:
                        actions[pos] = [0, 0, 0, 0, 0, 0, 0]

            if action[0] == 5:  #attack
                turn = action[6]
                if turn == 17:
                    turn = 0
                elif turn == 25:
                    turn = 1
                elif turn == 31:
                    turn = 2
                else:
                    turn = 3
                target_pos = get_targetpos_inmask(pos, turn)
                if (target_pos in have_enemy_unit_pos):
                    continue
                else:
                    new_turn = choose_attack_turn(pos, have_enemy_unit_pos)
                    if new_turn != -1:
                        attack_pos = build_attack_pos(pos,new_turn)
                        action[6] = attack_pos
                        actions[pos] = action
                    else:  #change to move
                        new_turn, new_targetpos = choose_new_turn(pos, -1, have_our_unit_pos, have_enemy_unit_pos,
                                                                  have_buildormove_unit_pos)
                        if new_turn != -1:
                            actions[pos] = [1, new_turn, 0, 0, 0, 0, 0]
                        else:
                            actions[pos] = [0, 0, 0, 0, 0, 0, 0]
            if action[0] == 1: #move
                target_pos = unit_can_attack(pos,have_enemy_unit_pos,1)  # 1:light or heavy
                if target_pos != -1:  #can attack
                    attack_pos = build_attack_pos(pos,target_pos)
                    action = [5,0,0,0,0,0,attack_pos]
                    actions[pos] = action
                else:   #only move
                    turn = action[1]
                    target_pos = get_targetpos_inmask(pos, turn)
                    if (target_pos not in have_our_unit_pos) and (target_pos not in have_enemy_unit_pos) and (
                            target_pos not in have_buildormove_unit_pos) and(inside(target_pos)):
                        have_buildormove_unit_pos.append(target_pos)
                    else:
                        new_turn, new_targetpos = choose_new_turn(pos, turn, have_our_unit_pos, have_enemy_unit_pos,
                                                                  have_buildormove_unit_pos)
                        if new_turn != -1:
                            have_buildormove_unit_pos.append(new_targetpos)
                            action[1] = new_turn
                            actions[pos] = action
                        else:
                            actions[pos] = [0, 0, 0, 0, 0, 0, 0]


def heavy_turn_inmask(pos,state,global_value,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos):
    heavy_fix = global_value.heavy_fix
    heavy_follow = global_value.heavy_follow
    enemy_pos = []
    heavy_pos = []
    heavy_i = pos // 16
    heavy_j = pos % 16
    for i in range(16):
        for j in range(16):
            pos_ = 16*i + j
            if state[i][j][12] == 1 and (state[i][j][18] == 1 or state[i][j][19] == 1 or state[i][j][20] == 1 or state[i][j][17] == 1):
                enemy_pos.append(pos_)
            if state[i][j][11] == 1 and state[i][j][19] == 1:
                heavy_pos.append(pos_)

    for item in heavy_fix:
        flag = 0
        for pos_ in heavy_pos:
            if pos_ in item:
                flag = 1
        if flag == 0:
            heavy_fix.remove(item)

    for item in heavy_follow:
        flag = 0
        for pos_ in heavy_pos:
            if pos_ in item:
                flag = 1
        if flag == 0:
            heavy_follow.remove(item)

    if heavy_follow == []:
        min_dis = 2000
        index = -1
        for i in range(len(enemy_pos)):
            pos_ = enemy_pos[i]
            dis = (pos_ // 16 - heavy_i) + (pos_ % 16 - heavy_j)
            if dis < min_dis:
                min_dis = dis
                index = i
        if enemy_pos != []:
            pos_ = enemy_pos[index]
            target_i = pos_ // 16
            target_j = pos_ % 16
            turn, next_pos = heavy_turn_pos(pos, heavy_i, heavy_j, target_i, target_j, have_our_unit_pos, have_enemy_unit_pos,
                                            have_buildormove_unit_pos)

            if turn != -1:
                heavy_follow.append([pos,next_pos])
                global_value.heavy_follow = heavy_follow

            return turn
        return -1

    if heavy_follow != []:
        for item in heavy_follow:
            if pos == item[1] or pos == item[0]:
                min_dis = 2000
                index = -1
                for i in range(len(enemy_pos)):
                    pos_ = enemy_pos[i]
                    dis = (pos_ // 16 - heavy_i) + (pos_ % 16 - heavy_j)
                    if dis < min_dis:
                        min_dis = dis
                        index = i
                if enemy_pos != []:
                    pos_ = enemy_pos[index]
                    target_i = pos_ // 16
                    target_j = pos_ % 16
                    turn, next_pos = heavy_turn_pos(pos, heavy_i, heavy_j, target_i, target_j, have_our_unit_pos,
                                                    have_enemy_unit_pos,
                                                    have_buildormove_unit_pos)
                    if turn != -1:
                        heavy_follow.remove(item)
                        heavy_follow.append([pos, next_pos])
                        global_value.heavy_follow = heavy_follow
                    return turn


    if heavy_fix == []:
        turn,next_pos = heavy_turn_pos(pos,heavy_i,heavy_j,13,13,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos)
        if turn != -1:
            heavy_fix.append([pos,next_pos])
            global_value.heavy_fix = heavy_fix
        return turn

    if heavy_fix != []:
        for item in heavy_fix:
            if pos == item[1] or pos == item[0]:
                turn, next_pos = heavy_turn_pos(pos, heavy_i, heavy_j, 13, 13, have_our_unit_pos,
                                                have_enemy_unit_pos,
                                                have_buildormove_unit_pos)
                if turn != -1:
                    heavy_fix.remove(item)
                    heavy_fix.append([pos, next_pos])
                    global_value.heavy_fix = heavy_fix
                return turn

    min_dis = 2000
    index = -1
    for i in range(len(enemy_pos)):
        pos_ = enemy_pos[i]
        dis = (pos_ // 16 - heavy_i) + (pos_ % 16 - heavy_j)
        if dis < min_dis:
            min_dis = dis
            index = i
    if enemy_pos != []:
        pos_ = enemy_pos[index]
        target_i = pos_ // 16
        target_j = pos_ % 16
        turn, next_pos = heavy_turn_pos(pos, heavy_i, heavy_j, target_i, target_j, have_our_unit_pos,
                                        have_enemy_unit_pos,
                                        have_buildormove_unit_pos)

        if turn != -1:
            heavy_follow.append([pos, next_pos])
            global_value.heavy_follow = heavy_follow
        return turn






    return -1


def heavy_turn_pos(pos,heavy_i,heavy_j,target_i,target_j,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos):
    p = random.random()
    if heavy_i < target_i:
        if heavy_j < target_j:
            if p < 0.5 and nounit(pos+1,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos):
                turn = 1
                next_pos = pos + 1
            elif nounit(pos+16,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos):
                turn = 2
                next_pos = pos + 16
            else:
                turn = -1
                next_pos = -1
        elif heavy_j == target_j and nounit(pos+16,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos):
            turn = 2
            next_pos = pos + 16
        else:
            if p < 0.5 and nounit(pos+16,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos):
                turn = 2
                next_pos = pos + 16
            elif nounit(pos-1,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos):
                turn = 3
                next_pos = pos - 1
            else:
                turn = -1
                next_pos = -1
    elif heavy_i == target_i:
        if heavy_j < target_j and nounit(pos+1,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos):
            turn = 1
            next_pos = pos + 1
        elif heavy_j>=target_j and nounit(pos-1,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos):
            turn = 3
            next_pos = pos - 1
        else:
            turn = -1
            next_pos = -1
    else:
        if heavy_j < target_j:
            if p < 0.5 and  nounit(pos+1,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos):
                turn = 1
                next_pos = pos + 1
            elif nounit(pos-16,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos):
                turn = 0
                next_pos = pos - 16
            else:
                turn = -1
                next_pos = -1
        elif heavy_j == target_j and nounit(pos-16,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos):
            turn = 0
            next_pos = pos - 16
        else:
            if p < 0.5 and nounit(pos-1,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos):
                turn = 3
                next_pos = pos - 1
            elif nounit(pos-16,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos):
                turn = 0
                next_pos = pos - 16
            else:
                turn = -1
                next_pos = -1
    return turn, next_pos


def nounit(pos,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos):
    if (pos not in have_our_unit_pos) and (pos not in have_enemy_unit_pos) and (
            pos not in have_buildormove_unit_pos) and (inside(pos)):
        return True
    else:
        return False








def get_targetpos_inmask(pos,turn):
    if turn == 0:
        targetpos = pos - 16
    elif turn == 1:
        targetpos = pos + 1
    elif turn == 2:
        targetpos = pos + 16
    else:
        targetpos = pos - 1
    return targetpos

def choose_new_turn(pos,turn,have_our_unit_pos,have_enemy_unit_pos,have_buildormove_unit_pos):
    turn_ = [1,2,0,3]
    for new_turn in turn_:
        if new_turn == turn:
            continue
        else:
            target_pos = get_targetpos_inmask(pos,new_turn)
            if (target_pos not in have_our_unit_pos) and (target_pos not in have_enemy_unit_pos) and (target_pos not in have_buildormove_unit_pos) and(inside(target_pos)):
                return new_turn,target_pos
    return -1,-1

def choose_attack_turn(pos, have_enemy_unit_pos):
    pos1 = pos + 1
    pos2 = pos + 16
    pos3 = pos - 1
    pos4 = pos - 16
    if pos1 in have_enemy_unit_pos:
        return pos1
    if pos2 in have_enemy_unit_pos:
        return pos2
    if pos3 in have_enemy_unit_pos:
        return pos3
    if pos4 in have_enemy_unit_pos:
        return pos4
    return -1

def unit_can_attack(pos,have_enemy_unit_pos,type):
    if type == 1:
        pos1 = pos + 1
        pos2 = pos + 16
        pos3 = pos - 1
        pos4 = pos - 16
        if pos1 in have_enemy_unit_pos:
            return pos1
        if pos2 in have_enemy_unit_pos:
            return pos2
        if pos3 in have_enemy_unit_pos:
            return pos3
        if pos4 in have_enemy_unit_pos:
            return pos4
        return -1
    else:
        i1 = pos // 16
        j1 = pos % 16
        for pos_ in have_enemy_unit_pos:
            i2 = pos_ // 16
            j2 = pos_ % 16
            if abs(i1 - i2) <= 2 and abs(j1 - j2) <= 2:
                return pos_
        return -1

def build_attack_pos(pos,target_pos):
    pos_i = pos // 16
    pos_j = pos % 16
    target_i = target_pos // 16
    target_j = target_pos % 16
    attack_pos = 24+(target_i - pos_i) * 7 + (target_j - pos_j)
    return attack_pos

def need_to_build(state):
    my_worker_num = 0
    my_barrack_num = 0
    building_worker = 0
    for i in range(16):
        for j in range(16):
            if  state[i][j][11] == 1 and state[i][j][17] == 1:
                my_worker_num = my_worker_num + 1
                if state[i][j][25] == 1:
                    building_worker = building_worker + 1
            if  state[i][j][11] == 1 and state[i][j][16] == 1:
                my_barrack_num = my_barrack_num + 1
    if my_worker_num >= 2 and my_barrack_num == 0 and building_worker == 0:
        return True
    else:
        return  False

def base_need_idle(state):
    my_worker_num = 0
    my_barrack_num = 0
    for i in range(16):
        for j in range(16):
            if state[i][j][11] == 1 and state[i][j][17] == 1:
                my_worker_num = my_worker_num + 1
            if state[i][j][11] == 1 and state[i][j][16] == 1:
                my_barrack_num = my_barrack_num + 1
    if my_worker_num >= 4 and my_barrack_num == 0 :
        return True
    elif my_worker_num >= 5:
        return True
    else:
        return False

def inside(target_pos):
    i = target_pos // 16
    j = target_pos % 16
    if i < 0 or i >15 or j < 0 or j > 15:
        return False
    else:
        return True

def change_turn(pos,turn):
    i = pos // 16
    j = pos % 16
    if i < 13 and j < 13:
        target = [1, 2]
    elif i < 13 and j >=13:
        target = [2, 3]
    elif i >= 13 and j < 13:
        target = [0, 1]
    else:
        target = [0, 3]
    if turn in target:
        return turn
    else:
        p = random.random()
        if p < 0.5:
            return target[0]
        else:
            return target[1]