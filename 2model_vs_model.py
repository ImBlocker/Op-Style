# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import argparse
import datetime
import itertools
import os
import random
import shutil
import uuid
from distutils.util import strtobool
from enum import Enum

import numpy as np
import pandas as pd
import torch
import get_action

import json

from peewee import (
    JOIN,
    CharField,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    Model,
    SmallIntegerField,
    SqliteDatabase,
    fn,
)
from stable_baselines3.common.vec_env import VecMonitor,VecVideoRecorder
from trueskill import Rating, quality_1vs1, rate_1vs1

from gym_microrts import microrts_ai  # fmt: off
import  htn_network

torch.set_num_threads(1)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")

    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, the game will have partial observability')
    parser.add_argument('--evals', nargs='+', default=["randomBiasedAI","workerRushAI","lightRushAI", "coacAI"], # [],
        help='the ais')
    parser.add_argument('--num-matches', type=int, default=10,
        help='seed of the experiment')
    parser.add_argument('--update-db', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, the database will be updated')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--highest-sigma', type=float, default=1.4,
        help='the highest sigma of the trueskill evaluation')
    parser.add_argument('--output-path', type=str, default=f"league.temp.csv",
        help='the output path of the leaderboard csv')
    parser.add_argument('--model-type', type=str, default=f"ppo", choices=["ppo", "ppocpm"],
        help='the output path of the leaderboard csv')
    parser.add_argument('--maps', nargs='+', default=["maps/16x16/basesWorkers16x16A.xml"],
        help="the maps to do trueskill evaluations")
    # ["randomBiasedAI","workerRushAI","lightRushAI","coacAI"]
    # default=["randomBiasedAI","workerRushAI","lightRushAI","coacAI","randomAI","passiveAI","naiveMCTSAI","mixedBot","rojo","izanagi","tiamat","droplet","guidedRojoA3N"]
    args = parser.parse_args()
    # fmt: on
    return args


args = parse_args()
dbname = "league"
if args.partial_obs:
    dbname = "po_league"
dbpath = f"gym-microrts-static-files/{dbname}.db"
csvpath = f"gym-microrts-static-files/{dbname}.csv"
if not args.update_db:
    if not os.path.exists(f"gym-microrts-static-files/tmp"):
        os.makedirs(f"gym-microrts-static-files/tmp")
    tmp_dbpath = f"gym-microrts-static-files/tmp/{str(uuid.uuid4())}.db"
    shutil.copyfile(dbpath, tmp_dbpath)
    dbpath = tmp_dbpath
db = SqliteDatabase(dbpath)

if args.model_type == "ppo_gridnet_large":
    from wan0 import Agent, MicroRTSStatsRecorder

    from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
else:
    from wan0 import Agent, MicroRTSStatsRecorder

    from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv


class BaseModel(Model):
    class Meta:
        database = db


class AI(BaseModel):
    name = CharField(unique=True)
    mu = FloatField()
    sigma = FloatField()
    ai_type = CharField()

    def __str__(self):
        return f"🤖 {self.name} with N({round(self.mu, 3)}, {round(self.sigma, 3)})"


class MatchHistory(BaseModel):
    challenger = ForeignKeyField(AI, backref="challenger_match_histories")
    defender = ForeignKeyField(AI, backref="defender_match_histories")
    win = SmallIntegerField()
    draw = SmallIntegerField()
    loss = SmallIntegerField()
    created_date = DateTimeField(default=datetime.datetime.now)


db.connect()
db.create_tables([AI, MatchHistory])


class Outcome(Enum):
    WIN = 1
    DRAW = 0
    LOSS = -1


class Match:
    def __init__(self, partial_obs: bool, match_up=None, map_path="maps/16x16/basesWorkers16x16A.xml"):
        # mode 0: rl-ai vs built-in-ai
        # mode 1: rl-ai vs rl-ai
        # mode 2: built-in-ai vs built-in-ai

        built_in_ais = None
        built_in_ais2 = None
        rl_ai = None
        rl_ai2 = None
        self.map_path = map_path

        # determine mode
        rl_ais = []
        built_in_ais = []
        for ai in match_up:
            if ai[-3:] == ".pt":
                rl_ais += [ai]
            else:
                built_in_ais += [ai]
        if len(rl_ais) == 1:
            mode = 0
            p0 = rl_ais[0]
            p1 = built_in_ais[0]
            rl_ai = p0
            built_in_ais = [eval(f"microrts_ai.{p1}")]
        elif len(rl_ais) == 2:
            mode = 1
            p0 = rl_ais[0]
            p1 = rl_ais[1]
            rl_ai = p0
            rl_ai2 = p1
        else:
            mode = 2
            p0 = built_in_ais[0]
            p1 = built_in_ais[1]
            built_in_ais = [eval(f"microrts_ai.{p0}")]
            built_in_ais2 = [eval(f"microrts_ai.{p1}")]

        self.p0, self.p1 = p0, p1

        self.mode = mode
        self.partial_obs = partial_obs
        self.built_in_ais = built_in_ais
        self.built_in_ais2 = built_in_ais2
        self.rl_ai = rl_ai
        self.rl_ai2 = rl_ai2
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        max_steps = 500

        if mode == 0:
            self.envs = MicroRTSGridModeVecEnv(
                num_bot_envs=len(built_in_ais),
                num_selfplay_envs=0,
                partial_obs=self.partial_obs,
                max_steps=max_steps,
                render_theme=2,
                ai2s=built_in_ais,
                map_paths=[map_path],
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
                autobuild=False,
            )
            self.agent = Agent(self.envs).to(self.device)
            self.agent.load_state_dict(torch.load(self.rl_ai, map_location=self.device))
            self.agent.eval()
        elif mode == 1:
            self.envs = MicroRTSGridModeVecEnv(
                num_bot_envs=0,
                num_selfplay_envs=2,
                partial_obs=self.partial_obs,
                max_steps=max_steps,
                render_theme=2,
                ai2s=np.array(built_in_ais),
                map_paths=[map_path],
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
                autobuild=False,
            )
            self.agent = Agent(self.envs).to(self.device)
            self.agent.load_state_dict(torch.load(self.rl_ai, map_location=self.device))
            self.agent.eval()
            self.agent2 = Agent(self.envs).to(self.device)
            self.agent2.load_state_dict(torch.load(self.rl_ai2, map_location=self.device))
            self.agent2.eval()
        else:
            self.envs = MicroRTSBotVecEnv(
                ai1s=built_in_ais,
                ai2s=built_in_ais2,
                max_steps=max_steps,
                render_theme=2,
                map_paths=[map_path],
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
                autobuild=False,
            )
        self.envs = MicroRTSStatsRecorder(self.envs)
        self.envs = VecMonitor(self.envs)
        # self.envs = VecVideoRecorder(
        #     self.envs, f"videos/{'sth_test'}", record_video_trigger=lambda x: x % 100000 == 0, video_length=2000
        # )



    def run(self, num_matches=7):
        if self.mode == 0:
            return self.run_m0(num_matches)
        elif self.mode == 1:
            return self.run_m1(num_matches)
        else:
            return self.run_m2(num_matches)



    # Process data


    def run(self, num_matches=1):
        if self.mode == 0:
            results, game_history = self.run_m0(num_matches)
        elif self.mode == 1:
            results, game_history = self.run_m1(num_matches)
        else:
            results, game_history = self.run_m2(num_matches)

        # Generate a unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'game_history_{timestamp}.json'

        # Save game history to JL folder
        self.save_game_history(filename,results)

        return results, game_history  # Return two values



    def save_game_history(self, filename,results):
        # Ensure JL folder exists
        folder_path = 'JL/PPOG'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)



        # Complete file path
        file_path = os.path.join(folder_path, filename)

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.int32):
                return int(obj)
            elif isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        def flatten_to_1d_with_subarrays(multi_dim_array):
            flattened = []
            for sub_array in multi_dim_array:
                if isinstance(sub_array, list) and any(isinstance(item, list) for item in sub_array):
                    # If the subarray is multidimensional, flatten it
                    flattened.extend(flatten_to_1d_with_subarrays(sub_array))
                else:
                    # If the subarray is one-dimensional or the smallest array, keep it unchanged
                    flattened.append(sub_array)
            return flattened


        def tensor_to_list(tensor):
            if isinstance(tensor, torch.Tensor):
                return tensor.cpu().numpy().tolist()
            elif isinstance(tensor, np.ndarray):
                return tensor.tolist()
            elif isinstance(tensor, list):
                return [tensor_to_list(item) for item in tensor]
            else:
                return tensor

        def process_data(game_history):
            processed_data = []
            for item in game_history:
                processed_item = {

                    "step": item['step'],
                    "observations": tensor_to_list(item['observations']),  # Current observation
                    "actions": tensor_to_list(item['actions']),  # Current action
                    "rewards": item['rewards'],  # Current reward
                    "dones": item['dones'],  # Current done flag
                    "infos": item['infos']  # Current additional information
                }
                processed_data.append(processed_item)
            return processed_data



        game_history_serializable = convert_to_serializable(self.game_history)

        processed_data = process_data(game_history_serializable)

        # for i, result in enumerate(results):
        #     if i < len(processed_data):
        #         processed_data[i]['result'] = result

        with open(file_path, 'w') as f:
            json.dump(processed_data, f, indent=4)


    def run_m0(self, num_matches):
        results = []
        self.game_history = []  # Reset game history
        next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        step = 0
        while True:
            step += 1
            # self.envs.render()
            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                mask = torch.tensor(np.array(self.envs.get_action_mask())).to(self.device)
                action, _, _, _, _ = self.agent.get_action_and_value(
                    next_obs, envs=self.envs, invalid_action_masks=mask, device=self.device
                )

            try:
                next_obs, rs, ds, infos = self.envs.step(action.cpu().numpy().reshape(self.envs.num_envs, -1))
                next_obs = torch.Tensor(next_obs).to(self.device)
            except Exception as e:
                print(e)
                raise

            # Record current step
            self.game_history.append({
                'step': step,
                'observation': next_obs.cpu().numpy().tolist(),  # Convert to list
                'action': action.cpu().numpy().tolist(),  # Convert to list
                # 'done': ds,
                # 'info': infos
            })

            for idx, info in enumerate(infos):
                if "episode" in info.keys():
                    results += [info["microrts_stats"]["WinLossRewardFunction"]]
                    if len(results) >= num_matches:
                        return results, self.game_history  # Return game history


    def run_m1(self, num_matches):
        results = []
        self.game_history = []  # Reset game history
        next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        step = 0
        while True:
            step += 1
            with torch.no_grad():
                mask, source_unit_mask = self.envs.get_action_mask()
                mask = torch.tensor(np.array(mask)).to(self.device)
                p1_obs = next_obs[::2]
                p2_obs = next_obs[1::2]
                # p1_mask = mask[::2]
                # p2_mask = mask[1::2]
                # # print("p2_mask:", p2_mask.shape)
                #
                # # print(f"p1_obs shape: {p1_obs.shape}")
                # Select first 27 channels
                p1_obs = p1_obs[:, :, :, :27]
                # p2_obs = p2_obs[:, :, :, :27]
                # Adjust dimension order to (batch_size, channels, height, width)
                p1_obs = p1_obs.permute(0, 1, 2, 3)
                p2_obs = p2_obs.permute(0, 1, 2, 3)


                # # Make sure the shape of the passed data is (16, 16, 27)
                # p1_obs = p1_obs[0].permute(1, 2, 0)


                p1_action = get_action.get_action_type1(p1_obs.cpu().numpy(), get_action.GLOBAL_VALUE())
                p2_action, _, _, _, _ = self.agent2.get_action_and_value(
                    p2_obs, envs=self.envs, invalid_action_masks=mask, device=self.device
                )

                action = torch.zeros((self.envs.num_envs, p2_action.shape[1], p2_action.shape[2]))
                action[::2] = torch.tensor(p1_action)
                action[1::2] = p2_action

            try:
                next_obs, rs, ds, infos = self.envs.step(action.cpu().numpy().reshape(self.envs.num_envs, -1))
                next_obs = torch.Tensor(next_obs).to(self.device)
            except Exception as e:
                print(e)
                raise

            # Record current step
            self.game_history.append({
                # 'step': step,
                # 'p1': {
                #     'observations': p1_obs.cpu().numpy().tolist(),  # Player One all observations
                #     'actions': p1_action  # Player One all actions
                # },
                # 'p2': {
                #     'observations': p2_obs.cpu().numpy().tolist(),  # Player Two all observations
                #     'actions': p2_action  # Player Two all actions
                # },
                # 'done': ds,
                # 'info': infos
                "step": step,
                "observations": next_obs.cpu().numpy().tolist(),  # Current observation
                "actions": action.cpu().numpy().tolist(),  # Current action
                "rewards": rs.tolist(),  # Current reward
                "dones": ds.tolist(),  # Current done flag
                "infos": infos  # Current additional information

            })

            for idx, info in enumerate(infos):
                if "episode" in info.keys():
                    # Update info of last step
                    # self.game_history[-1]["info"] = info
                    results += [info["microrts_stats"]["WinLossRewardFunction"]]
                    if len(results) >= num_matches:
                        return results, self.game_history  # Return game history

    def run_m2(self, num_matches):
        results = []
        self.game_history = []  # Reset game history
        self.envs.reset()
        step = 0
        while True:
            step += 1
            # self.envs.render()
            # dummy actions
            next_obs, reward, done, infos = self.envs.step(
                [
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ]
            )
            next_obs = torch.Tensor(next_obs).to(self.device)

            # Record current step
            self.game_history.append({
                'step': step,
                'observation': next_obs.cpu().numpy().tolist(),  # Convert to list
                'action': [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],  # Convert to list
                'done': done,
                # 'info': infos
            })

            for idx, info in enumerate(infos):
                if "episode" in info.keys():
                    results += [info["microrts_stats"]["WinLossRewardFunction"]]
                    if len(results) >= num_matches:
                        return results, self.game_history  # Return game history

def get_ai_type(ai_name):
    if ai_name[-3:] == ".pt":
        return "rl_ai"
    else:
        return "built_in_ai"


def get_match_history(ai_name):
    query = (
        MatchHistory.select(
            AI.name,
            fn.SUM(MatchHistory.win).alias("wins"),
            fn.SUM(MatchHistory.draw).alias("draws"),
            fn.SUM(MatchHistory.loss).alias("losss"),
        )
        .join(AI, JOIN.LEFT_OUTER, on=MatchHistory.defender)
        .group_by(MatchHistory.defender)
        .where(MatchHistory.challenger == AI.get(name=ai_name))
    )
    return pd.DataFrame(list(query.dicts()))


def get_leaderboard():
    query = AI.select(
        AI.name,
        AI.mu,
        AI.sigma,
        (AI.mu - 3 * AI.sigma).alias("trueskill"),
    ).order_by((AI.mu - 3 * AI.sigma).desc())
    return pd.DataFrame(list(query.dicts()))


def get_leaderboard_existing_ais(existing_ai_names):
    query = (
        AI.select(
            AI.name,
            AI.mu,
            AI.sigma,
            (AI.mu - 3 * AI.sigma).alias("trueskill"),
        )
        .where((AI.name.in_(existing_ai_names)))
        .order_by((AI.mu - 3 * AI.sigma).desc())
    )
    return pd.DataFrame(list(query.dicts()))


if __name__ == "__main__":
    otheragent = ['workerRushAI', 'rojo', 'mixedBot', 'guidedRojoA3N', 'randomAI', 'lightRushAI', 'coacAI', 'izanagi',
                  'naiveMCTSAI', 'droplet',
                  'passiveAI', 'tiamat', 'randomBiasedAI']
    path1 = 'models/MicroRTSGridModeVecEnv__ppo_gridnet__1__1749430260/43302912.pt'
    path2 = 'models/MicroRTSGridModeVecEnv__ppo_gridnet__1__1749430260/43302912.pt'
    match_up = [path1, path2]
    m = Match(args.partial_obs, match_up, args.maps[0])
    win = 0
    loss = 0
    draw = 0
    print('myai VS sota')
    for i in range(334):
        result, game_history = m.run(1)
        # print('Game History:', game_history)
        print('result', i, '=', result)
        if result[0] > 0:
            win = win + 1
        elif result[0] < 0:
            loss = loss + 1
        else:
            draw = draw + 1
    print('win=', win)
    print('loss=', loss)
    print('draw=', draw)