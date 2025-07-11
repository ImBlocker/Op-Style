# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import argparse
import os
import random
import time
import json
import datetime
from distutils.util import strtobool

import numpy as np
import torch
from gym.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from torch.utils.tensorboard import SummaryWriter

from gym_microrts import microrts_ai  # noqa
import sys


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicroRTSGridModeVecEnv",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=1000000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='whether to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, the game will have partial observability')
    parser.add_argument('--num-steps', type=int, default=256,
        help='the number of steps per game environment')
    parser.add_argument("--agent-model-path", type=str, default="gym-microrts-static-files/agent_sota.pt",
        help="the path to the agent's model")
    parser.add_argument("--agent2-model-path", type=str, default="gym-microrts-static-files/agent_sota.pt",
        help="the path to the agent's model")
    parser.add_argument('--ai', type=str, default="",
        help='the opponent AI to evaluate against')
    parser.add_argument('--model-type', type=str, default=f"ppo_gridnet", choices=["ppo_gridnet_large", "ppo_gridnet"],
        help='the output path of the leaderboard csv')
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    if args.ai:
        args.num_bot_envs, args.num_selfplay_envs = 1, 0
    else:
        args.num_bot_envs, args.num_selfplay_envs = 0, 2
    args.num_envs = args.num_selfplay_envs + args.num_bot_envs
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_updates = args.total_timesteps // args.batch_size
    # fmt: on
    return args


def save_game_history(filename, game_history):
    # Ensure the folder exists
    folder_path = '/mnt/671cbd8b-55cf-4eb4-af6d-a4ab48e8c9d2/JL/1PP'
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
                "observations": tensor_to_list(item['observations']),
                "actions": tensor_to_list(item['actions']),
                "rewards": item['rewards'],
                "dones": item['dones'],
                "infos": item['infos']
            }
            processed_data.append(processed_item)
        return processed_data

    game_history_serializable = convert_to_serializable(game_history)
    processed_data = process_data(game_history_serializable)

    with open(file_path, 'w') as f:
        json.dump(processed_data, f, indent=4)


if __name__ == "__main__":
    args = parse_args()

    if args.model_type == "ppo_gridnet_large":
        from ppo_gridnet_large import Agent, MicroRTSStatsRecorder
        print("Using ppo_gridnet_large")

        from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
    else:
        from ppo_gridnet import Agent, MicroRTSStatsRecorder
        print("Using ppo_gridnet")

        from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

    # TRY NOT TO MODIFY: setup the environment
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.prod_mode:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=experiment_name,
            monitor_gym=True,
            save_code=True,
        )
        CHECKPOINT_FREQUENCY = 10
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text(
        "hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )

    # TRY NOT TO MODIFY: seeding
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    ais = []
    if args.ai:
        ais = [eval(f"microrts_ai.{args.ai}")]
    envs = MicroRTSGridModeVecEnv(
        num_bot_envs=len(ais),
        num_selfplay_envs=args.num_selfplay_envs,
        partial_obs=args.partial_obs,
        max_steps=5000,
        render_theme=2,
        ai2s=ais,
        map_paths=["maps/16x16/basesWorkers16x16A.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    )
    envs = MicroRTSStatsRecorder(envs)
    envs = VecMonitor(envs)
    if args.capture_video:
        envs = VecVideoRecorder(
            envs, f"videos/{experiment_name}", record_video_trigger=lambda x: x % 100000 == 0, video_length=2000
        )
    assert isinstance(envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"

    agent = Agent(envs).to(device)
    agent2 = Agent(envs).to(device)

    # ALGO Logic: Storage for epoch data
    mapsize = 16 * 16
    invalid_action_shape = (mapsize, envs.action_plane_space.nvec.sum())

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    ## CRASH AND RESUME LOGIC:
    starting_update = 1
    agent.load_state_dict(torch.load(args.agent_model_path, map_location=device))
    agent.eval()
    if not args.ai:
        agent2.load_state_dict(torch.load(args.agent2_model_path, map_location=device))
        agent2.eval()

    print("Model's state_dict:")
    for param_tensor in agent.state_dict():
        print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
    total_params = sum([param.nelement() for param in agent.parameters()])
    print("Model's total parameters:", total_params)

    game_history = []  # Initialize game history recording

    for update in range(starting_update, args.num_updates + 1):
        for step in range(0, args.num_steps):
            envs.render()
            global_step += 1 * args.num_envs
            with torch.no_grad():
                invalid_action_masks = torch.tensor(np.array(envs.get_action_mask())).to(device)

                if args.ai:
                    print("111")
                    action, logproba, _, _, vs = agent.get_action_and_value(
                        next_obs, envs=envs, invalid_action_masks=invalid_action_masks, device=device
                    )
                else:
                    p1_obs = next_obs[::2]
                    p2_obs = next_obs[1::2]
                    p1_mask = invalid_action_masks[::2]
                    p2_mask = invalid_action_masks[1::2]

                    p1_action, _, _, _, _ = agent.get_action_and_value(
                        p1_obs, envs=envs, invalid_action_masks=p1_mask, device=device
                    )
                    p2_action, _, _, _, _ = agent2.get_action_and_value(
                        p2_obs, envs=envs, invalid_action_masks=p2_mask, device=device
                    )
                    action = torch.zeros((args.num_envs, p2_action.shape[1], p2_action.shape[2]))
                    action[::2] = p1_action
                    action[1::2] = p2_action

            try:
                next_obs, rs, ds, infos = envs.step(action.cpu().numpy().reshape(envs.num_envs, -1))
                next_obs = torch.Tensor(next_obs).to(device)
            except Exception as e:
                e.printStackTrace()
                raise

            # Record current step
            game_history.append({
                "step": step,
                "observations": next_obs.cpu().numpy().tolist(),
                "actions": action.cpu().numpy().tolist(),
                "rewards": rs.tolist(),
                "dones": ds.tolist(),
                "infos": infos
            })

            # Check if a round has ended
            for idx, info in enumerate(infos):
                if "episode" in info.keys():
                    # Save the game history for the current round
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f'game_history_{timestamp}_match_{idx}.json'
                    save_game_history(filename, game_history)

                    # Reset game_history to record the next round
                    game_history = []

                    # Print the result of the match
                    if args.ai:
                        print("against", args.ai, info["microrts_stats"]["WinLossRewardFunction"])
                    else:
                        if idx % 2 == 0:
                            print(f"player{idx % 2}", info["microrts_stats"]["WinLossRewardFunction"])

    envs.close()
    writer.close()