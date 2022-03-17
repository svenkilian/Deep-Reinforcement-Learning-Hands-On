#!/usr/bin/env python3
import os
import gym
from gym import wrappers
import ptan
import argparse
import numpy as np
import logging
import sys

import torch
import torch.optim as optim

from lib import environ, data, models, common, validation

from tensorboardX import SummaryWriter

BATCH_SIZE = 32
BARS_COUNT = 10
TARGET_NET_SYNC = 1000
# DEFAULT_STOCKS = "data/YNDX_160101_161231.csv"
DEFAULT_STOCKS = "data/BTC_history_n.csv"
# DEFAULT_VAL_STOCKS = "data/YNDX_150101_151231.csv"
DEFAULT_VAL_STOCKS = "data/BTC_history_n.csv"
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

GAMMA = 0.99

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000

UPDATE_FREQ = 5  # Interval to update model parameters
REWARD_STEPS = 2

LEARNING_RATE = 0.0003

STATES_TO_EVALUATE = 1000
EVAL_EVERY_STEP = 1000

EPSILON_START = 1.0
EPSILON_STOP = 0.1
EPSILON_STEPS = 5e5  # was 1000000

CHECKPOINT_EVERY_STEP = 1000000
VALIDATION_EVERY_STEP = 1e4  # Was 100000

logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__.split('.')[0])


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("--data", default=DEFAULT_STOCKS,
                        help="Stocks file or dir to train on, default=" + DEFAULT_STOCKS)
    parser.add_argument(
        "--year", type=int, help="Year to be used for training, if specified, overrides --data option")
    parser.add_argument("--valdata", default=DEFAULT_VAL_STOCKS,
                        help="Stocks data for validation, default=" + DEFAULT_VAL_STOCKS)
    parser.add_argument("-r", "--run", required=True, help="Run name")
    parser.add_argument("--loss", required=False,
                        default='MSE', help="Used loss function", )
    parser.add_argument("--reset-close", default=False,
                        action="store_true", help="End episode on close")
    parser.add_argument("--reward-close", default=False,
                        action="store_true", help="Only reward on close")
    parser.add_argument("--state-1d", default=False, action="store_true",
                        help="Use convolution-ready state representation")
    parser.add_argument("--volumes", default=False, action="store_true",
                        help="Use trading volume as part of observation space")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args(argparse.ArgumentParser())

    # Set flags from args
    reset_on_close = args.reset_close
    reward_on_close = args.reward_close
    use_state_1d = args.state_1d
    use_volumes = args.volumes
    loss_function = args.loss
    logger.info(f'Args: {args}')
    device = torch.device("cuda" if args.cuda else "cpu")

    saves_path = os.path.join(BASE_DIR, "saves", args.run)
    os.makedirs(saves_path, exist_ok=True)

    # Create environment
    if args.year is not None or os.path.isfile(args.data):
        if args.year is not None:
            stock_data = data.load_year_data(args.year)
        else:
            stock_data = {"YNDX": data.load_relative(args.data)}
        env = environ.StocksEnv(stock_data, bars_count=BARS_COUNT,
                                reset_on_close=reset_on_close, reward_on_close=reward_on_close, state_1d=use_state_1d, volumes=use_volumes)
        env_tst = environ.StocksEnv(
            stock_data, bars_count=BARS_COUNT, reset_on_close=reset_on_close, reward_on_close=reward_on_close, state_1d=use_state_1d, volumes=use_volumes)
    elif os.path.isdir(args.data):
        env = environ.StocksEnv.from_dir(
            args.data, bars_count=BARS_COUNT, reset_on_close=reset_on_close, reward_on_close=reward_on_close, state_1d=use_state_1d, volumes=use_volumes)
        env_tst = environ.StocksEnv.from_dir(
            args.data, bars_count=BARS_COUNT, reset_on_close=reset_on_close, reward_on_close=reward_on_close, state_1d=use_state_1d, volumes=use_volumes)
    else:
        raise RuntimeError("No data to train on")
    # Apply time limit wrapper
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    logger.info(f'Created environment: {env}')
    logger.info(f'Stock data size: {len(stock_data.get("YNDX").open)}')

    val_data = {"YNDX": data.load_relative(args.valdata)}
    logger.info(f'Val data size: {len(val_data.get("YNDX").open)}')
    env_val = environ.StocksEnv(
        val_data, bars_count=BARS_COUNT, reset_on_close=reset_on_close, state_1d=use_state_1d)

    # Configure SummaryWriter
    writer = SummaryWriter(comment="-simple-" + "-".join(
        f'{key}={str(val).replace("/", "%")}' for key, val in args._get_kwargs()))

    # Define neural networks, selector, agent, experience source, buffer, and optimizer
    net = models.SimpleFFDQN(
        env.observation_space.shape[0], env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(EPSILON_START)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, GAMMA, steps_count=REWARD_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Main training loop
    step_idx = 0
    eval_states = None
    best_mean_val = None

    with common.RewardTracker(writer, np.inf, group_rewards=100) as reward_tracker:
        while True:
            step_idx += 1
            buffer.populate(1)  # Populate buffer with one sample
            selector.epsilon = max(
                EPSILON_STOP, EPSILON_START - step_idx / EPSILON_STEPS)

            new_rewards = exp_source.pop_rewards_steps()
            if new_rewards:
                reward_tracker.reward(
                    new_rewards[0], step_idx, selector.epsilon)
                # logger.info(
                #     f'New reward: {new_rewards[0][0]} in {new_rewards[0][1]} steps.')

            if len(buffer) < REPLAY_INITIAL:
                # Continue filling buffer with random actions until REPLAY_INITAL steps have been taken
                if len(buffer) == 0:
                    logger.info(
                        f'Initially filling replay buffer with {REPLAY_INITIAL} samples')
                if len(buffer) % 1000 == 0:
                    logger.info(
                        f'Sampled environment: {len(buffer)} interactions')
                continue

            if eval_states is None:
                logger.info(f'Initial buffer populated, start training')
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False)
                               for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)

            if step_idx % EVAL_EVERY_STEP == 0:
                mean_val = common.calc_values_of_states(
                    eval_states, net, device=device)
                writer.add_scalar("values_mean", mean_val, step_idx)
                if best_mean_val is None or best_mean_val < mean_val:
                    if best_mean_val is not None:
                        print("%d: Best mean value updated %.3f -> %.3f" %
                              (step_idx, best_mean_val, mean_val))
                    best_mean_val = mean_val
                    torch.save(net.state_dict(), os.path.join(
                        saves_path, "mean_val-%.3f.data" % mean_val))

            if step_idx % UPDATE_FREQ:
                optimizer.zero_grad()
                # Sample random batch from Experience Replay Buffer
                batch = buffer.sample(BATCH_SIZE)
                loss_v = common.calc_loss(
                    batch, net, tgt_net.target_model, GAMMA ** REWARD_STEPS, loss_func=loss_function, device=device)
                loss_v.backward()
                optimizer.step()

            if step_idx % TARGET_NET_SYNC == 0:
                # Sync target network every TARGET_NET_SYNC steps
                logger.info(
                    f'{step_idx} steps played - syncing target network')
                tgt_net.sync()

            if step_idx % CHECKPOINT_EVERY_STEP == 0:
                # Save model checkpoint every CHECKPOINT_EVERY_STEP steps
                logger.info(f'{step_idx} steps played - saving network')
                idx = step_idx // CHECKPOINT_EVERY_STEP
                torch.save(net.state_dict(), os.path.join(
                    saves_path, "checkpoint-%3d.data" % idx))

            if step_idx % VALIDATION_EVERY_STEP == 0:
                # Perform validation every VALIDATION_EVERY_STEP steps
                logger.info(f'{step_idx} steps played - validating policy')
                res = validation.validation_run(env_tst, net, device=device)
                for key, val in res.items():
                    writer.add_scalar(key + "_test", val, step_idx)
                res = validation.validation_run(env_val, net, device=device)
                for key, val in res.items():
                    writer.add_scalar(key + "_val", val, step_idx)
