#!/usr/bin/env python3
import gym
import time
import sys
import ptan
import numpy as np
import argparse
from tensorboardX import SummaryWriter
from stable_baselines3 import A2C, PPO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 8

REWARD_STEPS = 10


class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


def print_v(text: str, verbose=False):
    """Print text to standard out only if verbose equals `True`

    Args:
        text (str): Text to print
        verbose (bool, optional): Verbosity flag. Defaults to False.
    """
    if verbose:
        print(text)


def act_on_interrupt(func):
    def wrapper():
        try:
            func()
        except KeyboardInterrupt as kbi:
            print('\nRegistered keyboard interruption')
            print('Shutting down execution')

    return wrapper


def pause_exec(speed: float):
    if speed > 0:
        time.sleep(speed)


if __name__ == "__main__":

    # ADDED: Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--speed", type=float, default=0.0,
                        help="Execution pause")
    parser.add_argument("--verbose", default=False, action="store_true",
                        help="Execute in verbose mode")
    parser.add_argument("--stable", default=False, action="store_true",
                        help="Use Stable Baselines3 in backend")
    args = parser.parse_args()

    verbose = args.verbose
    speed = args.speed

    if args.stable:
        env = gym.make("CartPole-v0")
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=10000)

        obs = env.reset()
        for i in range(1000):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()

        sys.exit(0)

    @act_on_interrupt
    def main():
        print_v(args, verbose=verbose)

        print_v('Creating env ...', verbose)
        env = gym.make("CartPole-v0")
        print_v('Creating SummaryWriter ...', verbose)
        writer = SummaryWriter(comment="-cartpole-pg")

        net = PGN(env.observation_space.shape[0], env.action_space.n)
        print_v('Creating Neural Network ...', verbose)
        print(net)

        print_v('Creating agent ...', verbose)
        agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,
                                       apply_softmax=True)

        print_v('Creating experience source ...', verbose)
        exp_source = ptan.experience.ExperienceSourceFirstLast(
            env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

        total_rewards = []
        step_rewards = []
        step_idx = 0
        done_episodes = 0
        reward_sum = 0.0

        print_v('Resetting batch_states, batch_actions and batch_scales ...', verbose)
        batch_states, batch_actions, batch_scales = [], [], []

        for step_idx, exp in enumerate(exp_source):
            print_v(f'Taking step {step_idx} ...', verbose)
            print_v(f'Return sample: {exp}', verbose)

            reward_sum += exp.reward
            baseline = reward_sum / (step_idx + 1)
            writer.add_scalar("baseline", baseline, step_idx)
            batch_states.append(exp.state)
            batch_actions.append(int(exp.action))
            batch_scales.append(exp.reward - baseline)

            # handle new rewards
            new_rewards = exp_source.pop_total_rewards()
            pause_exec(speed)

            if new_rewards:
                print_v(
                    f'Receiving total reward for completed episode: {new_rewards} ...', verbose)
                done_episodes += 1
                reward = new_rewards[0]
                total_rewards.append(reward)
                mean_rewards = float(np.mean(total_rewards[-100:]))
                print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                    step_idx, reward, mean_rewards, done_episodes))
                writer.add_scalar("reward", reward, step_idx)
                writer.add_scalar("reward_100", mean_rewards, step_idx)
                writer.add_scalar("episodes", done_episodes, step_idx)
                if mean_rewards > 195:
                    print("Solved in %d steps and %d episodes!" %
                          (step_idx, done_episodes))
                    break

            if len(batch_states) < BATCH_SIZE:
                continue

            states_v = torch.FloatTensor(batch_states)
            batch_actions_t = torch.LongTensor(batch_actions)
            batch_scale_v = torch.FloatTensor(batch_scales)

            optimizer.zero_grad()
            logits_v = net(states_v)
            log_prob_v = F.log_softmax(logits_v, dim=1)
            log_prob_actions_v = batch_scale_v * \
                log_prob_v[range(BATCH_SIZE), batch_actions_t]
            loss_policy_v = -log_prob_actions_v.mean()

            prob_v = F.softmax(logits_v, dim=1)
            entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
            entropy_loss_v = -ENTROPY_BETA * entropy_v
            loss_v = loss_policy_v + entropy_loss_v

            loss_v.backward()
            optimizer.step()

            # calc KL-div
            new_logits_v = net(states_v)
            new_prob_v = F.softmax(new_logits_v, dim=1)
            kl_div_v = -((new_prob_v / prob_v).log()
                         * prob_v).sum(dim=1).mean()
            writer.add_scalar("kl", kl_div_v.item(), step_idx)

            grad_max = 0.0
            grad_means = 0.0
            grad_count = 0
            for p in net.parameters():
                grad_max = max(grad_max, p.grad.abs().max().item())
                grad_means += (p.grad ** 2).mean().sqrt().item()
                grad_count += 1

            writer.add_scalar("baseline", baseline, step_idx)
            writer.add_scalar("entropy", entropy_v.item(), step_idx)
            writer.add_scalar("batch_scales", np.mean(batch_scales), step_idx)
            writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
            writer.add_scalar("loss_policy", loss_policy_v.item(), step_idx)
            writer.add_scalar("loss_total", loss_v.item(), step_idx)
            writer.add_scalar("grad_l2", grad_means / grad_count, step_idx)
            writer.add_scalar("grad_max", grad_max, step_idx)

            batch_states.clear()
            batch_actions.clear()
            batch_scales.clear()

        writer.close()

    main()
