import numpy as np
import pandas as pd

from helpers import environ
import tensorforce
from tensorforce import Agent, Environment
import sys


class Validator():
    """
    Validator class
    """

    def __init__(self, env: Environment, agent: tensorforce.Agent, num_episodes=100, commission=None):
        self.stats = {
            'episode_reward': [],
            'episode_profit': [],
        }
        self.environment = env
        self.agent = agent
        if commission is None:
            self.commission_perc = self.environment.environment.commission_perc
        else:
            self.commission_perc = commission
        self.window_size = self.environment.environment.window_size
        self.num_episodes = num_episodes

        self.environment.reset()
        self.environment.environment.random_ofs_on_reset = False
        self.prices = self.environment.environment.prices
        self.dates = self.environment.environment.dates
        print(f'Number of close prices: {len(self.prices)}')

    def run(self):
        """
        Run validation
        """
        print(
            f'Starting Validation Run for {self.num_episodes} episodes ...\n')
        for episode in range(self.num_episodes):
            print(f'Running validation episode: {episode + 1}')

            # Reset counters
            total_reward = 0.0
            total_profit = 1.0
            episode_rewards = []
            episode_profits = []

            # Reset states
            self.environment.environment.random_ofs_on_reset = False
            states = self.environment.reset()
            internals = self.agent.initial_internals()
            terminal = False
            episode_step = 0

            # Loop over validation data
            while not terminal:
                episode_step += 1
                rel_profit = 0

                # Decide on action to take
                actions, internals = self.agent.act(
                    states=states, internals=internals,
                    independent=True, deterministic=True
                )

                actions = self.environment.environment.action_space.sample()

                # Print if relevant action was taken
                if (
                    (actions == environ.Actions.Long.value)
                    or (actions == environ.Actions.Short.value)
                    or (actions == environ.Actions.Flat.value)
                ):
                    print(f'\nStep: {episode_step}')
                    print(
                        f'Date: {self.dates[episode_step - 2 + self.window_size].date()}')
                    print(f'Performing action: {actions}')

                # Get previous and current close prices
                prev_price = self.prices[episode_step - 2 + self.window_size]
                cur_price = self.prices[episode_step - 1 + self.window_size]

                price_diff = cur_price - prev_price
                # print(f'Close price: {close_price}')

                # Update counters based on action taken
                if (
                    (actions == environ.Actions.Long.value)
                    or (actions == environ.Actions.Short.value)
                ):
                    if actions == environ.Actions.Long.value:
                        print(
                            f'Going long at {prev_price:.2f} and selling at {cur_price:.2f}')
                    elif actions == environ.Actions.Short.value:
                        price_diff *= -1
                        print(
                            f'Going short at {prev_price:.2f} and selling at {cur_price:.2f}')

                    shares = environ.TradingEnv.calculate_shares(
                        total_profit=self.environment.environment._total_profit, commission_perc=self.commission_perc, last_trade_price=prev_price)
                    rel_profit = environ.TradingEnv.calculate_rel_profit(price_diff=price_diff,
                                                                         shares=shares, prev_total_profit=self.environment.environment._total_profit, current_price=cur_price, commission_perc=self.commission_perc)
                    # Update total profit
                    total_profit = self.environment.environment._total_profit * \
                        (1 + rel_profit)

                else:
                    print(
                        f'Going flat at {prev_price:.2f} (next price: {cur_price:.2f})')

                # Interact with environment: take action and receive reward
                states, terminal, reward = self.environment.execute(
                    actions=actions)

                episode_profits.append(rel_profit)
                episode_rewards.append(reward)

                # Update reward
                total_reward += reward

                print(f'Reward calcualtion:')
                print(f'Price diff: {price_diff:.4f}')
                print(
                    f'Commission: Buy: {(self.commission_perc * prev_price):.4f}, Sell: {(self.commission_perc * cur_price):.4f}, Total: {(self.commission_perc * (prev_price + cur_price)):.4f}')
                print(
                    f'Total reward: (price_diff - total_commission)/last_trade_price = {reward:.4f}')
                print(f'Gained reward: {reward:.4f}')
                print(f'Total reward: {total_reward:.4f}')
                print(f'Relative profit: {rel_profit:.4f}')
                print(f'Total profit: {total_profit:.4f}')

                # Calculate metrics for last state
                if terminal:
                    pass
                    if False:
                        episode_step += 1
                        prev_price = self.prices[episode_step -
                                                 2 + self.window_size]
                        cur_price = self.prices[episode_step -
                                                1 + self.window_size]
                        shares = environ.TradingEnv.calculate_shares(
                            total_profit=self.environment.environment._total_profit, commission_perc=self.environment.environment.commission_perc, last_trade_price=prev_price)
                        rel_profit = environ.TradingEnv.calculate_rel_profit(price_diff=price_diff,
                                                                             shares=shares, prev_total_profit=self.environment.environment._total_profit, commission_perc=self.commission_perc)
                        # Update total profit
                        total_profit = self.environment.environment._total_profit * \
                            (1 + rel_profit)

            # Update results dict and print results
            self.stats['episode_reward'].append(total_reward)
            self.stats['episode_profit'].append(total_profit)
            print(
                f'\nTotal Reward in Validation Episode {episode}: {total_reward:.4f}')
            print(f'Total Profit in Validation Episode: {total_profit:.4f}')

        return self.stats


# class Validator_():
#     """
#     Validator class
#     """

#     def __init__(self, env: Environment, o_env: environ.StocksEnv, agent: tensorforce.Agent, num_episodes=100, commission=0.0):
#         self.stats = {
#             'episode_reward': [],
#             'episode_steps': [],
#             'order_profits': [],
#             'order_steps': [],
#             'positions': []
#         }
#         self.environment = env
#         self.o_env = o_env
#         self.agent = agent
#         self.commission = commission
#         self.num_episodes = num_episodes

#         self.o_env.reset()
#         self.prices = self.o_env._prices['btc']
#         self.close_prices = [
#             self.prices.open[i] * (1.0 + self.prices.close[i]) for i in range(len(self.prices.close))]
#         print(f'Number of close prices: {len(self.close_prices)}')

#     def run(self):
#         """
#         Run validation
#         """
#         print(
#             f'Starting Validation Run for {self.num_episodes} episodes ...\n')
#         for episode in range(self.num_episodes):
#             print(f'Running validation episode: {episode + 1}')

#             # Reset counters
#             total_reward = 0.0
#             position = None
#             position_steps = None
#             episode_steps = 0

#             # Reset states
#             states = self.environment.reset()
#             internals = self.agent.initial_internals()
#             terminal = False
#             episode_step = 0

#             # Loop over validation data
#             while not terminal:
#                 episode_step += 1

#                 # Decide on action to take
#                 actions, internals = self.agent.act(
#                     states=states, internals=internals,
#                     independent=True, deterministic=True
#                 )

#                 # Print if relevant action was taken
#                 if (
#                     ((actions == environ.Actions.Buy.value)
#                      and (position_steps is None))
#                     or ((actions == environ.Actions.Close.value) and (position_steps is not None))
#                 ):
#                     print(f'\nStep: {episode_step}')
#                     print(
#                         f'Position: {round(position, 2) if position else position} (steps: {position_steps})')
#                     print(f'Performing action: {actions}')

#                 # Get close price
#                 close_price = self.close_prices[episode_step - 1 + 10]
#                 # print(f'Close price: {close_price}')

#                 # Update counters based on action taken
#                 if actions == environ.Actions.Buy.value and position is None:
#                     position = close_price
#                     print(f'Buying at {round(position, 2)}')
#                     # self.profit -= self.position * self.commission
#                     position_steps = 0

#                 elif actions == environ.Actions.Close.value and position is not None:
#                     abs_profit = (close_price - position -
#                                   (close_price + position) * self.commission / 100)
#                     profit = 100.0 * abs_profit / position
#                     self.stats['order_profits'].append(profit)
#                     self.stats['order_steps'].append(position_steps)
#                     print(
#                         f'Selling at {round(close_price, 2)} after {round(position_steps, 2)}')
#                     print(f'Open price was {round(position, 2)}')
#                     print(
#                         f'Absolute profit: {round(abs_profit, 2)} (+ {round(profit, 2)} %)')
#                     position = None
#                     position_steps = None

#                 # Interact with environment: take action and receive reward
#                 states, terminal, reward = self.environment.execute(
#                     actions=actions)

#                 # Update reward
#                 total_reward += reward

#                 if reward > 0:
#                     print(
#                         f'Total reward (step {episode_step}): {round(total_reward, 2)}')
#                 episode_steps += 1

#                 if position_steps is not None:
#                     position_steps += 1

#                 # Calculate metrics for last state
#                 if terminal:
#                     if position is not None:
#                         profit = (close_price - position -
#                                   (close_price + position) * self.commission / 100)
#                         profit = 100.0 * profit / position
#                         self.stats['order_profits'].append(profit)
#                         self.stats['order_steps'].append(position_steps)

#             # Update results dict and print results
#             self.stats['episode_reward'].append(total_reward)
#             self.stats['episode_steps'].append(episode_steps)
#             print(
#                 f'Total Reward in Validation Episode {episode}: {total_reward}')
#             print(f'Mean Episode Reward: {total_reward / self.num_episodes}')

#         return self.stats
