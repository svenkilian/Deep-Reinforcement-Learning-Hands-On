import gym
from gym import spaces
from gym.envs.registration import EnvSpec
from gym.utils import seeding
import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
from typing import Union, Tuple


class Actions(Enum):
    """
    Actions agents can take in the environment
    """
    Short = 0
    Long = 1
    Flat = 2


class Positions(Enum):
    """
    Positions agents can hold
    """
    Short = 0
    Long = 1
    Neutral = 2

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    spec = EnvSpec("TradingEnv-v0")

    def __init__(self, window_size, commission_perc=0.01, random_ofs_on_reset=True, date_range: Union[None, Tuple[str, str]] = None):
        self.seed()
        # self.df = df
        self.window_size = window_size
        self.frame_bound = [self.window_size, None]
        self.date_range = date_range
        self.dates, self.prices, self.signal_features = self._process_data()
        assert self.df.ndim == 2
        self.commission_perc = commission_perc
        self.random_ofs_on_reset = random_ofs_on_reset

        self.shape = (1 * self.window_size, )
        # print(self.shape)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(n=len(Actions))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.shape,
            dtype=np.float32
        )

        # Episode
        self._start_tick = self.window_size - 1
        self._end_tick = len(self.prices) - 1
        self._offset = 0
        self._done = None
        self._current_tick = None
        self._prev_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

    def reset(self) -> np.ndarray:
        """
        Reset environment internals and return first observation
        """
        if self.random_ofs_on_reset:
            self._offset = self.np_random.choice(
                self.prices.shape[0] - self.window_size * 10)
        else:
            self._offset = 0
        self._done = False
        self._current_tick = self._start_tick + self._offset
        self._prev_tick = self._current_tick - 1
        self._position_history = ((self.window_size - 1) * [None])
        self._total_reward = 0.0
        self._total_profit = 1.0  # unit
        self._first_rendering = True
        self.history = {}

        return self._get_observation()

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take action in environment

        Args:
            action (Actions): Action to take in current step

        Returns:
            Tuple: Tuple of (observation, step_reward, self._done, info)
        """
        self._done = False
        # print(
        #     f'Taking step in tick {self._current_tick} ({self.dates[self._current_tick].date()})')
        self._prev_tick = self._current_tick
        self._current_tick += 1  # Increase current tick

        if self._current_tick == self._end_tick:
            self._done = True

        # Calculate immediate reward for current action
        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)
        self._position = action

        self._position_history.append(self._position)

        # Get new observation and update history
        observation = self._get_observation()
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position,
            offset=self._offset
        )
        self._update_history(info)

        return observation, step_reward, self._done, info

    def _get_observation(self) -> np.ndarray:
        return self.signal_features[(self._current_tick + 1 - self.window_size):self._current_tick + 1]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _process_data(self) -> Tuple[np.ndarray, np.ndarray]:
        self.df = pd.read_csv('../data/BTC_history_n.csv',
                              sep=',', parse_dates=True, index_col=0)
        if self.date_range is not None:
            self.df = self.df.reindex(pd.date_range(
                start=self.date_range[0], end=self.date_range[1], freq='d'))
        dates = self.df.index
        prices = self.df.loc[:, 'close'].to_numpy(dtype=np.float32)
        pct_change = self.df.loc[:, 'close'].pct_change().replace(
            np.nan, 0.0).to_numpy(dtype=np.float32)
        if self.frame_bound[1] is None:
            self.frame_bound[1] = len(prices)

        assert self.frame_bound[0] - self.window_size == 0
        prices = prices[self.frame_bound[0] -
                        self.window_size:self.frame_bound[1]]

        # diff = np.insert(np.diff(prices), 0, 0)
        diff = pct_change
        # signal_features = np.column_stack((prices, diff))
        signal_features = diff

        return dates, prices, signal_features

    def _calculate_reward(self, action) -> float:
        """
        Calculate step reward based on action

        Args:
            action (Actions): Action taken in time step

        Returns:
            float: Step reward
        """
        step_reward = 0.0

        if action in [Actions.Long.value, Actions.Short.value]:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._prev_tick]
            price_diff = current_price - last_trade_price
            if action == Actions.Short.value:
                price_diff *= -1
            perc_diff = (price_diff - self.commission_perc * (
                         last_trade_price + current_price)) / last_trade_price
            step_reward = perc_diff

        return step_reward

    @staticmethod
    def calculate_rel_profit(price_diff: float, shares: float, prev_total_profit: float, current_price: float, commission_perc: float):
        relative_profit = (
            shares * (price_diff - commission_perc * current_price)) / prev_total_profit

        return relative_profit

    @staticmethod
    def calculate_shares(total_profit: float, commission_perc: float, last_trade_price: float):
        return total_profit * (1 - commission_perc) / last_trade_price

    def _update_profit(self, action):
        trade = False
        if action in [Actions.Long.value, Actions.Short.value]:
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._prev_tick]
            price_diff = current_price - last_trade_price
            # perc_diff = (price_diff - self.commission_perc *
            #              last_trade_price) / last_trade_price

            # Calculate relative number of shares bought based on previous total profit
            shares = self.calculate_shares(
                total_profit=self._total_profit, commission_perc=self.commission_perc, last_trade_price=last_trade_price)

            rel_profit = self.calculate_rel_profit(price_diff=price_diff,
                                                   shares=shares, prev_total_profit=self._total_profit, current_price=current_price, commission_perc=self.commission_perc)
            # Update total profit
            self._total_profit = self._total_profit * (1 + rel_profit)

    def max_possible_profit(self) -> float:  # Trade fees are ignored
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.0

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit

    def seed(self, seed=None) -> list:
        self.np_random, seed = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed + 1) % 2 ** 31
        return [seed, seed2]

    def close(self):
        plt.close()

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short.value:
                color = 'red'
            elif position == Positions.Long.value:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)

    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        # print(f'Position history:')
        # print(self._position_history)

        fig, ax = plt.subplots(figsize=(18, 12))
        plot = plt.plot(self.dates, self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short.value:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long.value:
                long_ticks.append(tick)

        plt.plot(self.dates[short_ticks], self.prices[short_ticks], 'ro')
        plt.plot(self.dates[long_ticks], self.prices[long_ticks], 'go')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()
