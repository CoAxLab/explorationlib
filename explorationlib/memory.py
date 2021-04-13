import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from collections import deque

from sklearn.neighbors import KernelDensity
from scipy.stats import entropy as scientropy

import random


class NoveltyMemory:
    def __init__(self, bonus=0):
        self.bonus = bonus
        self.memory = []

    def __call__(self, state):
        return self.forward(state)

    def forward(self, state):
        if state in self.memory:
            bonus = 0
        else:
            self.memory.append(state)
            bonus = self.bonus

        return bonus

    def state_dict(self):
        return self.memory

    def load_state_dict(self, state_dict):
        self.memory = state_dict


class GridCountMemory:
    """A simple grid counter."""
    def __init__(self):
        self.memory = dict()

    def __call__(self, pos, obs):
        return self.forward(pos, obs)

    def forward(self, pos, obs):
        # Init?
        if pos not in self.memory:
            self.memory[pos] = CountMemory()

        # Update count in memory
        # and then return it
        self.memory[pos][obs] += 1

        return self.memory[pos][obs]

    def state_dict(self):
        return self.memory

    def load_state_dict(self, state_dict):
        self.memory = state_dict


class CountMemory:
    """A simple counter."""
    def __init__(self):
        self.memory = dict()

    def __call__(self, state):
        return self.forward(state)

    def forward(self, state):
        # Init?
        if state not in self.memory:
            self.memory[state] = 0

        # Update count in memory
        # and then return it
        self.memory[state] += 1

        return self.memory[state]

    def state_dict(self):
        return self.memory

    def load_state_dict(self, state_dict):
        self.memory = state_dict


class DiscreteDistribution:
    """A discrete distribution."""
    def __init__(self, initial_bins=None, initial_count=1):
        # Init the count model
        self.N = 0
        self.initial_count = initial_count
        self.count = OrderedDict()

        # Preinit its values?
        if initial_bins is not None:
            for x in initial_bins:
                self.count[x] = self.initial_count
                self.N += 1

    def __len__(self):
        return len(self.count)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # Init, if necessary
        if x not in self.count:
            self.count[x] = self.initial_count

        # Update the counts
        self.count[x] += 1
        self.N += 1

        # Return prob
        return self.count[x] / self.N

    def keys(self):
        return list(self.count.keys())

    def values(self):
        return list(self.count.values())

    def probs(self):
        if len(self.count) == 0:
            return 0.0
        elif self.N == 0:
            return 0.0
        else:
            probs = []
            for c in self.count.values():
                probs.append(c / self.N)
            return probs

    def state_dict(self):
        return self.count

    def load_state_dict(self, state_dict):
        self.count = state_dict


class DiscreteDistributionGrid:
    def __init__(self, initial_bins=None, initial_count=1):
        self.initial_bins = initial_bins
        self.initial_count = initial_count
        self.grid = dict()

    def __len__(self):
        return len(self.grid)

    def __call__(self, pos, obs):
        return self.forward(pos, obs)

    def _convert_pos(self, pos):
        return tuple(list(pos))

    def forward(self, pos, obs):
        # Init, if necessary
        pos = self._convert_pos(pos)
        if pos not in self.grid:
            self.grid[pos] = DiscreteDistribution(self.initial_bins,
                                                  self.initial_count)
        # Update the counts
        prob = self.grid[pos](obs)
        return prob

    def probs(self, pos):
        pos = self._convert_pos(pos)
        if pos not in self.grid:
            self.grid[pos] = DiscreteDistribution(self.initial_bins,
                                                  self.initial_count)
        return self.grid[pos].probs()

    def keys(self):
        return list(self.grid.keys())

    def values(self):
        return list(self.grid.values())

    def state_dict(self):
        return self.grid

    def load_state_dict(self, state_dict):
        self.grid = state_dict


class EntropyMemory:
    """Estimate policy entropy."""
    def __init__(self, initial_bins=None, initial_count=1, base=None):
        # Init the count model
        if initial_bins is None:
            self.N = 1
        else:
            self.N = len(initial_bins)

        self.base = base
        self.initial_count = initial_count
        self.memory = dict()

        # Preinit its values?
        if initial_bins is not None:
            for x in initial_bins:
                self.memory[x] = self.initial_count

    def __call__(self, action):
        return self.forward(action)

    def forward(self, action):
        # Init?
        if action not in self.memory:
            self.memory[action] = self.initial_count

        # Update count in memory
        self.N += 1
        self.memory[action] += 1

        # Estimate H
        self.probs = [(n / self.N) for n in self.memory.values()]
        return scientropy(np.asarray(self.probs), base=self.base)

    def state_dict(self):
        return self.memory

    def load_state_dict(self, state_dict):
        self.memory = state_dict
