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


class CountMemory:
    """A simple state counter."""
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
