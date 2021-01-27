import numpy as np

from copy import deepcopy
from scipy.stats import powerlaw
from collections import defaultdict


class Agent2d:
    """API stub."""
    def __init__(self):
        self.seed()
        self.history = defaultdict(list)

    def seed(self, seed=None):
        """Init RandomState"""
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def _angle(self, state):
        """Sample angle"""
        return self.np_random.uniform(0, 2 * np.pi)

    def _l(self, state):
        return NotImplementedError("Stub.")

    def _convert(self, angle, l):
        """Convert from (angle, l) to (x, y)"""
        dx = l * np.cos(angle)
        dy = l * np.sin(angle)
        action = np.array([dx, dy])
        return action

    def forward(self, state):
        """Step forward."""
        return NotImplementedError("Stub.")

    def __call__(self, state):
        return self.forward(state)

    def reset(self):
        return NotImplementedError("Stub.")

    def update(self, *args):
        return NotImplementedError("Stub.")


class Uniform2d(Agent2d):
    """Uniform (maxent) search"""
    def __init__(self, min_l=0.1, max_l=1000, detection_radius=1):
        super().__init__()
        self.min_l = min_l
        self.max_l = max_l
        self.detection_radius = detection_radius
        self.reset()

    def _l(self, state):
        """Sample length"""
        l = self.np_random.uniform(self.min_l, self.max_l)
        return l

    def _delta(self, state):
        """Set step size"""

        # r * 4 steps for each l (magic number)
        div = int(self.l / self.detection_radius) * 4
        if div > 1:
            delta = np.linspace(0, self.l, num=div)[1]
        else:
            delta = self.l

        return delta

    def forward(self, state):
        """Step forward."""
        # Go? or Turn?
        if self.l > self.step:
            self.num_step += 1
            self.step += self.delta
        else:
            self.num_turn += 1
            self.num_step = 0
            self.l = self._l(state)
            self.angle = self._angle(state)
            self.delta = self._delta(state)
            self.step = self.delta

        # Step
        action = self._convert(self.angle, self.delta)

        # Log
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_step"].append(deepcopy(self.delta))
        self.history["agent_num_step"].append(deepcopy(self.num_step))
        self.history["agent_action"].append(deepcopy(action))

        return action

    def reset(self):
        """Reset all counters, turns, and steps"""
        self.num_turn = 0
        self.l = 0
        self.angle = 0
        self.num_step = 0
        self.delta = 0
        self.step = 0
        self.history = defaultdict(list)


class Levy2d(Agent2d):
    """Levy search"""
    def __init__(self, min_l=0.1, exponent=2, detection_radius=1):
        super().__init__()
        self.min_l = min_l
        self.exponent = exponent
        self.detection_radius = detection_radius
        self.reset()

    def _l(self, state):
        """Sample length"""
        i = 0
        while True and i < 10000:
            i += 1
            l = np.power(self.np_random.uniform(), (-1 / self.exponent))
            if l > self.min_l:
                return l

    def _delta(self, state):
        """Set step size"""
        # r * 4 steps for each l (magic number)
        div = int(self.l / self.detection_radius) * 4
        if div > 1:
            delta = np.linspace(0, self.l, num=div)[1]
        else:
            delta = self.l

        return delta

    def forward(self, state):
        """Step the agent forward"""
        # Go? or Turn?
        if self.l > self.step:
            self.num_step += 1
            self.step += self.delta
        else:
            self.num_turn += 1
            self.num_step = 0
            self.l = self._l(state)
            self.angle = self._angle(state)
            self.delta = self._delta(state)
            self.step = self.delta

        # Step
        action = self._convert(self.angle, self.delta)

        # Log
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_step"].append(deepcopy(self.delta))
        self.history["agent_num_step"].append(deepcopy(self.num_step))
        self.history["agent_action"].append(deepcopy(action))

        return action

    def reset(self):
        """Reset all counters, turns, and steps"""
        self.num_turn = 0
        self.l = 0
        self.angle = 0
        self.num_step = 0
        self.delta = 0
        self.step = 0
        self.history = defaultdict(list)


class TruncatedLevy2d(Agent2d):
    """Truncated Levy search"""
    def __init__(self, min_l=0.1, max_l=100, exponent=2, detection_radius=1):
        super().__init__()
        self.min_l = min_l
        self.max_l = max_l
        self.exponent = exponent
        self.detection_radius = detection_radius
        self.reset()

    def _l(self, state):
        """Sample length"""
        i = 0
        while True and i < 10000:
            i += 1
            l = np.power(self.np_random.uniform(), (-1 / self.exponent))
            if (l > self.min_l) and (l <= self.max_l):
                return l

    def _delta(self, state):
        """Set step size"""
        # delta - r * 4 steps for each l
        div = int(self.l / self.detection_radius) * 4
        if div > 1:
            delta = np.linspace(0, self.l, num=div)[1]
        else:
            delta = self.l
        return delta

    def forward(self, state):
        """Step forward."""
        # Go? Or Turn?
        if self.l > self.step:
            self.num_step += 1
            self.step += self.delta
        else:
            self.num_turn += 1
            self.num_step = 0
            self.l = self._l(state)
            self.angle = self._angle(state)
            self.delta = self._delta(state)
            self.step = self.delta

        # Step
        action = self._convert(self.angle, self.delta)

        # Log
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_step"].append(deepcopy(self.delta))
        self.history["agent_num_step"].append(deepcopy(self.num_step))
        self.history["agent_action"].append(deepcopy(action))

        return action

    def reset(self):
        """Reset all counters, turns, and steps"""
        self.num_turn = 0
        self.l = 0
        self.angle = 0
        self.num_step = 0
        self.delta = 0
        self.step = 0
        self.history = defaultdict(list)


class Diffusion2d(Agent2d):
    """Diffusion search"""
    def __init__(self, min_l=0.1, scale=2, detection_radius=1):
        super().__init__()
        self.min_l = min_l
        self.scale = scale
        self.detection_radius = detection_radius
        self.reset()

    def _l(self, state):
        """Sample length"""
        i = 0
        while True and i < 10000:
            i += 1
            l = self.np_random.exponential(self.scale)
            if l > self.min_l:
                return l

    def _delta(self, state):
        """Set step size"""
        # delta - r * 4 steps for each l
        div = int(self.l / self.detection_radius) * 4
        if div > 1:
            delta = np.linspace(0, self.l, num=div)[1]
        else:
            delta = self.l

        return delta

    def forward(self, state):
        """Step forward."""
        # Go? Or turn?
        if self.l > self.step:
            self.step += self.delta
            self.num_step += 1
        else:
            self.num_turn += 1
            self.num_step = 0
            self.l = self._l(state)
            self.angle = self._angle(state)
            self.delta = self._delta(state)
            self.step = self.delta

        # Step
        action = self._convert(self.angle, self.delta)

        # Log
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_step"].append(deepcopy(self.delta))
        self.history["agent_num_step"].append(deepcopy(self.num_step))
        self.history["agent_action"].append(deepcopy(action))

        return action

    def reset(self):
        """Reset all counters, turns, and steps"""
        self.num_turn = 0
        self.l = 0
        self.angle = 0
        self.num_step = 0
        self.delta = 0
        self.step = 0
        self.history = defaultdict(list)
