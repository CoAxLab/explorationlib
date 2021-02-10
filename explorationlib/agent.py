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
    def __init__(self,
                 min_length=0.1,
                 max_length=1000,
                 detection_radius=1,
                 step_size=0.01):
        super().__init__()
        self.step_size = step_size

        self.min_length = min_length
        self.max_length = max_length
        self.detection_radius = detection_radius
        self.reset()

    def _l(self, state):
        """Sample length"""
        l = self.np_random.uniform(self.min_length, self.max_length)
        return l

    def forward(self, state):
        """Step forward."""
        # Go? or Turn?
        if self.l > self.step:
            self.num_step += 1
            self.step += self.step_size
        else:
            self.num_turn += 1
            self.num_step = 0
            self.l = self._l(state)
            self.angle = self._angle(state)
            self.step = self.step_size

        # Step
        action = self._convert(self.angle, self.step_size)

        # Log
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_step"].append(deepcopy(self.step_size))
        self.history["agent_num_step"].append(deepcopy(self.num_step))
        self.history["agent_action"].append(deepcopy(action))

        return action

    def reset(self):
        """Reset all counters, turns, and steps"""
        self.num_turn = 0
        self.l = 0
        self.angle = 0
        self.num_step = 0
        self.step = 0
        self.history = defaultdict(list)


class Levy2d(Agent2d):
    """Levy search
    
    Citation
    --------
    Renorm scheme taken from:

    - Jansen, V. A. A., Mashanova, A. & Petrovskii, S. Comment on ‘Levy Walks Evolve Through Interaction Between Movement and Environmental Complexity’. Science 335, 918–918 (2012).
    """
    def __init__(self,
                 min_length=0.1,
                 exponent=2,
                 detection_radius=1,
                 step_size=0.01):
        super().__init__()
        self.step_size = step_size

        self.min_length = min_length
        self.exponent = exponent
        self.detection_radius = detection_radius
        self.renorm = (self.exponent - 1) / (self.min_length
                                             **(1 - self.exponent))

        self.reset()

    def _l(self, state):
        """Sample length"""
        i = 0
        while True and i < 10000:
            i += 1
            xi = self.np_random.rand()
            l = self.renorm * np.power(xi, (-1 / self.exponent))
            if l > self.min_length:
                return l

    def forward(self, state):
        """Step the agent forward"""
        # Go? or Turn?
        if self.l > self.step:
            self.num_step += 1
            self.step += self.step_size
        else:
            self.num_turn += 1
            self.num_step = 0
            self.l = self._l(state)
            self.angle = self._angle(state)
            self.step = self.step_size

        # Step
        action = self._convert(self.angle, self.step_size)

        # Log
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_step"].append(deepcopy(self.step_size))
        self.history["agent_num_step"].append(deepcopy(self.num_step))
        self.history["agent_action"].append(deepcopy(action))

        return action

    def reset(self):
        """Reset all counters, turns, and steps"""
        self.num_turn = 0
        self.l = 0
        self.angle = 0
        self.num_step = 0
        self.step = 0
        self.history = defaultdict(list)


class TruncatedLevy2d(Agent2d):
    """Truncated Levy search
    
    Citation
    --------
    Renorm scheme taken from:
    Jansen, V. A. A., Mashanova, A. & Petrovskii, S. Comment on ‘Levy Walks Evolve Through Interaction Between Movement and Environmental Complexity’. Science 335, 918–918 (2012).
    """
    def __init__(self,
                 min_length=0.1,
                 max_length=100,
                 exponent=2,
                 detection_radius=1,
                 step_size=0.01):
        super().__init__()
        self.step_size = step_size

        self.detection_radius = detection_radius
        self.exponent = exponent

        self.min_length = min_length
        self.max_length = max_length

        min_norm = self.min_length**(1 - self.exponent)
        max_norm = self.max_length**(1 - self.exponent)
        self.renorm = (self.exponent - 1) / (min_norm - max_norm)

        self.reset()

    def _l(self, state):
        """Sample length"""
        i = 0
        while True and i < 10000:
            i += 1
            l = np.power(self.np_random.uniform(), (-1 / self.exponent))
            if (l > self.min_length) and (l <= self.max_length):
                return l

    def forward(self, state):
        """Step forward."""
        # Go? Or Turn?
        if self.l > self.step:
            self.num_step += 1
            self.step += self.step_size
        else:
            self.num_turn += 1
            self.num_step = 0
            self.l = self._l(state)
            self.angle = self._angle(state)
            self.step = self.step_size

        # Step
        action = self._convert(self.angle, self.step_size)

        # Log
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_step"].append(deepcopy(self.step_size))
        self.history["agent_num_step"].append(deepcopy(self.num_step))
        self.history["agent_action"].append(deepcopy(action))

        return action

    def reset(self):
        """Reset all counters, turns, and steps"""
        self.num_turn = 0
        self.l = 0
        self.angle = 0
        self.num_step = 0
        self.step = 0
        self.history = defaultdict(list)


class Diffusion2d(Agent2d):
    """Diffusion search"""
    def __init__(self,
                 min_length=0.1,
                 scale=2,
                 detection_radius=1,
                 step_size=0.01):
        super().__init__()
        self.step_size = step_size

        self.min_length = min_length
        self.scale = scale
        self.detection_radius = detection_radius

        self.reset()

    def _l(self, state):
        """Sample length"""
        i = 0
        while True and i < 10000:
            i += 1
            l = self.np_random.exponential(self.scale)
            if l > self.min_length:
                return l

    def forward(self, state):
        """Step forward."""
        # Go? Or turn?
        if self.l > self.step:
            self.step += self.step_size
            self.num_step += 1
        else:
            self.num_turn += 1
            self.num_step = 0
            self.l = self._l(state)
            self.angle = self._angle(state)
            self.step = self.step_size

        # Step
        action = self._convert(self.angle, self.step_size)

        # Log
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_step"].append(deepcopy(self.step_size))
        self.history["agent_num_step"].append(deepcopy(self.num_step))
        self.history["agent_action"].append(deepcopy(action))

        return action

    def reset(self):
        """Reset all counters, turns, and steps"""
        self.num_turn = 0
        self.l = 0
        self.angle = 0
        self.num_step = 0
        self.step = 0
        self.history = defaultdict(list)
