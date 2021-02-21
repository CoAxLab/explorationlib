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


class UniformDiscrete(Agent2d):
    def __init__(self, num_actions=4, min_length=1, max_length=4):
        super().__init__()
        self.num_actions = num_actions
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.step_size = 1

    def _angle(self, state):
        return int(self.np_random.randint(0, self.num_actions))

    def _l(self, state):
        """Sample length"""
        l = int(self.np_random.uniform(self.min_length, self.max_length))
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
        action = self.angle

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
        self.angle = None
        self.num_step = 0
        self.step = 0
        self.history = defaultdict(list)


class DiffusionDiscrete(Agent2d):
    """Diffusion search"""
    def __init__(self, num_actions=4, min_length=1, scale=2):
        super().__init__()

        self.num_actions = int(num_actions)
        self.min_length = int(min_length)
        self.scale = float(scale)

        self.step_size = 1
        self.reset()

    def _angle(self, state):
        return int(self.np_random.randint(0, self.num_actions))

    def _l(self, state):
        """Sample length"""
        i = 0
        while True and i < 10000:
            i += 1
            l = self.np_random.exponential(self.scale)
            l = int(l)
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
        action = self.angle

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
        self.angle = None
        self.num_step = 0
        self.step = 0
        self.history = defaultdict(list)


class TruncatedLevyDiscrete(Agent2d):
    """Truncated Levy search, for discrete worlds.
    
    Citation
    --------
    Renorm scheme taken from:
    Jansen, V. A. A., Mashanova, A. & Petrovskii, S. Comment on ‘Levy Walks Evolve Through Interaction Between Movement and Environmental Complexity’. Science 335, 918–918 (2012).
    """
    def __init__(self, num_actions=4, min_length=1, max_length=10, exponent=2):
        super().__init__()

        self.exponent = float(exponent)
        self.num_actions = int(num_actions)
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.step_size = 1

        min_norm = self.min_length**(1 - self.exponent)
        max_norm = self.max_length**(1 - self.exponent)
        self.renorm = (self.exponent - 1) / (min_norm - max_norm)

        self.reset()

    def _angle(self, state):
        return int(self.np_random.randint(0, self.num_actions))

    def _l(self, state):
        """Sample length"""
        i = 0
        while True and i < 10000:
            i += 1
            l = np.power(self.np_random.uniform(), (-1 / self.exponent))
            l = int(l)
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
        action = self.angle

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
        self.angle = None
        self.num_step = None
        self.step = 0
        self.history = defaultdict(list)


class GradientDiffusionDiscrete(Agent2d):
    """Diffusion search, but the sense/obs gradient 
    effects turn probability. 
    
    Note: 
    ----
    Positive gradients set the turn prob. to p_pos.
    """
    def __init__(self,
                 num_actions=4,
                 min_length=1,
                 scale=2,
                 p_neg=0.8,
                 p_pos=0.2):
        super().__init__()

        self.scale = float(scale)
        self.num_actions = int(num_actions)
        self.min_length = int(min_length)

        self.p_pos = float(p_pos)
        self.p_neg = float(p_neg)
        self.last_obs = 0.0
        self.step_size = 1
        self.reset()

    def _angle(self, state):
        return int(self.np_random.randint(0, self.num_actions))

    def _l(self, state):
        """Sample length"""
        i = 0
        while True and i < 10000:
            i += 1
            l = self.np_random.exponential(self.scale)
            l = int(l)
            if l > self.min_length:
                return l

    def forward(self, state):
        """Step forward."""
        # Parse
        pos, obs = state

        # Est grad (crudely)
        grad = np.sign(obs - self.last_obs)
        self.last_obs = deepcopy(obs)

        # Grad weighted coinf
        p = self.p_neg
        if grad > 0:
            p = self.p_pos

        if self.l > self.step:
            self.step += self.step_size
            self.num_step += 1
        else:
            xi = self.np_random.rand()
            # print(grad, p, xi, p > xi)
            if p > xi:
                self.num_turn += 1
                self.num_step = 0
                self.l = self._l(state)
                self.angle = self._angle(state)
                self.step = self.step_size

        # print(pos, obs, grad, p, self.angle)
        # Step
        action = self.angle

        # Log
        self.history["agent_grad"].append(deepcopy(grad))
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_step"].append(deepcopy(self.step_size))
        self.history["agent_num_step"].append(deepcopy(self.num_step))
        self.history["agent_action"].append(deepcopy(action))

        return action

    def reset(self):
        """Reset all counters, turns, and steps"""

        # Safe intial values
        self.l = self.min_length
        self.angle = int(self.np_random.randint(0, self.num_actions))

        # Clean
        self.num_turn = 0
        self.num_step = 0
        self.last_obs = 0.0
        self.step = 0
        self.history = defaultdict(list)


class Uniform2d(Agent2d):
    """Uniform (maxent) search"""
    def __init__(self,
                 min_length=1,
                 max_length=10,
                 detection_radius=1,
                 step_size=0.1):
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
                 min_length=1,
                 exponent=2,
                 detection_radius=1,
                 step_size=0.1):
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
                 min_length=1,
                 max_length=10,
                 exponent=2,
                 detection_radius=1,
                 step_size=0.1):
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
                 min_length=1,
                 scale=2,
                 detection_radius=1,
                 step_size=0.):
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
