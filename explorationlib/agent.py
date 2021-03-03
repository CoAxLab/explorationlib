import numpy as np

from copy import deepcopy
from scipy.stats import powerlaw
from collections import defaultdict
from sklearn.neighbors import KDTree


class Agent2d:
    """API stub."""
    def __init__(self):
        self.seed()
        self.total_distance = 0.0
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
        self.total_distance += self.step
        action = self.angle

        # Log
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_total_l"].append(deepcopy(self.total_distance))
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
        self.total_distance = 0.0
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
        self.total_distance += self.step
        action = self.angle

        # Log
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_total_l"].append(deepcopy(self.total_distance))
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
        self.total_distance = 0.0
        self.history = defaultdict(list)


class DiffusionMemoryCardinal(DiffusionDiscrete):
    """Diffusion search, with a short-term memory"""
    def __init__(self, min_length=1, scale=0.1, **kd_kwargs):
        super().__init__(num_actions=4, min_length=min_length, scale=scale)
        self.possible_steps = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.memory = {}
        self.kd_kwargs = kd_kwargs

    def update(self, state, reward, info):
        # A good memory has a reward
        if np.nonzero(reward):
            self.memory[state] = reward
            self._positions = sorted(list(self.memory.keys()))
            self._kd = KDTree(np.vstack(self._positions), **self.kd_kwargs)
        # Only keep good memories
        else:
            del self.memory[state]

    def forward(self, state):
        """Step forward."""
        # Go? Or turn?
        if self.l > self.step:
            self.step += self.step_size
            self.num_step += 1
        # Is memory empty?
        elif len(self.memory) > 0:
            # Find the best (closest memory)
            state = np.atleast_2d(np.asarray(state))
            _, ind = self._kd.query(state, k=1)
            best = self._positions[ind]

            # Distance to best from all possible_steps?
            candidates = [
                np.linalg.norm(
                    np.asarray(state) + np.asarray(s), np.asarray(best))
                for s in self.possible_steps
            ]

            # Pick the nearest
            ind = np.argmin(candidates)

            # Set direction
            self.angle = self.possible_steps[ind]
            self.l = self.step_size
        else:
            self.num_turn += 1
            self.num_step = 0
            self.l = self._l(state)
            self.angle = self.possible_steps[self._angle(state)]

        # Step
        action = state + self.angle
        self.step = self.step_size
        self.total_distance += self.step

        # Log
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_total_l"].append(deepcopy(self.total_distance))
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
        self.total_distance = 0.0
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
        self.total_distance += self.step

        # Log
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_total_l"].append(deepcopy(self.total_distance))
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
        self.total_distance = 0.0
        self.history = defaultdict(list)


class AccumulatorGradientDiscrete(Agent2d):
    """Incremental search, using evidence accumulation to 
    estimate the gradient, which in turn effects turn probability. 
    
    AKA information accumulating E. Coli.

    Note: 
    ----
    Positive gradients set the turn prob. to p_pos.
    """
    def __init__(self,
                 num_actions=4,
                 min_length=1,
                 max_steps=100,
                 drift_rate=1.0,
                 accumulate_sigma=1.0,
                 threshold=10.0,
                 p_neg=0.8,
                 p_pos=0.2):
        super().__init__()
        self.max_steps = int(max_steps)
        self.drift_rate = float(drift_rate)
        self.accumulate_sigma = float(accumulate_sigma)
        self.threshold = float(threshold)

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
        return self.step_size

    def _w(self, n):
        """Wiener process"""
        # Init/reset
        if n == 0: self.w = 0.0
        # Draw
        xi = self.np_random.normal(0.0, self.accumulate_sigma)
        # Step process
        self.w += (xi / np.sqrt(n + 1))
        return self.w

    def _accumulate_grad(self, obs):
        """Weight the evidence for an observation"""
        # If no info, treat as p_neg?
        if np.isclose(obs, 0.0):
            return obs, -1.0

        # Consider things....
        evidence = self.drift_rate * (obs - self.last_obs)
        for n in range(self.max_steps):
            evidence += self.drift_rate * self._w(n)
            if np.abs(evidence) > self.threshold:
                print("!")
                break

        return evidence, np.sign(evidence)

    def forward(self, state):
        """Step forward."""
        # Parse
        _, obs = state

        # Deliberate by accumulation
        evidence, grad = self._accumulate_grad(obs)
        # print(obs, obs - self.last_obs, evidence, grad)

        # Update the past
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
            if p > xi:
                self.num_turn += 1
                self.num_step = 0
                self.l = self._l(state)
                self.angle = self._angle(state)
                self.step = self.step_size

        # Step
        action = self.angle
        self.total_distance += self.step

        # Log
        self.history["agent_obs"].append(deepcopy(obs))
        self.history["agent_delta"].append(deepcopy(obs - self.last_obs))
        self.history["agent_evidence"].append(deepcopy(evidence))
        self.history["agent_grad"].append(deepcopy(grad))
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_total_l"].append(deepcopy(self.total_distance))
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
        self.total_distance = 0.0
        self.history = defaultdict(list)


class AccumulatorGradientCardinal(Agent2d):
    """Incremental search, using evidence accumulation to 
    estimate the gradient, which in turn effects turn probability. 
    
    AKA information accumulating E. Coli.

    Note: 
    ----
    Positive gradients set the turn prob. to p_pos.
    """
    def __init__(self,
                 min_length=1,
                 max_steps=100,
                 drift_rate=1.0,
                 accumulate_sigma=1.0,
                 threshold=10.0,
                 p_neg=0.8,
                 p_pos=0.2):
        super().__init__()
        self.possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        self.max_steps = int(max_steps)
        self.drift_rate = float(drift_rate)
        self.accumulate_sigma = float(accumulate_sigma)
        self.threshold = float(threshold)

        self.min_length = int(min_length)

        self.p_pos = float(p_pos)
        self.p_neg = float(p_neg)
        self.last_obs = 0.0
        self.evidence = 0.0
        self.step_size = 1
        self.reset()

    def _angle(self, state):
        i = int(self.np_random.randint(0, len(self.possible_actions)))
        return self.possible_actions[i]

    def _l(self, state):
        """Sample length"""
        return self.step_size

    def _w(self, n):
        """Wiener process"""
        # Init/reset
        if n == 0: self.w = 0.0
        # Draw
        xi = self.np_random.normal(0.0, self.accumulate_sigma)
        # Step process
        self.w += (xi / np.sqrt(n + 1))
        return self.w

    def _accumulate_grad(self, obs, evidence):
        """Weight the evidence for an observation"""
        # If no info, treat as p_neg?
        if np.isclose(obs, 0.0):
            return obs, -1.0, True

        # Consider things....
        stop = False
        delta = self.drift_rate * (obs - self.last_obs)
        for n in range(self.max_steps):
            evidence = delta * self.drift_rate * self._w(n)
            if np.abs(evidence) > self.threshold:
                stop = True
                break

        return evidence, np.sign(evidence), stop

    def forward(self, state):
        """Step forward."""
        # Parse
        _, obs = state

        # Deliberate by accumulation
        self.evidence, grad, stop = self._accumulate_grad(obs, self.evidence)

        # Default is no-op
        action = (0, 0)
        self.l = 0.0
        self.step = 0.0

        # Move only when accum has stopped:
        if stop:
            # Reset
            self.evidence = 0.0

            # Grad weighted coin toss:
            #
            # Pick p controller
            p = self.p_neg
            if grad > 0:
                p = self.p_pos
            # Keep going?
            if self.l > self.step:
                self.step += self.step_size
                self.num_step += 1
            # Turn?
            else:
                if p > self.np_random.rand():
                    self.num_turn += 1
                    self.num_step = 0
                    self.l = self._l(state)
                    self.angle = self._angle(state)
                    self.step = self.step_size
            # Update the past
            self.last_obs = deepcopy(obs)
            # Set new direction
            action = self.angle

        self.total_distance += self.step

        # Log
        self.history["agent_obs"].append(deepcopy(obs))
        self.history["agent_delta"].append(deepcopy(obs - self.last_obs))
        self.history["agent_evidence"].append(deepcopy(self.evidence))
        self.history["agent_stop"].append(deepcopy(stop))
        self.history["agent_grad"].append(deepcopy(grad))
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_total_l"].append(deepcopy(self.total_distance))
        self.history["agent_step"].append(deepcopy(self.step_size))
        self.history["agent_num_step"].append(deepcopy(self.num_step))
        self.history["agent_action"].append(deepcopy(action))

        # print(action, self.evidence, self.threshold)
        return action

    def reset(self):
        """Reset all counters, turns, and steps"""

        # Safe intial values
        self.l = self._l(np.zeros(2))
        self.angle = self._angle(np.zeros(2))

        # Clean
        self.num_turn = 0
        self.num_step = 0
        self.last_obs = 0.0
        self.step = 0
        self.evidence = 0.0
        self.total_distance = 0.0
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
        self.total_distance += self.step

        # Log
        self.history["agent_grad"].append(deepcopy(grad))
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_total_l"].append(deepcopy(self.total_distance))
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
        self.total_distance = 0.0
        self.history = defaultdict(list)


class GradientDiffusionCardinal(Agent2d):
    """Diffusion search, but the sense/obs gradient 
    effects turn probability. 
    
    Note: 
    ----
    Positive gradients set the turn prob. to p_pos.
    """
    def __init__(self, min_length=1, scale=2, p_neg=0.8, p_pos=0.2):
        super().__init__()
        self.possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        self.scale = float(scale)
        self.min_length = int(min_length)

        self.p_pos = float(p_pos)
        self.p_neg = float(p_neg)
        self.last_obs = 0.0
        self.step_size = 1
        self.reset()

    def _angle(self, state):
        i = int(self.np_random.randint(0, len(self.possible_actions)))
        return self.possible_actions[i]

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

        # Grad weighted coin toss:
        #
        # Pick p controller
        p = self.p_neg
        if grad > 0:
            p = self.p_pos
        # Keep going?
        if self.l > self.step:
            self.step += self.step_size
            self.num_step += 1
        # Turn?
        else:
            xi = self.np_random.rand()
            if p > xi:
                self.num_turn += 1
                self.num_step = 0
                self.l = self._l(state)
                self.angle = self._angle(state)
                self.step = self.step_size

        # print(pos, obs, grad, p, self.angle)
        # Step
        action = self.angle
        self.total_distance += self.step

        # Log
        self.history["agent_grad"].append(deepcopy(grad))
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_total_l"].append(deepcopy(self.total_distance))
        self.history["agent_step"].append(deepcopy(self.step_size))
        self.history["agent_num_step"].append(deepcopy(self.num_step))
        self.history["agent_action"].append(deepcopy(action))

        return action

    def reset(self):
        """Reset all counters, turns, and steps"""

        # Safe intial values
        self.l = self._l(np.zeros(2))
        self.angle = self._angle(np.zeros(2))

        # Clean
        self.num_turn = 0
        self.num_step = 0
        self.last_obs = 0.0
        self.step = 0
        self.total_distance = 0.0
        self.history = defaultdict(list)


class Uniform2d(Agent2d):
    """Uniform (maxent) search"""
    def __init__(self,
                 min_length=0.1,
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
        self.total_distance += self.step

        # Log
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_total_l"].append(deepcopy(self.total_distance))
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
        self.total_distance = 0.0
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
        self.total_distance += self.step

        # Log
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_total_l"].append(deepcopy(self.total_distance))
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
        self.total_distance = 0.0
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
        self.total_distance += self.step

        # Log
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_total_l"].append(deepcopy(self.total_distance))
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
        self.total_distance = 0.0
        self.history = defaultdict(list)


class Diffusion2d(Agent2d):
    """Diffusion search"""
    def __init__(self,
                 min_length=0.1,
                 scale=0.1,
                 detection_radius=1,
                 step_size=0.1):
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
        self.total_distance += self.step

        # Log
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_total_l"].append(deepcopy(self.total_distance))
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
        self.total_distance = 0.0
        self.history = defaultdict(list)
