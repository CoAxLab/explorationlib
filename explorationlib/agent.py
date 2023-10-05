import numpy as np

from copy import deepcopy
from collections import defaultdict
from collections import OrderedDict

from scipy.special import softmax
from scipy.stats import powerlaw
from scipy.stats import entropy as scientropy

from sklearn.neighbors import KDTree

from explorationlib.memory import CountMemory
from explorationlib.memory import EntropyMemory
from explorationlib.memory import NoveltyMemory
from explorationlib.memory import DiscreteDistributionGrid


# -----------------------------------------------------------------------------
# RL - bandits
def R_update(state, R, critic, lr):
    """TD-ish update"""
    update = lr * (R - critic(state))
    critic.update(state, update)

    return critic


def Q_grid_update(state, action, R, next_state, critic, lr, gamma):
    Q = critic.get_value(state, action)
    max_Q = np.max(critic(next_state))
    update = lr * ((R + gamma * max_Q) - Q)
    critic.update(state, action, update)

    return critic


def E_update(state, E, critic, lr):
    """Bellman update"""
    update = lr * E
    critic.replace(state, update)

    return critic


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

    def update(self, state, action, R, next_state, info):
        return NotImplementedError("Stub.")


class CriticGrid(Agent2d):
    """Template for a Critic agent"""
    def __init__(self, default_value=0.5):
        super().__init__()
        self.possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.default_value = default_value
        self.model = OrderedDict()

    def __call__(self, state):
        return self.forward(state)

    def _init_model(self, state):
        self.model[state] = OrderedDict()
        for action in self.possible_actions:
            self.model[state][action] = self.default_value

    def forward(self, state):
        if state not in self.model:
            self._init_model(state)
        return list(self.model[state].values())

    def get_value(self, state, action):
        Qs = self.forward(state)
        idx = self.possible_actions.index(action)
        Q = Qs[idx]
        return Q

    def update(self, state, action, update):
        if state not in self.model:
            self._init_model(state)
        self.model[state][action] += update

    def replace(self, state, action, update):
        if state not in self.model:
            self._init_model(state)
        self.model[state][action] = update

    def state_dict(self):
        return self.model

    def load_state_dict(self, state_dict):
        self.model = state_dict

    def reset(self):
        self.model = OrderedDict()

    def seed(self, seed=None):
        """Init RandomState"""
        self.np_random = np.random.RandomState(seed)
        return [seed]


class ActorCriticGrid:
    """Actor-Critic agent, with Q learning."""
    def __init__(self, actor, critic, lr=0.1, gamma=0.5):
        self.seed()
        self.history = defaultdict(list)
        self.total_distance = 0

        self.actor = actor
        self.critic = critic
        self.lr = float(lr)
        self.gamma = float(gamma)

    def __call__(self, state):
        pos, _ = state
        pos = tuple(pos)

        return self.forward(pos)

    def update(self, state, action, R, next_state, info):
        # Parse
        pos, _ = state
        pos = tuple(pos)

        next_pos, _ = next_state
        next_pos = tuple(next_pos)

        self.critic = Q_grid_update(
            pos,
            action,
            R,
            next_pos,
            self.critic,
            self.lr,
            self.gamma,
        )

    def forward(self, state):
        action = self.actor(self.critic(state))

        # Log
        self.total_distance += 1
        self.history["agent_reward_value"].append(
            deepcopy(self.critic.get_value(state, action)))
        self.history["agent_total_l"].append(deepcopy(self.total_distance))
        self.history["agent_step"].append(deepcopy(1))
        self.history["agent_num_step"].append(deepcopy(self.total_distance))
        self.history["agent_action"].append(deepcopy(action))

        return action

    def reset(self):
        # Don't reset RL algs between experiments
        self.history = defaultdict(list)
        self.total_distance = 0

    def seed(self, seed=None):
        """Init RandomState"""
        self.np_random = np.random.RandomState(seed)
        return [seed]


class WSLSGrid:
    """A Win-stay lose-switch agent"""
    def __init__(self,
                 actor_E,
                 critic_E,
                 actor_R,
                 critic_R,
                 initial_bins,
                 lr=0.1,
                 gamma=1,
                 boredom=0.0):
        super().__init__()
        self.seed()
        self.history = defaultdict(list)
        self.total_distance = 0

        # Agents init
        self.actor_R = actor_R
        self.critic_R = critic_R
        self.actor_E = actor_E
        self.critic_E = critic_E

        # HP init
        self.boredom = float(boredom)
        self.lr = float(lr)
        self.gamma = float(gamma)

        # Memory init
        self.initial_bins = initial_bins
        self.bin_index = list(range(1, len(self.initial_bins) + 1))
        self.grid = DiscreteDistributionGrid(self.bin_index)

        # Values init
        self.E = self.critic_E.default_value
        self.R = self.critic_R.default_value

    def __call__(self, state):
        pos, _ = state
        pos = tuple(pos)

        return self.forward(pos)

    def update(self, state, action, R, next_state, info):
        # Parse
        pos, _ = state
        pos = tuple(pos)

        next_pos, obs = next_state
        next_pos = tuple(next_pos)

        # Simplify the obs
        obs = np.digitize(obs, self.initial_bins)

        # Add to grid memory
        p_old = deepcopy(self.grid.probs(pos))
        self.grid(pos, obs)
        p_new = deepcopy(self.grid.probs(pos))

        # Info gain, by KL
        E = scientropy(p_old, qk=p_new, base=2)

        # Update instant Values
        self.E = E
        self.R = R

        # Update total Value
        self.critic_R = Q_grid_update(
            pos,
            action,
            R,
            next_pos,
            self.critic_R,
            self.lr,
            self.gamma,
        )
        self.critic_E = Q_grid_update(
            pos,
            action,
            E,
            next_pos,
            self.critic_E,
            self.lr,  # Update to Qmax proper
            self.gamma,
        )

    def forward(self, state):
        # Meta-choice
        if (self.E - self.boredom) > self.R:
            critic = self.critic_E
            actor = self.actor_E
        else:
            critic = self.critic_R
            actor = self.actor_R

        # Choose action
        action = actor(critic(state))

        # Log
        self.total_distance += 1
        self.history["agent_reward_value"].append(
            deepcopy(self.critic_R.get_value(state, action)))
        self.history["agent_info_value"].append(
            deepcopy(self.critic_E.get_value(state, action)))
        self.history["agent_total_l"].append(deepcopy(self.total_distance))
        self.history["agent_step"].append(deepcopy(1))
        self.history["agent_num_step"].append(deepcopy(self.total_distance))
        self.history["agent_action"].append(deepcopy(action))

        return action

    def reset(self):
        # Don't reset RL algs between experiments
        self.history = defaultdict(list)
        self.total_distance = 0

    def seed(self, seed=None):
        """Init RandomState"""
        self.np_random = np.random.RandomState(seed)
        return [seed]


class DeterministicWSLSGrid(WSLSGrid):
    def __init__(self, lr=0.1, gamma=0.1, boredom=0.001):
        # Action space (NSEW)
        possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        num_action = len(possible_actions)
        initial_bins = np.linspace(0, 1, 10)

        # Create agents
        critic_R = CriticGrid(default_value=0.5)
        critic_E = CriticGrid(default_value=np.log(num_action))
        actor_R = SoftmaxActor(num_actions=4,
                               actions=possible_actions,
                               beta=50)
        actor_E = SoftmaxActor(num_actions=4,
                               actions=possible_actions,
                               beta=50)
        # !
        super().__init__(actor_E,
                         critic_E,
                         actor_R,
                         critic_R,
                         initial_bins,
                         lr=lr,
                         gamma=gamma,
                         boredom=boredom)


class BanditAgent:
    def __init__(self):
        self.seed()
        self.history = defaultdict(list)

    def seed(self, seed=None):
        """Init RandomState"""
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        raise NotImplementedError("Mandatory to implement...")


class BanditActorCritic(BanditAgent):
    """Actor-Critic agent"""
    def __init__(self, actor, critic, lr_reward=0.1):
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.lr_reward = float(lr_reward)

    def __call__(self, state):
        return self.forward(state)

    def update(self, state, action, R, next_state, info):
        self.critic_R = R_update(action, R, self.critic, self.lr_reward)

    def forward(self, state):
        action = self.actor(list(self.critic.model.values()))
        return action

    def reset(self):
        self.actor.reset()
        self.critic.reset()


class Critic(BanditAgent):
    """Template for a Critic agent"""
    def __init__(self, num_inputs, default_value=0.0):
        super().__init__()

        self.num_inputs = num_inputs
        self.default_value = default_value

        self.model = OrderedDict()
        for n in range(self.num_inputs):
            self.model[n] = self.default_value

    def __call__(self, state):
        return self.forward(state)

    def forward(self, state):
        return self.model[state]

    def update(self, state, update):
        self.model[state] += update

    def replace(self, state, update):
        self.model[state] = update

    def state_dict(self):
        return self.model

    def load_state_dict(self, state_dict):
        self.model = state_dict

    def reset(self):
        self.model = OrderedDict()
        for n in range(self.num_inputs):
            self.model[n] = self.default_value


class CriticUCB(Critic):
    """Critic with a UCB bonus"""
    def __init__(self, num_inputs, default_value=0.0, bonus_weight=1.0):
        # Init base
        super().__init__(num_inputs, default_value)
        self.bonus_weight = float(bonus_weight)

        # Init UCB
        self.reset()

    def forward(self, state):
        # UCB bonus
        bonus = ((2 * np.log(self.n + 1)) / self.memory(state))**(0.5)
        bonus *= self.bonus_weight  # Scale

        # Value est.
        value = self.model[state] + bonus

        # Step global count
        self.n += 1

        # !
        # print(state, self.model[state], bonus)
        return value

    def reset(self):
        self.memory = CountMemory()
        self.n = 0


class CriticNovelty(Critic):
    """Critic with a novelty bonus"""
    def __init__(self,
                 num_inputs,
                 default_value=0.0,
                 novelty_bonus=1.0,
                 bonus_weight=1.0):
        # Init base
        super().__init__(num_inputs, default_value)

        # Init novelty (value of 1)
        self.novelty_bonus = float(novelty_bonus)
        self.bonus_weight = float(bonus_weight)
        self.reset()

    def forward(self, state):
        # Novelty
        bonus = self.memory(state)
        bonus *= self.bonus_weight  # Scale
        value = self.model[state] + bonus

        return value

    def reset(self):
        self.memory = NoveltyMemory(bonus=self.novelty_bonus)


class RandomActor(BanditAgent):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.actions = list(range(self.num_actions))

    def __call__(self, values):
        return self.forward(values)

    def forward(self, values):
        """Values are a dummy var. Pick at random"""
        action = self.np_random.choice(self.actions)
        return action

    def reset(self):
        pass


class BoundedRandomActor(BanditAgent):
    def __init__(self, num_actions, bound=100):
        super().__init__()
        self.num_actions = num_actions
        self.bound = int(bound)
        self.actions = list(range(self.num_actions))

    def __call__(self, values):
        return self.forward(values)

    def forward(self, values):
        """Values are a dummy var. Pick at random"""
        self.action_count += 1
        if self.action_count < self.bound:
            action = self.np_random.choice(self.actions)
        else:
            action = np.argmax(values)

        return action

    def reset(self):
        self.action_count = 0


class SequentialActor(BanditAgent):
    """Choose actions in sequence; ignore value"""
    def __init__(self, num_actions, initial_action=0):
        super().__init__()

        self.num_actions = int(num_actions)
        self.initial_action = int(initial_action)
        self.action_count = self.initial_action

    def __call__(self, values):
        return self.forward(values)

    def forward(self, values):
        self.action_count += 1
        action = self.action_count % self.num_actions

        return action

    def reset(self):
        self.action_count = self.initial_action


class BoundedSequentialActor(BanditAgent):
    """Choose actions in sequence; ignore value"""
    def __init__(self, num_actions, bound=100, initial_action=0):
        super().__init__()

        self.num_actions = int(num_actions)
        self.initial_action = int(initial_action)
        self.action_count = self.initial_action
        self.bound = int(bound)

    def __call__(self, values):
        return self.forward(values)

    def forward(self, values):
        self.action_count += 1

        if self.action_count < self.bound:
            action = self.action_count % self.num_actions
        else:
            action = np.argmax(values)

        return action

    def reset(self):
        self.action_count = self.initial_action


class DeterministicActor(BanditAgent):
    """Choose the best, with heuristics for ties"""
    def __init__(self, num_actions, tie_break='next', boredom=0.0):
        super().__init__()

        self.num_actions = num_actions
        self.actions = list(range(self.num_actions))
        self.tie_break = tie_break
        self.boredom = boredom
        self.action_count = 0
        self.tied = False

    def _is_tied(self, values):
        # One element can't be a tie
        if len(values) < 1:
            return False

        # Apply the threshold, rectifying values less than 0
        t_values = [max(0, v - self.boredom) for v in values]

        # Check for any difference, if there's a difference then
        # there can be no tie.
        tied = True  # Assume tie
        v0 = t_values[0]
        for v in t_values[1:]:
            if np.isclose(v0, v):
                continue
            else:
                tied = False
        return tied

    def __call__(self, values):
        return self.forward(values)

    def forward(self, values):
        # Pick the best as the base case, ....
        action = np.argmax(values)

        # then check for ties.
        #
        # Using the first element is argmax's tie breaking strategy
        if self.tie_break == 'first':
            pass
        # Round robin through the options for each new tie.
        elif self.tie_break == 'next':
            self.tied = self._is_tied(values)
            if self.tied:
                self.action_count += 1
                action = self.action_count % self.num_actions
        else:
            raise ValueError("tie_break must be 'first' or 'next'")

        return action

    def reset(self):
        self.action_count = 0
        self.tied = False


class SoftmaxActor(BanditAgent):
    """Softmax actions"""
    def __init__(self, num_actions, actions=None, beta=1.0):
        super().__init__()

        self.beta = float(beta)
        self.num_actions = num_actions
        if actions is None:
            self.actions = list(range(self.num_actions))
            self.action_idx = deepcopy(self.actions)
        else:
            self.actions = actions
            self.action_idx = list(range(self.num_actions))

    def __call__(self, values):
        return self.forward(values)

    def forward(self, values):
        values = np.asarray(values)
        probs = softmax(values * self.beta)
        idx = self.np_random.choice(self.action_idx, p=probs)
        return self.actions[idx]

    def reset(self):
        pass


class EpsilonActor(BanditAgent):
    def __init__(self, num_actions, epsilon=0.1, decay_tau=0.001):
        super().__init__()

        self.epsilon = epsilon
        self.decay_tau = decay_tau
        self.num_actions = num_actions

    def __call__(self, values):
        return self.forward(values)

    def decay_epsilon(self):
        self.epsilon -= (self.decay_tau * self.epsilon)

    def forward(self, values):
        # If values are zero, be random.
        if np.isclose(np.sum(values), 0):
            action = self.np_random.randint(0, self.num_actions, size=1)[0]
            return action

        # Otherwise, do Ep greedy
        if self.np_random.rand() < self.epsilon:
            action = self.np_random.randint(0, self.num_actions, size=1)[0]
        else:
            action = np.argmax(values)

        return action

    def reset(self):
        pass


# -----------------------------------------------------------------------------
# Field - random, grad, simple memory
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


# TODO - Grid form please...
# class DiffusionMemoryGrid(DiffusionDiscrete):
#     """Diffusion search, with a short-term memory"""
#     def __init__(self, min_length=1, scale=0.1, **kd_kwargs):
#         super().__init__(num_actions=4, min_length=min_length, scale=scale)
#         self.possible_steps = [(0, 1), (0, -1), (1, 0), (-1, 0)]
#         self.memory = {}
#         self.kd_kwargs = kd_kwargs

#     def update(self, state, reward, info):
#         # A good memory has a reward
#         if np.nonzero(reward):
#             self.memory[state] = reward
#             self._positions = sorted(list(self.memory.keys()))
#             self._kd = KDTree(np.vstack(self._positions), **self.kd_kwargs)
#         # Only keep good memories
#         else:
#             del self.memory[state]

#     def forward(self, state):
#         """Step forward."""
#         # Go? Or turn?
#         if self.l > self.step:
#             self.step += self.step_size
#             self.num_step += 1
#         # Is memory empty?
#         elif len(self.memory) > 0:
#             # Find the best (closest memory)
#             state = np.atleast_2d(np.asarray(state))
#             _, ind = self._kd.query(state, k=1)
#             best = self._positions[ind]

#             # Distance to best from all possible_steps?
#             candidates = [
#                 np.linalg.norm(
#                     np.asarray(state) + np.asarray(s), np.asarray(best))
#                 for s in self.possible_steps
#             ]

#             # Pick the nearest
#             ind = np.argmin(candidates)

#             # Set direction
#             self.angle = self.possible_steps[ind]
#             self.l = self.step_size
#         else:
#             self.num_turn += 1
#             self.num_step = 0
#             self.l = self._l(state)
#             self.angle = self.possible_steps[self._angle(state)]

#         # Step
#         action = state + self.angle
#         self.step = self.step_size
#         self.total_distance += self.step

#         # Log
#         self.history["agent_num_turn"].append(deepcopy(self.num_turn))
#         self.history["agent_angle"].append(deepcopy(self.angle))
#         self.history["agent_l"].append(deepcopy(self.l))
#         self.history["agent_total_l"].append(deepcopy(self.total_distance))
#         self.history["agent_step"].append(deepcopy(self.step_size))
#         self.history["agent_num_step"].append(deepcopy(self.num_step))
#         self.history["agent_action"].append(deepcopy(action))

#         return action

#     def reset(self):
#         """Reset all counters, turns, and steps"""
#         self.num_turn = 0
#         self.l = 0
#         self.angle = None
#         self.num_step = 0
#         self.step = 0
#         self.total_distance = 0.0
#         self.history = defaultdict(list)


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
                 max_steps=1,
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


class AccumulatorGradientGrid(Agent2d):
    """Incremental search, using evidence accumulation to 
    estimate the gradient, which in turn effects turn probability. 
    
    AKA accumulating gradient E. Coli.

    Note: 
    ----
    Positive gradients set the turn prob. to p_pos.
    """
    def __init__(self,
                 min_length=1,
                 max_steps=1,
                 drift_rate=1.0,
                 accumulate_sigma=1.0,
                 threshold=10.0,
                 p_neg=0.8,
                 p_pos=0.2,
                 step_size=1):
        super().__init__()
        self.possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.step_size = int(step_size)
        if self.step_size < 1:
            raise ValueError("step musst be >= 1")

        self.min_length = int(min_length)
        if self.min_length < 1:
            raise ValueError("min_length must be >= 1")
        self.max_steps = int(max_steps)

        self.drift_rate = float(drift_rate)
        self.accumulate_sigma = float(accumulate_sigma)
        self.threshold = float(threshold)

        self.p_pos = float(p_pos)
        self.p_neg = float(p_neg)
        self.last_obs = 0.0
        self.evidence = 0.0
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


class AccumulatorInfoGrid(Agent2d):
    """Incremental search, using evidence accumulation to 
    estimate the information hits, which in turn effects 
    turn probability. 
    
    AKA signal detection entropy in E. Coli.

    Note: 
    ----
    Positive info biases turn prob 
    """
    def __init__(self,
                 min_length=1,
                 max_steps=100,
                 drift_rate=1.0,
                 accumulate_sigma=1.0,
                 threshold=10.0,
                 p_pos=0.8,
                 p_neg=0.8,
                 step_size=1):
        super().__init__()
        # Action init
        self.possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.step_size = int(step_size)
        if self.step_size < 1:
            raise ValueError("step musst be >= 1")
        self.min_length = int(min_length)
        if self.min_length < 1:
            raise ValueError("min_length must be >= 1")
        self.max_steps = int(max_steps)

        # Accum init
        self.drift_rate = float(drift_rate)
        self.accumulate_sigma = float(accumulate_sigma)
        self.threshold = float(threshold)

        self.p_pos = float(p_pos)
        self.p_neg = float(p_neg)
        self.last_obs = 0.0
        self.evidence = 0.0

        # Memory init
        initial_bins = (0, 1)
        self.grid = DiscreteDistributionGrid(initial_bins=initial_bins)
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

    def _accumulate_hit(self, obs, evidence):
        """Weight the evidence for an observation"""
        # If no info, treat as p_neg?
        hit = 0
        if np.isclose(obs, 0.0):
            return obs, hit, True

        # Consider things....
        stop = False
        delta = self.drift_rate * obs
        for n in range(self.max_steps):
            evidence = delta * self.drift_rate * self._w(n)
            if evidence > self.threshold:
                hit = 1
                stop = True
                break

        return evidence, hit, stop

    def forward(self, state):
        """Step forward."""
        # Parse
        pos, obs = state

        # Deliberate by accumulation
        self.evidence, hit, stop = self._accumulate_hit(obs, self.evidence)

        # Add to grid memory,
        p_old = deepcopy(self.grid.probs(pos))
        self.grid(pos, hit)
        p_new = deepcopy(self.grid.probs(pos))

        # Info gain (by KL), and its grad
        info_gain = scientropy(p_old, qk=p_new, base=2)
        try:
            grad = info_gain - self.history["agent_info_gain"][-1]
            # grad = self.history["agent_info_gain"][-1] - info_gain
        except IndexError:
            grad = info_gain
            # grad = -info_gain

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

            #  Set new direction
            action = self.angle

        self.total_distance += self.step

        # Log
        self.history["agent_obs"].append(deepcopy(obs))
        self.history["agent_delta"].append(deepcopy(obs - self.last_obs))
        self.history["agent_evidence"].append(deepcopy(self.evidence))
        self.history["agent_stop"].append(deepcopy(stop))
        self.history["agent_info_gain"].append(deepcopy(info_gain))
        self.history["agent_hit"].append(deepcopy(hit))
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


class DiffusionGrid(Agent2d):
    """Diffusion search, on a NSEW grid"""
    def __init__(self, min_length=1, scale=2, step_size=1):
        super().__init__()
        self.possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        self.step_size = int(step_size)
        if self.step_size < 1:
            raise ValueError("step musst be >= 1")

        self.min_length = int(min_length)
        if self.min_length < 1:
            raise ValueError("min_length must be >= 1")

        self.scale = float(scale)
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

        # Keep going?
        if self.l > self.step:
            # Step
            self.step += self.step_size
            # Clip?
            if self.step > self.l:
                self.step = int(self.step - self.l)
            self.num_step += 1
        # Turn?
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

        # Safe intial values
        self.l = self._l(np.zeros(2))
        self.angle = self._angle(np.zeros(2))

        # Clean
        self.num_turn = 0
        self.num_step = 0
        self.step = 0
        self.total_distance = 0.0
        self.history = defaultdict(list)


class GradientDiffusionGrid(Agent2d):
    """Diffusion search, but the sense/obs gradient 
    effects turn probability. 
    
    Note: 
    ----
    Positive gradients set the turn prob. to p_pos.
    """
    def __init__(self, min_length=1, scale=2, p_neg=0.8, p_pos=0.2):
        super().__init__()
        self.possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        self.step_size = 1
        if self.step_size < 1:
            raise ValueError("step musst be >= 1")

        self.min_length = int(min_length)
        if self.min_length < 1:
            raise ValueError("min_length must be >= 1")

        self.scale = float(scale)
        self.p_pos = float(p_pos)
        self.p_neg = float(p_neg)
        self.last_obs = 0.0
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


class QlearnGrid(Agent2d):
    pass


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


class LevyGrid(Agent2d):
    """Levy search, on a NSEW grid"""
    def __init__(self, min_length=1, exponent=2, step_size=1):
        super().__init__()
        self.possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        self.step_size = int(step_size)
        if self.step_size < 1:
            raise ValueError("step musst be >= 1")

        self.min_length = int(min_length)
        if self.min_length < 1:
            raise ValueError("min_length must be >= 1")

        self.exponent = exponent
        self.renorm = (self.exponent - 1) / (self.min_length
                                             **(1 - self.exponent))

    def _angle(self, state):
        """Sample NSEW"""
        i = int(self.np_random.randint(0, len(self.possible_actions)))
        return self.possible_actions[i]

    def _l(self, state):
        """Sample length"""
        i = 0
        while True and i < 10000:
            i += 1
            xi = self.np_random.rand()
            l = int(self.renorm * np.power(xi, (-1 / self.exponent)))
            if l > self.min_length:
                return l

    def forward(self, state):
        """Step forward."""

        # Keep going?
        if self.l > self.step:
            self.step += self.step_size
            self.num_step += 1
        # Turn?
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

        # Safe intial values
        self.l = self._l(np.zeros(2))
        self.angle = self._angle(np.zeros(2))

        # Clean
        self.num_turn = 0
        self.num_step = 0
        self.step = 0
        self.total_distance = 0.0
        self.history = defaultdict(list)


class FreezeLevyGrid(LevyGrid):
    def __init__(self, p_freeze=0.5, min_length=1, exponent=2, step_size=1):
        self.p_freeze = float(p_freeze)
        self.num_freeze = 0
        super().__init__(min_length, exponent, step_size)

    def forward(self, state):
        """Step forward."""
        # Default
        self.freeze = False

        # Keep going?
        if self.l > self.step:
            self.step += self.step_size
            self.num_step += 1
        # Freeze or turn?
        else:
            if self.p_freeze > self.np_random.rand():
                self.freeze = True
                self.angle = (0, 0)  # Go no where
                self.step = 0
                self.l = 0
                self.num_freeze += 1
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
        self.history["agent_freeze"].append(deepcopy(self.freeze))
        self.history["agent_num_freeze"].append(deepcopy(self.num_freeze))
        self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        self.history["agent_angle"].append(deepcopy(self.angle))
        self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_total_l"].append(deepcopy(self.total_distance))
        self.history["agent_step"].append(deepcopy(self.step_size))
        self.history["agent_num_step"].append(deepcopy(self.num_step))
        self.history["agent_action"].append(deepcopy(action))

        return action

    def reset(self):
        self.num_freeze = 0
        super().reset()


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

class GreedyPredatorGrid(Agent2d):
    """Greedily moves towards nearest other agent on the grid. 
       Must use with multi_experiment().
       Note: Agnostic to agent type & whether the other agents are prey/predator.
             Future implementation might take into account agent type.
    """

    def __init__(self, step_size=1):
        super().__init__()
        self.env = env
        self.possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        self.step_size = int(step_size)
        if self.step_size < 1:
            raise ValueError("step musst be >= 1")

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
        """Step forward greedily toward nearest other agent.

          state is assumed to be list len()==2:
          - state[0]: 2x1 state of this agent
          - state[1]: list of 2x1, states of all other agents
        """

        # greedily choose angle
        state_agent = state[0]
        states_others = np.asarray(state[1])

        # get distances to all other agents
        distances = np.linalg.norm(state_agent[None] - states_others, axis=1)
        other_closest_idx = np.argmin(distances)

        # now get action to minimize distance to closest agent
        distances = np.linalg.norm(
            states_others[other_closest_idx][None] - np.asarray(self.possible_actions), axis=1)
        greedy_action = self.possible_actions[np.argmin(distances)]

        # Step
        action = greedy_action
        self.total_distance += self.step_size

        # Log
        # self.history["agent_num_turn"].append(deepcopy(self.num_turn))
        # self.history["agent_angle"].append(deepcopy(self.angle))
        # self.history["agent_l"].append(deepcopy(self.l))
        self.history["agent_total_l"].append(deepcopy(self.total_distance))
        self.history["agent_step"].append(deepcopy(self.step_size))
        # self.history["agent_num_step"].append(deepcopy(self.num_step))
        self.history["agent_action"].append(deepcopy(action))

        return action

    def reset(self):
        """Reset all counters, turns, and steps"""

        # Clean
        self.num_turn = 0
        self.num_step = 0
        self.step = 0
        self.total_distance = 0.0
        self.history = defaultdict(list)
