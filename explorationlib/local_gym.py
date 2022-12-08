#! /usr/bin/env python
import numpy as np
import random

from copy import deepcopy
from itertools import cycle
from collections import defaultdict
from sklearn.neighbors import KDTree

import gym
from gym import spaces
from gym.utils import seeding
from gym_maze.envs.maze_view_2d import MazeView2D
from gym_maze.envs import MazeEnv
from explorationlib.agent import Levy2d

# Gym is annoying these days...
import warnings
warnings.filterwarnings("ignore")

#countSteps = 0

# -------------------------------------------------------------------------
# Enviroments


# ---------------------------------------------------------------------------
# Basic Bandits
# - A base class,
# - then several working examples
# ---------------------------------------------------------------------------
class BanditEnv(gym.Env):
    """
    n-armed bandit environment  

    Params
    ------
    p_dist : list
        A list of probabilities of the likelihood that a particular bandit 
        will pay out
    r_dist : list or list or lists
        A list of either rewards (if number) or means and standard deviations
        (if list) of the payout that bandit has
    """
    def __init__(self, p_dist, r_dist):
        if len(p_dist) != len(r_dist):
            raise ValueError(
                "Probability and Reward distribution must be the same length")

        if min(p_dist) < 0 or max(p_dist) > 1:
            raise ValueError("All probabilities must be between 0 and 1")

        for reward in r_dist:
            if isinstance(reward, list) and reward[1] <= 0:
                raise ValueError(
                    "Standard deviation in rewards must all be greater than 0")

        self.p_dist = p_dist
        self.r_dist = r_dist

        self.n_bandits = len(p_dist)
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(1)
        self.seed()

    def step(self, action):
        assert self.action_space.contains(action)
        self.state = 0
        self.reward = 0
        self.done = False

        if self.np_random.uniform() < self.p_dist[action]:
            if not isinstance(self.r_dist[action], list):
                self.reward = self.r_dist[action]
            else:
                self.reward = self.np_random.normal(self.r_dist[action][0],
                                                    self.r_dist[action][1])

        return self.state, self.reward, self.done, {}

    def last(self):
        return self.state, self.reward, self.done, {}

    def reset(self):
        self.state = 0
        self.reward = 0
        self.done = False

    def render(self, mode='human', close=False):
        pass


class BanditUniform4(BanditEnv):
    """A 4 armed bandit."""
    def __init__(self, p_min=0.1, p_max=0.3, p_best=0.6, best=2):
        self.best = [best]
        self.num_arms = 4

        # ---
        self.p_min = p_min
        self.p_max = p_max
        self.p_best = p_best

        # Generate intial p_dist
        # (gets overwritten is seed())
        p_dist = np.random.uniform(self.p_min, self.p_max,
                                   size=self.num_arms).tolist()
        p_dist[self.best[0]] = self.p_best

        # reward
        r_dist = [1] * self.num_arms

        # !
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Reset p(R) dist with the seed
        self.p_dist = self.np_random.uniform(self.p_min,
                                             self.p_max,
                                             size=self.num_arms).tolist()
        self.p_dist[self.best[0]] = self.p_best

        return [seed]


class BanditUniform10(BanditEnv):
    """A 4 armed bandit."""
    def __init__(self, p_min=0.1, p_max=0.3, p_best=0.6, best=2):
        self.best = [best]
        self.num_arms = 10

        # ---
        self.p_min = p_min
        self.p_max = p_max
        self.p_best = p_best

        # Generate intial p_dist (gets overwritten is seed())
        p_dist = np.random.uniform(self.p_min, self.p_max,
                                   size=self.num_arms).tolist()
        p_dist[self.best[0]] = self.p_best

        # reward
        r_dist = [1] * self.num_arms

        # !
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Reset p(R) dist with the seed
        self.p_dist = self.np_random.uniform(self.p_min,
                                             self.p_max,
                                             size=self.num_arms).tolist()
        self.p_dist[self.best[0]] = self.p_best

        return [seed]


class BanditChange4:
    """Change the best to the worst - BanditUniform4"""
    def __init__(self,
                 num_change=60,
                 p_min=0.1,
                 p_max=0.3,
                 p_best=0.6,
                 p_change=0.1):
        super().__init__()

        # Init
        self.num_arms = 4
        self.num_change = num_change
        self.state = 0
        self.reward = 0
        self.done = False

        self.p_min = p_min
        self.p_max = p_max
        self.p_best = p_best
        self.p_change = p_change

        # Original...
        self.best = 2
        self.orginal = BanditUniform4(p_min=self.p_min,
                                      p_max=self.p_max,
                                      p_best=self.p_best,
                                      best=self.best)
        # Create change
        self.change = deepcopy(self.orginal)
        self.change.p_dist[self.best] = self.p_change
        self.change.best = [np.argmax(self.change.p_dist)]

    def step(self, action):
        # Reset
        self.state = 0
        self.reward = 0
        self.done = False

        # Step
        if self.num_steps < self.num_change:
            self.state, self.reward, self.done, _ = self.orginal.step(action)
        else:
            self.state, self.reward, self.done, _ = self.change.step(action)

        self.num_steps += 1

        # Return
        return self.state, self.reward, self.done, {}

    def last(self):
        return self.state, self.reward, self.done, {}

    def reset(self):
        self.num_steps = 0
        self.orginal.reset()
        self.change.reset()

    def seed(self, seed=None):
        # Set
        self.np_random, seed = seeding.np_random(seed)
        self.orginal.seed(seed)

        # Copy
        self.change = deepcopy(self.orginal)

        # Update
        self.change.p_dist[self.best] = self.p_change
        self.change.best = [np.argmax(self.change.p_dist)]

        return [seed]

    def render(self, mode='human', close=False):
        pass


# -------------------------------------------------------------------------
# Maze
# - A modified version of MazeEnv
# -------------------------------------------------------------------------
class ScentMazeEnv(MazeEnv):
    """
    A maze, where maze exist has a reward that emits a scent

    Based on gym_maze's MazeEnv
    https://github.com/MattChanTK/gym-maze

    See above for info in init params
    """
    def __init__(self,
                 maze_file=None,
                 maze_size=None,
                 mode=None,
                 enable_render=False):
        """A maze, where the target has a scent"""
        # Init Maze then...
        super().__init__(maze_file=maze_file,
                         maze_size=maze_size,
                         mode=mode,
                         enable_render=enable_render)
        self.scent = None
        self.reward = 0
        self.state = None
        self.info = {}

    def add_scent(self, scent, noise_sigma=0.0):
        self.noise_sigma = noise_sigma
        self.scent_pdf = scent

    def step(self, action):
        # Maze step
        self.position, self.reward, self.done, self.info = super().step(action)
        # Scent?
        if self.scent_pdf is not None:
            # Get?
            x, y = int(self.position[0]), int(self.position[1])
            scent = self.scent_pdf[x, y]
            # Add noise?
            noise = np.abs(self.np_random.normal(0, self.noise_sigma))
            scent = +noise
        else:
            scent = 0.0
        # !
        self.state = (self.position, scent)
        return (self.state, self.reward, self.done, self.info)

    def last(self):
        return (self.state, self.reward, self.done, self.info)


# ---------------------------------------------------------------------------
# Basic 2D environments
# - Three bases classes
# - Then a Scent having class
# - Then multi-player classes
# ---------------------------------------------------------------------------
class Field(gym.Env):
    """An open-field to explore, with no boundries."""
    def __init__(self):
        self.info = {}
        self.reward = 0.0
        self.done = False

        self.detection_radius = None
        self.num_targets = None
        self.targets = None
        self.values = None

        self.seed()
        self.reset()

    def step(self, action):
        """Agent takes a step, then checks targets/friends."""
        self.state += action
        self.check_targets()
        return self.last()

    def last(self):
        """Transition - (state, reward, done, info)"""
        return (self.state, self.reward, self.done, self.info)

    def add_targets(self,
                    targets,
                    values,
                    detection_radius=1,
                    kd_kwargs=None,
                    p_target=1.0):
        """Add targets, and their values"""

        # Sanity
        if len(targets) != len(values):
            raise ValueError("targets and values must match.")

        # Will it be there?
        self.p_target = p_target
        self.detection_radius = detection_radius

        # Store raw targets simply (list)
        self.num_targets = len(targets)
        self.targets = targets
        self.values = values

        # Init Target tree (for fast lookup)
        if kd_kwargs is None:
            kd_kwargs = {}

        self._kd = KDTree(np.vstack(self.targets), **kd_kwargs)

    def check_targets(self):
        """Check for targets, and update self.reward"""
        # Short circuit if no targets
        if self.targets is None:
            return None

        # Reinit reward. Assume we are not at a target
        self.reward = 0.0

        # How far are we and is it close enough to
        # generate a reward? AKA are we at a target?
        state = np.atleast_2d(np.asarray(self.state))
        dist, ind = self._kd.query(state, k=1)

        # Care about the closest; Fmt
        dist = float(dist[0])
        ind = int(ind[0])
        self.ind = ind  # Save

        # Test proximity
        if dist <= self.detection_radius:
            # What's the value?
            value = self.values[ind]

            # Ignore None
            if value is None:
                self.reward = 0.0

            # Coin flip
            if self.np_random.rand() <= self.p_target:
                self.reward = value
            else:
                self.reward = 0.0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.zeros(2)
        self.reward = 0.0
        self.last()

    def render(self, mode='human', close=False):
        pass


class Bounded(Field):
    """An open-field to explore, with boundries.
    
    Parameters
    ----------
    boundary : 2-tuple (x, y)
        The absolute value of the 2d boundary.
    mode: str
        How to handle collisions with the boundary. 
        - stopping: stop movement
        - absorbing: stop movement and end the run
        - periodic: loop back around, aka pacman mode
                    (not yet implemented).
        """
    def __init__(self, boundary, mode="stopping"):
        # Init the field
        super().__init__()

        # New attrs
        self.boundary = boundary
        self.mode = mode

        # Sanity testing
        for s in boundary:
            if s < 0:
                raise ValueError("boundary must be positive")
            elif not np.isfinite(s):
                raise ValueError("boundary must be finite")

        valid = ("stopping", "absorbing", "periodic")
        if self.mode not in valid:
            raise ValueError(f"mode must be {valid}")

    def step(self, action):
        # step
        super().step(action)

        # check bounds. clipping and stopping
        # in a mode dependent way
        for i, s in enumerate(self.state):
            if np.abs(s) > self.boundary[i]:
                if self.mode == "stopping":
                    self.state[i] = np.sign(s) * self.boundary[i]
                elif self.mode == "absorbing":
                    self.state[i] = np.sign(s) * self.boundary[i]
                    self.done = True
                elif self.mode == "periodic":
                    raise NotImplementedError("[TODO]")
                else:
                    raise ValueError("Invalid mode")
        # ...
        super().check_targets()
        return self.last()


class Grid(Field):
    """A discrete open-ended grid-world."""
    def __init__(self, mode="discrete"):
        super().__init__()
        self.mode = mode

    def _card_step(self, action):
        # Interpret action as a cardinal direction
        action = int(action)
        if action == 0:
            super().step((0, 1))
        elif action == 1:
            super().step((0, -1))
        elif action == 2:
            super().step((1, 0))
        elif action == 3:
            super().step((-1, 0))
        super().check_targets()

    def step(self, action):
        if self.mode == "discrete":
            self._card_step(action)
        else:
            action = [int(a) for a in action]
            super().step(action)

        return self.last()

    def reset(self):
        self.state = np.zeros(2, dtype=int)
        self.last()


# ---
class ScentGrid(Grid):
    """Am open-grid, with scent"""
    def __init__(self, mode="discrete"):
        super().__init__(mode=mode)
        self.scent = None
        self.scent_fn = None
        self.obs = 0.0

    def add_scent(self,
                  target,
                  value,
                  coord,
                  scent,
                  detection_radius=1.0,
                  noise_sigma=0.0,
                  p_target=1.0):
        # Offset coords by target location
        # align them in other words
        self.scent_x_coord = coord[0] + target[0]
        self.scent_y_coord = coord[1] + target[1]

        # Set scent and its params
        self.noise_sigma = noise_sigma
        self.scent_pdf = scent
        self.add_targets([target], [value],
                         detection_radius=detection_radius,
                         kd_kwargs=None,
                         p_target=p_target)

        def scent_fn(state):
            x, y = state
            i = find_nearest(self.scent_x_coord, x)
            j = find_nearest(self.scent_y_coord, y)

            # Add noise?
            noise = np.abs(self.np_random.normal(0, self.noise_sigma))
            return self.scent_pdf[i, j] + noise

        self.scent_fn = scent_fn

    def add_scents(
            self,
            targets,
            values,
            coord,  # assume shared
            scents,
            detection_radius=1.0,
            noise_sigma=0.0,
            p_target=1.0):
        """Add several scents, and targets"""
        self.noise_sigma = noise_sigma
        self.scent_pdfs = scents
        self.add_targets(targets,
                         values,
                         detection_radius=detection_radius,
                         kd_kwargs=None,
                         p_target=p_target)

        # Offset coords by target location
        # align them in other words
        self.scent_x_coords = []
        self.scent_y_coords = []
        for target in self.targets:
            self.scent_x_coords.append(coord[0] + target[0])
            self.scent_y_coords.append(coord[1] + target[1])

        def scent_fn(state):
            # Pos
            x, y = state

            # Sum scents from all targets @ pos
            summed = 0.0
            for ind in range(self.num_targets):
                i = find_nearest(self.scent_x_coords[ind], x)
                j = find_nearest(self.scent_y_coords[ind], y)
                summed += self.scent_pdfs[ind][i, j]

            # Add noise?
            noise = np.abs(self.np_random.normal(0, self.noise_sigma))
            return summed + noise

        self.scent_fn = scent_fn

    def step(self, action):
        # Move
        super().step(action)  # sets self.ind, etc

        # Scent
        if self.scent_fn is not None:
            x, y = int(self.state[0]), int(self.state[1])
            self.obs = self.scent_fn((x, y))
        else:
            self.obs = 0.0

        # !
        self.state_obs = (self.state, self.obs)
        return self.last()

    def last(self):
        return (self.state_obs, self.reward, self.done, self.info)

    def reset(self):
        self.state = np.zeros(2, dtype=int)
        self.obs = 0.0
        self.state_obs = (self.state, self.obs)
        self.last()

# ---
class ScentGridMovingTargets(Grid):
    """An open-grid, with scent and moving targets"""
    def __init__(self, mode="discrete"):
        super().__init__(mode=mode)
        self.scent = None
        self.scent_fn = None
        self.obs = 0.0

    def add_scent(self,
                  target,
                  value,
                  coord,
                  scent,
                  detection_radius=1.0,
                  noise_sigma=0.0,
                  p_target=1.0):
        # Offset coords by target location
        # align them in other words
        self.scent_x_coord = coord[0] + target[0]
        self.scent_y_coord = coord[1] + target[1]

        # Set scent and its params
        self.noise_sigma = noise_sigma
        self.scent_pdf = scent
        self.add_targets([target], [value],
                         detection_radius=detection_radius,
                         kd_kwargs=None,
                         p_target=p_target)

        def scent_fn(state):
            x, y = state
            i = find_nearest(self.scent_x_coord, x)
            j = find_nearest(self.scent_y_coord, y)

            # Add noise?
            noise = np.abs(self.np_random.normal(0, self.noise_sigma))
            return self.scent_pdf[i, j] + noise

        self.scent_fn = scent_fn

    def add_scents(
            self,
            targets,
            values,
            coord,  # assume shared
            scents,
            detection_radius=1.0,
            noise_sigma=0.0,
            p_target=1.0):
        """Add several scents, and targets"""
        self.noise_sigma = noise_sigma
        self.scent_pdfs = scents
        self.add_targets(targets,
                         values,
                         detection_radius=detection_radius,
                         kd_kwargs=None,
                         p_target=p_target)

        # Offset coords by target location
        # align them in other words
        self.scent_x_coords = []
        self.scent_y_coords = []
        for target in self.targets:
            self.scent_x_coords.append(coord[0] + target[0])
            self.scent_y_coords.append(coord[1] + target[1])

        def scent_fn(state):
            # Pos
            x, y = state

            # Sum scents from all targets @ pos
            summed = 0.0
            for ind in range(self.num_targets):
                i = find_nearest(self.scent_x_coords[ind], x)
                j = find_nearest(self.scent_y_coords[ind], y)
                summed += self.scent_pdfs[ind][i, j]

            # Add noise?
            noise = np.abs(self.np_random.normal(0, self.noise_sigma))
            return summed + noise

        self.scent_fn = scent_fn

    def step(self, action):
        # Move
        super().step(action)  # sets self.ind, etc
        #global countSteps += 1

        # Scent
        if self.scent_fn is not None:
            x, y = int(self.state[0]), int(self.state[1])
            self.obs = self.scent_fn((x, y))
        else:
            self.obs = 0.0
            
        # New FP Code
        newTargets = []
        for target in self.targets:
            #if countSteps%50 == 0:
            #print("countSteps = ", countSteps)
            target[0] += random.randint(-3, 3)
            if target[0] > 10:
                target[0] = 10
            elif target[0] < -10:
                target[0] = -10
            target[1] += random.randint(-3, 3)
            if target[1] > 10:
                target[1] = 10
            elif target[1] < -10:
                target[1] = -10
            newTargets.append(target)
        #self.add_targets(newTargets, self.values)
        values = self.values
        targets = newTargets

        scentsX = []
        count = 0
        for _ in range(num_targets):
          if values[count] == 1:
            newAmp = 1
            newSig = 2
          else:
            newAmp = 1.5
            newSig = 1.75

          coord, scent = create_grid_scent_patches(
            target_boundary, p=1.0, amplitude=newAmp, sigma=newSig)
          scentsX.append(scent)
          count += 1
        self.add_scents(targets, values, coord, scentsX, noise_sigma=noise_sigma)
                
        # !
        self.state_obs = (self.state, self.obs)
        return self.last()

    def last(self):
        return (self.state_obs, self.reward, self.done, self.info)

    def reset(self):
        self.state = np.zeros(2, dtype=int)
        self.obs = 0.0
        self.state_obs = (self.state, self.obs)
        self.last()

# ---
# Multi
class CompetitiveField(gym.Env):
    """Am open-ended field, with prey."""
    def __init__(self, num_agents=2):
        self.num_agents = num_agents
        self.info = {}
        self.reward = 0.0
        self.done = False
        self.dead = None

        self.detection_radius = None
        self.targets = None
        self.initial_targets = None
        self.values = None
        self.seed()
        self.reset()

    def step(self, action, n):
        # Step
        self.n = n
        self.state[n] += action
        # Update/check targets
        self.update_targets(n)
        self.reward = 0.0
        self.check_targets(n)

        # -
        return self.last()

    def last(self):
        """Return last: (state, reward, done, info)"""
        return (self.state, self.reward, self.done, self.info)

    def add_targets(self,
                    index,
                    targets,
                    values,
                    detection_radius=1,
                    p_target=1.0):
        """Add targets, and their values.
        
        Params
        ------
        index : list
            The index of agents who act as targets (aka prey)
        targets : list
            Intial target (prey) locations
        values : list
            Intial target (prey) values
        detection_radius : int (> 1)
            How far the predator can 'see'
        p_target : float (0-1)
            Prob. predator find targets (in the detection_radius)
        """
        self.target_index = index

        # Sanity
        if len(index) != len(targets):
            raise ValueError("index and targets must match.")
        if len(targets) != len(values):
            raise ValueError("targets and values must match.")

        # Will it be there?
        self.p_target = p_target

        # Store raw targets simply (list)
        self.num_targets = len(targets)
        self.initial_targets = targets
        self.targets = deepcopy(self.initial_targets)
        self.values = values
        self.detection_radius = detection_radius

        # Step targets
        for i, t in zip(self.target_index, self.targets):
            self.step(t, i)

        # Init tree
        self._kd = KDTree(np.vstack(self.targets))

    def check_targets(self, n):
        """Check for targets, and update self.reward if
        some are found in the given detection_radius.

        Note: the deault d_func is the euclidian distance. 
        To override provide a func(x, y) -> distance.
        """
        if n not in self.target_index:
            # Short circuit if no targets
            if self.targets is None:
                return None

            # Reinit reward. Assume we are not at a target
            self.reward = 0.0

            # How far are we and is it close enough to
            # generate a reward? AKA are we at a target?
            state = np.atleast_2d(np.asarray(self.state[n]))
            dist, ind = self._kd.query(state, k=1)

            # Care about the closest; Fmt
            dist = float(dist[0])
            ind = int(ind[0])
            code = self.target_index[ind]

            # Test proximity
            if code not in self.dead:
                if dist <= self.detection_radius:

                    # What's the value?
                    value = self.values[ind]

                    # Coin flip
                    if self.np_random.rand() <= self.p_target:
                        self.reward = value
                        self.dead.append(code)  # death if detection
                    else:
                        self.reward = 0.0

    def update_targets(self, n):
        # Update targets
        if n in self.target_index:
            i = self.target_index.index(n)
            self.targets[i] = self.state[n]

        # Rebuild tree
        self._kd = KDTree(np.vstack(self.targets))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.n = 0
        # Reinit
        self.state = [np.zeros(2) for _ in range(self.num_agents)]

        # Restep targets
        if self.initial_targets is not None:
            self.targets = deepcopy(self.initial_targets)
            for i, t in zip(self.target_index, self.targets):
                self.step(t, i)

        # Revive the dead!
        self.dead = []

        # Reset
        self.reward = 0.0
        self.last()

    def render(self, mode='human', close=False):
        pass


class CompetitiveGrid(CompetitiveField):
    """Am open-ended grid-world, with prey."""
    def __init__(self, num_agents=2):
        super().__init__(num_agents=num_agents)

    def step(self, action, n):
        # Force int... so we are on a grid.
        action = [int(a) for a in action]
        super().step(action, n)

        return self.last()


class CooperativeField(gym.Env):
    """Am open-ended field, with prey, who form teams.
    
    Params
    -----
    num_agents: int
        The total number of agents

    Notes
    -----
    Teams move as one 'agent', with a value equal
    to the total individual values. 
    
    However!

    Predator `detection_radius` is scaled by the 
    value of each agent or team. That is, 
    `detection_radius * value`.
    """
    def __init__(self, num_agents=2):
        self.num_agents = num_agents
        self.info = {}
        self.reward = 0.0
        self.done = False

        self.detection_radius = None
        self.friend_radius = None
        self.targets = None
        self.initial_targets = None
        self.values = None
        self.initial_values = None
        self.seed()
        self.reset()

    def step(self, action, n):
        """Agent 'n' takes a step, then checks targets/friends."""
        # Step
        self.n = n
        self.state[n] += action

        # Update/check targets,
        # then check friends
        self.update_targets(n)
        self.reward = 0.0
        self.check_friends(n)
        self.check_targets(n)

        # Return last env transition
        return self.last()

    def last(self):
        """Transition - (state, reward, done, info)"""
        return (self.state, self.reward, self.done, self.info)

    def add_targets(self,
                    index,
                    targets,
                    values,
                    detection_radius=1,
                    friend_radius=1,
                    p_target=1.0,
                    p_friend=1.0):
        """Add targets, and their values.
        
        Params
        ------
        index : list
            The index of agents who act as targets (aka prey)
        targets : list
            Intial target (prey) locations
        values : list
            Intial target (prey) values
        detection_radius : int (> 1)
            How far the predator can 'see'
        friend_radius : int (> 1)
            How far the targets (prey) can 'see'
        p_target : float (0-1)
            Prob. predator find targets (in the detection_radius)
        p_friend : float (0-1)
            Prob. prey become friends (in the friend_radius)
        """
        # An index to seperate prey (targets)
        # from predators, when all are agents.
        self.target_index = index

        # Sanity check
        if len(index) != len(targets):
            raise ValueError("index and targets must match.")
        if len(targets) != len(values):
            raise ValueError("targets and values must match.")

        # Will it be there?
        self.p_target = p_target
        self.p_friend = p_friend
        self.detection_radius = detection_radius
        self.friend_radius = friend_radius

        # Store raw targets simply (list)
        self.num_targets = len(targets)

        self.initial_targets = deepcopy(targets)
        self.initial_values = deepcopy(values)

        self.targets = deepcopy(self.initial_targets)
        self.values = deepcopy(self.initial_values)

        # Step targets to initial positions
        for i, t in zip(self.target_index, self.targets):
            self.step(t, i)

        # Init Target tree (for fast lookup)
        self._kd = KDTree(np.vstack(self.targets))

    def update_targets(self, n):
        """Move target to new location"""
        # Update target location
        if n in self.target_index:
            i = self.target_index.index(n)
            self.targets[i] = self.state[n]

        # Rebuild tree
        self._kd = KDTree(np.vstack(self.targets))

    def check_targets(self, n):
        """Check for targets, and update self.reward"""
        # No targets
        if self.targets is None:
            return None

        # Is 'n' a predator?
        if n not in self.target_index:
            # How far are targets?
            state = np.atleast_2d(np.asarray(self.state[n]))
            dist, ind = self._kd.query(state, k=1)

            # Pick the closest
            dist = float(dist[0])
            ind = int(ind[0])
            code = self.target_index[ind]

            # What's the value?
            value = self.values[ind]

            # Are they in the radius?
            if code not in self.dead:
                if dist <= (self.detection_radius * value):
                    # Detection coin flip:
                    if self.np_random.rand() <= self.p_target:
                        # Pret value is
                        self.reward = value

                        # Death to prey, if detection
                        self.values[ind] = 0.0
                        self.dead.append(code)
                    else:
                        self.reward = 0.0

    def check_friends(self, n):
        """Check for friends, team up?, and update self.reward"""

        # No targets
        if self.targets is None:
            return None

        # Is 'n' a prey?
        if n in self.target_index:
            # How far are friends?
            state = np.atleast_2d(np.asarray(self.state[n]))
            dist, ind = self._kd.query(state, k=2)

            # Pick the closest (not itself, aka 0)
            dist = float(dist[0][1])
            ind = int(ind[0][1])
            code = self.target_index[ind]

            # Are they in the radius?
            if code not in self.dead:
                if (0.0 < dist <= self.friend_radius):
                    # Detection coin flip:
                    if self.np_random.rand() <= self.p_friend:

                        # What's the value?
                        value = self.values[ind]
                        # self.reward = value

                        # To form a team:
                        #
                        # 1. Merge value at n; zero at ind
                        # 2. Then kill at ind/code
                        self.values[n] += value
                        self.values[ind] = 0.0

                        self.friends.append((n, ind))
                        self.dead.append(code)  # death if detection
                    else:
                        self.reward = 0.0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # Reinit
        self.n = 0
        self.state = [np.zeros(2) for _ in range(self.num_agents)]

        # Restep targets and values
        if self.initial_targets is not None:
            self.targets = deepcopy(self.initial_targets)
            for i, t in zip(self.target_index, self.targets):
                self.step(t, i)

        if self.initial_values is not None:
            self.values = deepcopy(self.initial_values)

        # Revive the dead!
        self.dead = []
        self.friends = []

        # Reset
        self.reward = 0.0
        self.last()

    def render(self, mode='human', close=False):
        pass


class CooperativeGrid(CooperativeField):
    """Am open-ended grid-world, with prey, who form teams."""
    def __init__(self, num_agents=2):
        super().__init__(num_agents=num_agents)

    def step(self, action, n):
        # Force int... so we are on a grid.
        action = [int(a) for a in action]
        super().step(action, n)

        return self.last()


class CutthroatField(gym.Env):
    """An open-ended field, with predators who attack everyone!.
    
    Params
    -----
    num_agents: int
        The total number of agents
    """
    def __init__(self, num_agents=2):
        self.num_agents = num_agents
        self.info = {}
        self.reward = 0.0
        self.done = False

        # Prey
        self.detection_radius = None
        self.targets = None
        self.initial_targets = None
        self.values = None
        self.initial_values = None

        # Pred
        self.enemy_radius = None
        self.initial_enemy = None
        self.enemy = None
        self.enemy_index = None
        self.enemy_values = None

        # Init
        self.seed()
        self.reset()

    def step(self, action, n):
        """Agent 'n' takes a step, then checks targets/enemys."""
        # Step
        self.n = n
        self.state[n] += action

        # Update/check targets,
        # then check enemys
        self.reward = 0.0

        if self.targets is not None:
            self.update_targets(n)
            self.check_targets(n)
        if self.enemy is not None:
            self.update_enemy(n)
            self.check_enemy(n)

        # Return last env transition
        return self.last()

    def last(self):
        """Transition - (state, reward, done, info)"""
        return (self.state, self.reward, self.done, self.info)

    def add_targets(self,
                    index,
                    targets,
                    values,
                    detection_radius=1,
                    p_target=1.0):
        """Add targets, and their values.
        
        Params
        ------
        index : list
            The index of agents who act as targets (aka prey)
        targets : list
            Intial target (prey) locations
        values : list
            Intial target (prey) values
        detection_radius : int (> 1)
            How far the predator can 'see'
        p_target : float (0-1)
            Prob. predator find targets (in the detection_radius)
        """
        self.target_index = index

        # Sanity
        if len(index) != len(targets):
            raise ValueError("index and targets must match.")
        if len(targets) != len(values):
            raise ValueError("targets and values must match.")

        # Will it be there?
        self.p_target = p_target

        # Store raw targets simply (list)
        self.num_targets = len(targets)
        self.initial_targets = targets
        self.targets = deepcopy(self.initial_targets)
        self.values = deepcopy(values)
        self.initial_values = deepcopy(values)
        self.detection_radius = detection_radius

        # Step targets
        for i, t in zip(self.target_index, self.targets):
            self.step(t, i)

        # Init tree
        self._kd = KDTree(np.vstack(self.targets))

    def update_targets(self, n):
        """Move target to new location"""
        # Update target location
        if n in self.target_index:
            i = self.target_index.index(n)
            self.targets[i] = self.state[n]

        # Rebuild tree
        self._kd = KDTree(np.vstack(self.targets))

    def check_targets(self, n):
        """Check for targets, and update self.reward"""
        # No targets
        if self.targets is None:
            return None

        # Is 'n' a predator?
        if n not in self.target_index:
            # How far are targets?
            state = np.atleast_2d(np.asarray(self.state[n]))
            dist, ind = self._kd.query(state, k=1)

            # Pick the closest
            dist = float(dist[0])
            ind = int(ind[0])
            code = self.target_index[ind]

            # What's the value?
            value = self.values[ind]

            # Are they in the radius?
            if code not in self.dead:
                if dist <= (self.detection_radius):
                    # Detection coin flip:
                    if self.np_random.rand() <= self.p_target:
                        # Pret value is
                        self.reward = value

                        # Death to prey, if detection
                        self.values[ind] = 0.0
                        self.dead.append(code)
                    else:
                        self.reward = 0.0

    def add_enemy(self, index, enemy, values, enemy_radius=1, p_enemy=1.0):
        """Add targets, and their values.
        
        Params
        ------
        index : list
            The index of agents who act as targets (aka prey)
        enemy : list
            Initial predator locations
        values : list
            Initial target (prey) values
        enemy_radius : int (> 1)
            How far the predators can 'see' each other
        p_enemy : float (0-1)
            Prob. predatos kill each other.
        """
        # An index to seperate enemies
        # aka predators, when all are agents.
        self.enemy_index = index

        # Sanity check
        if len(index) != len(enemy):
            raise ValueError("index and enemies must match.")
        if len(enemy) != len(values):
            raise ValueError("enemies and values must match.")

        # Will it be there?
        self.p_enemy = p_enemy
        self.enemy_radius = enemy_radius

        # Store raw enemy simply (list)
        self.num_enemies = len(enemy)

        self.initial_enemy = deepcopy(enemy)
        self.initial_enemy_values = deepcopy(values)

        self.enemy = deepcopy(self.initial_enemy)
        self.enemy_values = deepcopy(self.initial_enemy_values)

        # Step enemy to initial positions
        for i, t in zip(self.enemy_index, self.enemy):
            self.step(t, i)

        # Init Target tree (for fast lookup)
        self._kd_enemy = KDTree(np.vstack(self.enemy))

    def update_enemy(self, n):
        """Move enemies to new location"""
        # Update target location
        if n in self.enemy_index:
            i = self.enemy_index.index(n)
            self.enemy[i] = self.state[n]

        # Rebuild tree
        self._kd_enemy = KDTree(np.vstack(self.enemy))

    def check_enemy(self, n):
        """Check for enemys, and update self.reward"""

        # No enemy
        if self.enemy is None:
            return None

        # Is 'n' a predator?
        if n in self.enemy_index:
            # How far are enemys?
            state = np.atleast_2d(np.asarray(self.state[n]))
            dist, ind = self._kd_enemy.query(state, k=2)

            # Pick the closest (not itself, aka 0)
            dist = float(dist[0][1])
            ind = int(ind[0][1])
            code = self.enemy_index[ind]

            # What's the value?
            value = self.enemy_values[ind]

            # Are they in the radius?
            if code not in self.dead:
                if (0.0 < dist <= self.enemy_radius):
                    # Detection coin flip:
                    if self.np_random.rand() <= self.p_enemy:
                        # Pred value is:
                        self.reward = value
                        self.enemy_values[ind] = 0.0
                        # Death to pred, if detection
                        self.dead_enemy.append((n, ind))
                        self.dead.append(code)  # death if detection
                    else:
                        self.reward = 0.0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # Reinit
        self.n = 0
        self.state = [np.zeros(2) for _ in range(self.num_agents)]

        # Restep targets and values
        if self.initial_targets is not None:
            self.values = deepcopy(self.initial_values)
            self.targets = deepcopy(self.initial_targets)
            for i, t in zip(self.target_index, self.targets):
                self.step(t, i)

        # Restep targets and values
        if self.initial_enemy is not None:
            self.enemy_values = deepcopy(self.initial_enemy_values)
            self.enemy = deepcopy(self.initial_enemy)
            for i, t in zip(self.enemy_index, self.enemy):
                self.step(t, i)

        # Revive the dead!
        self.dead = []
        self.dead_enemy = []

        # Reset
        self.reward = 0.0
        self.last()

    def render(self, mode='human', close=False):
        pass


class CutthroatGrid(CutthroatField):
    """An open-ended grid world, with predators who attack everyone!.
    
    Params
    -----
    num_agents: int
        The total number of agents
    """
    def __init__(self, num_agents=2):
        super().__init__(num_agents=num_agents)

    def step(self, action, n):
        # Force int... so we are on a grid.
        action = [int(a) for a in action]
        super().step(action, n)

        return self.last()


# -------------------------------------------------------------------------
# Scent functions
# -------------------------------------------------------------------------


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)


def create_maze_scent(shape, amplitude=1, sigma=1):
    """Make Guassian 'scent' grid, for the MazeEnv."""

    # Grid...
    x, y = shape
    x_grid, y_grid = np.meshgrid(np.linspace(x, 0, x), np.linspace(y, 0, y))
    distance = np.sqrt(x_grid * x_grid + y_grid * y_grid)

    # Sample
    gauss = np.exp(-((distance)**2 / (2.0 * sigma**2)))
    gauss * -amplitude

    # Coords
    x_coord = x_grid[0, :]
    y_coord = y_grid[:, 0]

    return (x_coord, y_coord), gauss


def create_grid_scent(shape, amplitude=1, sigma=10):
    """Make Guassian 'scent' grid, for the MazeEnv"""
    # Grid
    x, y = shape
    x_grid, y_grid = np.meshgrid(np.linspace(-x, x, 2 * x),
                                 np.linspace(-y, y, 2 * y))
    distance = np.sqrt(x_grid * x_grid + y_grid * y_grid)

    # Sample
    gauss = np.exp(-((distance)**2 / (2.0 * sigma**2)))
    gauss * -amplitude

    # Coords
    x_coord = x_grid[0, :]
    y_coord = y_grid[:, 0]

    return (x_coord, y_coord), gauss


def create_grid_scent_patches(shape, p=0.1, amplitude=1, sigma=10):
    """Make Guassian 'scent' grid, with random holes
    for the MazeEnv"""
    # Create scent
    (x_coord, y_coord), gauss = create_grid_scent(shape,
                                                  amplitude=amplitude,
                                                  sigma=sigma)
    # Random bool mask to make patches
    # out of the scent
    sample = np.random.rand(*gauss.shape)
    gauss[sample > p] = 0.0

    return (x_coord, y_coord), gauss


def add_noise(scent, sigma=0.1, prng=None):
    """Add white noise, with variance sigma (clipped > 0)"""

    if prng is None:
        prng = np.random.RandomState()
    noise = np.abs(prng.normal(0, sigma, size=scent.shape))
    corrupt = scent + noise
    return corrupt


# -------------------------------------------------------------------------
# Targets functions
# -------------------------------------------------------------------------


def _init_prng(prng):
    if prng is None:
        return np.random.RandomState(prng)
    else:
        return prng


def uniform_targets(N, shape, prng=None):
    prng = _init_prng(prng)

    targets = []
    for s in shape:
        locs = prng.uniform(-s, s, size=N)
        targets.append(deepcopy(locs))

    # Reorg into a list of location arrays
    targets = list(zip(*targets))
    targets = [np.asarray(t) for t in targets]

    return targets


def exponential_targets(N, shape, scale=1, clip=True, prng=None):
    prng = _init_prng(prng)

    targets = []
    for s in shape:
        # Sample
        locs = prng.exponential(scale=scale, size=N)

        # Clip
        if clip:
            locs[locs > s] = s

        # Save
        targets.append(deepcopy(locs.tolist()))

    # Reorg into a list of location arrays
    targets = list(zip(*targets))
    targets = [np.asarray(t) for t in targets]

    return targets


def poisson_targets(N, shape, rate, clip=True, prng=None):
    prng = _init_prng(prng)
    scale = 1 / rate
    targets = []

    # !
    # For each dim in shape, generate a set of locations
    # using a poisson process (sum exp(1/rate))
    for s in shape:
        locs = []
        t = 0
        for _ in range(N):
            # Sample
            t += prng.exponential(scale=scale)

            # Clip
            if clip and (t > s):
                t = s

            # Save
            locs.append(deepcopy(t))

        # Save this dim/shape
        targets.append(locs)

    # Reorg into a list of location arrays
    targets = list(zip(*targets))
    targets = [np.asarray(t) for t in targets]

    return targets


def levy_dust_targets(N, shape, exponent=2, clip=True, prng=None):
    # Sanity
    if len(shape) != 2:
        raise ValueError("shape must be 2d")
    shape = np.asarray(shape)

    # Init seed
    prng = _init_prng(prng)

    # Use a levy walker to make the targets
    walker = Levy2d(exponent=exponent)
    walker.np_random = prng  # set

    # -
    targets = []
    state = np.zeros(2)
    for _ in range(N + 1):
        # Move
        action = walker(state)
        state += action

        # Clip
        if clip:
            state[state > shape] = shape

        # Convert and save
        targets.append(state.copy())

    return targets


def constant_values(targets, value=1):
    return np.asarray([value for _ in targets])


def uniform_values(targets, low=0, high=1, prng=None):
    prng = _init_prng(prng)
    return prng.uniform(low=low, high=high, size=len(targets))


def exp_values(targets, scale=1, prng=None):
    prng = _init_prng(prng)
    return prng.exponential(scale=scale, size=len(targets))


def poisson_values(targets, rate=1, prng=None):
    prng = _init_prng(prng)
    return prng.poisson(lam=rate, size=len(targets))


def gamma_values(targets, shape=1.0, scale=2.0, prng=None):
    prng = _init_prng(prng)
    return prng.gamma(shape=shape, scale=scale, size=len(targets))


def levy_values(targets, exponent=2.0, prng=None):
    prng = _init_prng(prng)
    return np.power(prng.uniform(size=len(targets)), (-1 / exponent))
