#! /usr/bin/env python
import numpy as np

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

# -------------------------------------------------------------------------
# Enviroments
# -------------------------------------------------------------------------


class ScentMazeEnv(MazeEnv):
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

    def add_scent(self, scent):
        self.scent = scent

    def step(self, action):
        # Maze step
        self.position, self.reward, self.done, self.info = super().step(action)
        # Scent?
        if self.scent is not None:
            x, y = int(self.position[0]), int(self.position[1])
            scent = self.scent[x, y]
        else:
            scent = 0.0
        # !
        self.state = (self.position, scent)
        return (self.state, self.reward, self.done, self.info)

    def last(self):
        return (self.state, self.reward, self.done, self.info)


class Field(gym.Env):
    """An open-field to explore, with no boundries."""
    def __init__(self):
        self.info = {}
        self.reward = 0
        self.done = False

        self.detection_radius = None
        self.targets = None
        self.values = None

        self.reset()

    def step(self, action):
        self.state += action
        self.check_targets()
        return self.last()

    def last(self):
        """Return the last transition: (state, reward, done, info)
        """
        return (self.state, self.reward, self.done, self.info)

    def add_targets(self, targets, values, detection_radius=1, kd_kwargs=None):
        """Add targets and their values"""

        # Sanity
        if len(targets) != len(values):
            raise ValueError("targets and values must match.")

        # Store raw targets simply (list)
        self.targets = targets
        self.values = values
        self.detection_radius = detection_radius

        # Also store targets so lookup is efficient (tree)
        if kd_kwargs is None:
            kd_kwargs = {}
        self._kd = KDTree(np.vstack(self.targets), **kd_kwargs)

    def check_targets(self):
        """Check for targets, and update self.reward if
        some are found in the given detection_radius.

        Note: the deault d_func is the euclidian distance. 
        To override provide a func(x, y) -> distance.
        """
        # Short circuit if no targets
        if self.targets is None:
            return None

        # Reinit reward. Assume we are not at a target
        self.reward = 0

        # How far are we and is it close enough to
        # generate a reward? AKA are we at a target?
        state = np.atleast_2d(np.asarray(self.state))
        dist, ind = self._kd.query(state, k=1)

        # Care about the closest; Fmt
        dist = float(dist[0])
        ind = int(ind[0])

        # Test proximity
        if dist <= self.detection_radius:
            self.reward = self.values[ind]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.zeros(2)
        self.reward = 0
        self.last()

    def render(self, mode='human', close=False):
        pass


class ViswanathanField5000(Field):
    def __init__(self):
        super().__init__()
        targets, values = self._create_targets(5000)
        super().add_targets(targets, values, detection_radius=1)

    def _create_targets(self, num_targets):
        target_boundary = (100, 100)
        targets = uniform_targets(num_targets, target_boundary)
        values = constant_values(targets, 1)
        return targets, values


class ViswanathanField50000(Field):
    def __init__(self):
        super().__init__()
        targets, values = self._create_targets(50000)
        super().add_targets(targets, values, detection_radius=1)

    def _create_targets(self, num_targets):
        target_boundary = (100, 100)
        targets = uniform_targets(num_targets, target_boundary)
        values = constant_values(targets, 1)
        return targets, values


class ViswanathanField500000(Field):
    def __init__(self):
        super().__init__()
        targets, values = self._create_targets(500000)
        super().add_targets(targets, values, detection_radius=1)

    def _create_targets(self, num_targets):
        target_boundary = (100, 100)
        targets = uniform_targets(num_targets, target_boundary)
        values = constant_values(targets, 1)
        return targets, values


# -------------------------------------------------------------------------
class Grid(Field):
    """A discrete open-ended grid-world."""
    def __init__(self, mode="cardinal"):
        super().__init__()
        self.mode = mode
        self.scent = None

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
        if self.mode == "cardinal":
            self._card_step(action)
        return self.last()

    def reset(self):
        self.state = np.zeros(2, dtype=int)
        self.last()


class ScentGrid(Grid):
    """Am open-grid, with scent"""
    def __init__(self, mode="cardinal"):
        super().__init__(mode=mode)
        self.scent_fn = None
        self.obs = 0.0

    def add_scent(self, target, value, coord, scent, detection_radius=1):
        self.scent_x_coord = coord[0] + target[0]
        self.scent_y_coord = coord[1] + target[1]
        self.scent_pdf = scent
        self.add_targets([target], [value],
                         detection_radius=detection_radius,
                         kd_kwargs=None)

        def scent_fn(state):
            x, y = state
            i = find_nearest(self.scent_x_coord, x)
            j = find_nearest(self.scent_y_coord, y)
            return self.scent_pdf[i, j]

        self.scent_fn = scent_fn

    def step(self, action):
        # Move
        super().step(action)

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


# -------------------------------------------------------------------------
# Scents
# -------------------------------------------------------------------------


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)


def create_maze_scent(shape, amplitude=1, sigma=1):
    """Make Guassian 'scent' grid, for the MazeEnv"""
    # Grid
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


def add_noise(scent, sigma=0.1, prng=None):
    if prng is None:
        prng = np.random.RandomState()
    noise = prng.normal(0, sigma, size=scent.shape)
    corrupt = scent + noise
    corrupt = np.clip(corrupt, 0, corrupt.max())
    return corrupt


# -------------------------------------------------------------------------
# Targets
# -------------------------------------------------------------------------


def _init_prng(prng):
    if prng is None:
        return np.random.RandomState(prng)


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
