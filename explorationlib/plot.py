import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from explorationlib.util import load
from explorationlib.util import select_exp

# from celluloid import Camera

import base64
from IPython import display


def show_gif(name):
    """Show gifs, in notebooks.
    
    Code from:
    https://github.com/ipython/ipython/issues/10045#issuecomment-642640541
    """

    with open(name, 'rb') as fd:
        b64 = base64.b64encode(fd.read()).decode('ascii')

    return display.HTML(f'<img src="data:image/gif;base64,{b64}" />')


# def render_2d(name,
#               env,
#               exp_data,
#               num_experiment=0,
#               figsize=(4, 4),
#               boundary=(50, 50),
#               interval=200):
#     """Replay an experiment, as a movie.

#     NOTE: can be very slow to run for experiments
#     with more than a couple thousand steps.
#     """

#     # Init
#     fig = plt.figure(figsize=figsize)
#     camera = Camera(fig)

#     # Select data
#     sel_data = select_exp(exp_data, num_experiment)

#     # Iterate frames
#     targets = np.vstack(env.targets)
#     states = np.vstack(sel_data["exp_state"])
#     rewards = sel_data["exp_reward"]

#     for i in range(states.shape[0]):
#         # Field
#         plt.scatter(
#             targets[:, 0],
#             targets[:, 1],
#             env.values,  # value is size, literal
#             color="black",
#             alpha=1)

#         color = "black"
#         if rewards[i] > 0:
#             color = "red"  # wow!``

#         # Path
#         plt.plot(states[0:i, 0], states[0:i, 1], color=color, alpha=1)

#         # Agent
#         plt.plot(states[i, 0],
#                  states[i, 1],
#                  color=color,
#                  markersize=env.detection_radius,
#                  marker='o',
#                  alpha=1)

#         # Labels
#         plt.xlim(-boundary[0], boundary[0])
#         plt.ylim(-boundary[1], boundary[1])
#         plt.xlabel("x")
#         plt.ylabel("y")

#         # Frame
#         camera.snap()

#     # Render
#     animation = camera.animate(interval=interval)
#     animation.save(f'{name}')

#     return camera


def plot_scent_grid(env,
                    figsize=(3, 3),
                    boundary=(1, 1),
                    cmap='viridis',
                    title=None,
                    ax=None):
    # No targets no plot
    if env.scent is None:
        return None

    # Create a fig obj?
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    # !
    ax.imshow(env.scent, interpolation=None, cmap=cmap)
    ax.set_xlabel("i")
    ax.set_ylabel("j")

    # Labels, legends, titles?
    if title is not None:
        ax.set_title(title)

    return ax


def plot_targets2d(env,
                   figsize=(3, 3),
                   boundary=(1, 1),
                   color="black",
                   alpha=1.0,
                   label=None,
                   title=None,
                   ax=None):

    # No targets no plot
    if env.targets is None:
        return None

    # Fmt
    vec = np.vstack(env.targets)

    # Create a fig obj?
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    # !
    ax.scatter(
        vec[:, 0],
        vec[:, 1],
        env.values,  # value is size, literal
        color=color,
        label=label,
        alpha=alpha)
    ax.set_xlim(-boundary[0], boundary[0])
    ax.set_ylim(-boundary[1], boundary[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Labels, legends, titles?
    if title is not None:
        ax.set_title(title)
    if label is not None:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return ax


def plot_position2d(exp_data,
                    boundary=(1, 1),
                    figsize=(3, 3),
                    color="black",
                    alpha=1.0,
                    label=None,
                    title=None,
                    ax=None):
    # fmt
    var_name = "exp_state"
    state = np.vstack(exp_data[var_name])

    # Create a fig obj?
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    # !
    ax.plot(state[:, 0], state[:, 1], color=color, label=label, alpha=alpha)
    ax.set_xlim(-boundary[0], boundary[0])
    ax.set_ylim(-boundary[1], boundary[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Labels, legends, titles?
    if title is not None:
        ax.set_title(title)
    if label is not None:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return ax


def plot_length(exp_data,
                figsize=(4, 2),
                color="black",
                alpha=1.0,
                label=None,
                title=None,
                ax=None):
    # fmt
    length_name = "agent_l"
    step_name = "agent_num_turn"
    l = np.asarray(exp_data[length_name])
    step = np.asarray(exp_data[step_name])

    # Create a fig obj?
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    # !
    ax.plot(step, l, color=color, label=label, alpha=alpha)
    ax.set_xlabel("Turn count")
    ax.set_ylabel("Length")

    # Labels, legends, titles?
    if title is not None:
        ax.set_title(title)
    if label is not None:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return ax


def plot_length_hist(exp_data,
                     loglog=True,
                     bins=20,
                     figsize=(3, 3),
                     color="black",
                     alpha=1.0,
                     density=True,
                     label=None,
                     title=None,
                     ax=None):

    # fmt
    length_name = "agent_l"
    x = np.asarray(exp_data[length_name])

    # Create a fig obj?
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    if loglog:
        bins = np.geomspace(x.min(), x.max(), bins)
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.hist(x,
            bins=bins,
            color=color,
            alpha=alpha,
            density=density,
            label=label)
    ax.set_xlabel("Length")
    ax.set_ylabel("Count")

    # Labels, legends, titles?
    if title is not None:
        ax.set_title(title)
    if label is not None:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return ax