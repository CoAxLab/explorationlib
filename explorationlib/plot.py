import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from explorationlib.util import load
from explorationlib.util import select_exp

from celluloid import Camera


def render_exp2d(name,
                 env,
                 exp_data,
                 experiment=0,
                 figsize=(3, 3),
                 boundary=(1, 1)):
    """Replay an experiment as a .gif"""

    # Init the gif/fig
    fig = plt.figure(figsize=figsize)
    camera = Camera(fig)

    # Select the experiment's data
    sel_data = select_exp(exp_data, experiment)

    # Plot targets
    ax = plt.subplot(311)
    vec = np.vstack(env.targets)
    ax.scatter(
        vec[:, 0],
        vec[:, 1],
        env.values,  # value is size, literal
        color="black",
        alpha=1)
    ax.set_xlim(-boundary[0], boundary[0])
    ax.set_ylim(-boundary[1], boundary[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    camera.snap()

    # Plot the search
    d = env.detection_radius
    states = sel_data["exp_state"]
    rewards = sel_data["exp_reward"]
    lengths = sel_data["agent_l"]
    for s, r, l in zip(states, rewards, lengths):
        ax.plot(s[0], s[1], color="purple", alpha=0.6)
        camera.snap()

    # Render the movie
    animation = camera.animate()
    animation.save(f'{name}.gif')


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
    ax.set_xlabel("Turn")
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