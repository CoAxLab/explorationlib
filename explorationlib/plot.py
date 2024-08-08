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


def plot_bandit(env,
                figsize=(3, 3),
                color="black",
                alpha=0.6,
                title=None,
                ax=None):
    # Create a fig obj?
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    # Fmt
    probs = np.asarray(env.p_dist)
    values = np.asarray(env.r_dist)
    expected_value = probs * values
    names = [str(x) for x in range(env.n_bandits)]

    # !
    plt.bar(names, expected_value, color=color, alpha=alpha)
    plt.xlabel("Arm")
    plt.ylabel("Expected value")
    plt.tight_layout()
    sns.despine()

    # titles?
    if title is not None:
        ax.set_title(title)

    return ax


def plot_bandit_critic(critic,
                       figsize=(3, 3),
                       color="black",
                       alpha=0.6,
                       title=None,
                       ax=None):
    # Create a fig obj?
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    # Fmt
    values = critic.model.values()
    names = list(range(critic.num_inputs))

    # !
    plt.bar(names, values, color=color, alpha=alpha)
    plt.xlabel("Arm")
    plt.ylabel("Learned value")
    plt.tight_layout()
    sns.despine()

    # titles?
    if title is not None:
        ax.set_title(title)

    return ax


def plot_bandit_actions(exp_data,
                        num_arms=4,
                        max_steps=None,
                        figsize=(3, 3),
                        s=1,
                        color="black",
                        alpha=1.0,
                        label=None,
                        title=None,
                        ax=None):
    # fmt
    actions = np.asarray(exp_data["exp_action"])
    steps = np.asarray(exp_data["exp_step"])
    if max_steps is not None:
        mask = steps <= max_steps
        actions = actions[mask]
        steps = steps[mask]

    # Create a fig obj?
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    # !
    ax.scatter(steps, actions, s=s, color=color, label=label, alpha=alpha)
    ax.set_xlabel("Step")
    ax.set_ylabel("Arm")
    ax.set_ylim((0, num_arms))
    ax.set_yticks(list(range(0, num_arms)))
    sns.despine()

    # Labels, legends, titles?
    if title is not None:
        ax.set_title(title)
    if label is not None:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return ax


def plot_bandit_hist(exp_data,
                     max_steps=None,
                     figsize=(3, 3),
                     bins=None,
                     density=True,
                     color="black",
                     alpha=1.0,
                     label=None,
                     title=None,
                     ax=None):
    # fmt
    actions = np.asarray(exp_data["exp_action"])
    steps = np.asarray(exp_data["exp_step"])
    if max_steps is not None:
        mask = steps <= max_steps
        actions = actions[mask]
        steps = steps[mask]

    # Create a fig obj?
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    ax.hist(actions,
            bins=bins,
            color=color,
            alpha=alpha,
            density=density,
            label=label)
    ax.set_xlabel("Arm")
    ax.set_ylabel("Count")
    sns.despine()

    # Labels, legends, titles?
    if title is not None:
        ax.set_title(title)
    if label is not None:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return ax


def plot_scent_grid(env,
                    figsize=(3, 3),
                    boundary=(1, 1),
                    cmap='viridis',
                    title=None,
                    ax=None):
    # No targets no plot
    if env.scent_fn is None:
        return None

    # Create a fig obj?
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    # !
    ax.imshow(env.scent_pdf, interpolation=None, cmap=cmap)
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
    try:
        vec = np.vstack(env.initial_targets)
    except AttributeError:
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
                    competitive=True,
                    var_name="exp_state",
                    ax=None):
    # fmt
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


def plot_positions2d(exp_data,
                     num_agents,
                     boundary=(1, 1),
                     figsize=(3, 3),
                     colors=None,
                     alpha=1.0,
                     labels=None,
                     title=None,
                     competitive=True,
                     var_name="exp_state",
                     ax=None):
    # fmt
    states_vec = exp_data[var_name]
    states = [list() for _ in range(num_agents)]

    # defaults
    if colors is None:
        colors = [None for _ in range(num_agents)]
    if labels is None:
        labels = [None for _ in range(num_agents)]

    # repack
    for s in states_vec:
        for n in range(num_agents):
            states[n].append(s[n])
    states = [np.vstack(state) for state in states]

    # Create a fig obj?
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    # !
    for i, state in enumerate(states):
        ax.plot(state[:, 0],
                state[:, 1],
                color=colors[i],
                label=labels[i],
                alpha=alpha)
    ax.set_xlim(-boundary[0], boundary[0])
    ax.set_ylim(-boundary[1], boundary[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Labels, legends, titles?
    if title is not None:
        ax.set_title(title)
    if labels[0] is not None:
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


def plot_angle(exp_data,
               figsize=(4, 2),
               color="black",
               alpha=1.0,
               label=None,
               title=None,
               ax=None):
    # fmt
    length_name = "agent_angle"
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


# def plot_length_hist(exp_data,
#                      loglog=True,
#                      bins=20,
#                      figsize=(3, 3),
#                      color="black",
#                      alpha=1.0,
#                      density=True,
#                      label=None,
#                      title=None,
#                      ax=None):

#     # fmt
#     length_name = "agent_num_step"
#     x = np.asarray(exp_data[length_name])

#     # Create a fig obj?
#     if ax is None:
#         fig = plt.figure(figsize=figsize)
#         ax = fig.add_subplot(111)

#     if loglog:
#         bins = np.geomspace(x.min(), x.max(), bins)
#         ax.set_xscale('log')
#         ax.set_yscale('log')

#     ax.hist(x,
#             bins=bins,
#             color=color,
#             alpha=alpha,
#             density=density,
#             label=label)
#     ax.set_xlabel("Length")
#     ax.set_ylabel("Count")

#     # Labels, legends, titles?
#     if title is not None:
#         ax.set_title(title)
#     if label is not None:
#         ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#     return ax
    
def plot_length_hist(exp_data,
                     loglog=False,
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

    if density:
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("Count")

    # Labels, legends, titles?
    if title is not None:
        ax.set_title(title)
    if label is not None:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return ax

def plot_angle_hist(exp_data,
                     bins=20,
                     figsize=(3, 3),
                     color="black",
                     alpha=1.0,
                     density=True,
                     label=None,
                     title=None,
                     ax=None):

    # fmt
    length_name = "agent_angle"
    step_name = "agent_num_turn"
    l = np.asarray(exp_data[length_name])
    step = np.asarray(exp_data[step_name])

    # Create a fig obj?
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    ax.hist(l,
            bins=bins,
            color=color,
            alpha=alpha,
            density=density,
            label=label)
    ax.set_xlabel("Angle")
    ax.set_ylabel("Count")

    # Labels, legends, titles?
    if title is not None:
        ax.set_title(title)
    if label is not None:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return ax
