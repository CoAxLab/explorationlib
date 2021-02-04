import numpy as np
from scipy import stats
from explorationlib.util import select_exp


def average_reward(exp_data):
    """Average targets found"""

    target_name = "exp_reward"

    averages = []
    for n in range(exp_data["num_experiments"]):
        # Get one experiments data
        sel_data = select_exp(exp_data, n)
        rewards = sel_data[target_name]
        # Get stat
        averages.append(np.mean(rewards))

    return averages


def total_reward(exp_data):
    """Total targets found"""

    target_name = "exp_reward"

    totals = []
    for n in range(exp_data["num_experiments"]):
        # Get one experiments data
        sel_data = select_exp(exp_data, n)
        rewards = sel_data[target_name]
        # Get stat
        totals.append(np.sum(rewards))

    return totals


def first_reward(exp_data):
    """Find the inverse time to first target.
    
    Citation
    --------
    Bénichou, O., Loverdo, C., Moreau, M. & Voituriez, R. Intermittent 
    search strategies. Rev. Mod. Phys. 83, 81–129 (2011).
    """

    length_name = "agent_step"
    target_name = "exp_reward"

    firsts = []
    for n in range(exp_data["num_experiments"]):
        # Get one experiments data
        sel_data = select_exp(exp_data, n)
        rewards = sel_data[target_name]
        steps = sel_data[length_name]

        # Short circuit if no targets...
        if np.isclose(np.sum(rewards), 0.0):
            firsts.append(0.0)
            continue

        # Time it takes to find the
        # first target/reward
        time = 1
        for r, l in zip(rewards, steps):
            if np.abs(r) > 0:
                firsts.append(1 / time)
            else:
                time += l

        return firsts


def search_efficiency(exp_data):
    """Search efficiency, for each experiment.

    Citation
    --------
    Viswanathan, G. M. et al. Optimizing the success of random searches. Nature 401, 911–914 (1999).
    """

    # Fmt
    length_name = "agent_step"
    target_name = "exp_reward"

    effs = []
    for n in range(exp_data["num_experiments"]):
        sel_data = select_exp(exp_data, n)

        rewards = sel_data[target_name]
        steps = sel_data[length_name]

        # Short circuit if no targets...
        if np.isclose(np.sum(rewards), 0.0):
            effs.append(0.0)
            continue

        # Other wise count targets and total
        # path length travelled to find them
        total_l = 0
        total_N = 0
        for r, l in zip(rewards, steps):
            # Not a target?
            if np.isclose(r, 0.0):
                total_l += l
            # Target!
            else:
                total_l += l
                total_N += 1

        effs.append(total_N / total_l)

    return effs