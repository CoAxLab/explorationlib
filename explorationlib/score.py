import numpy as np
from tqdm.autonotebook import tqdm

from explorationlib.util import load
from explorationlib.util import select_exp


def average_reward(exp_data):
    """Average targets found"""

    # Load?
    if isinstance(exp_data, str):
        exp_data = load(exp_data)

    target_name = "exp_reward"
    averages = []
    for log in tqdm(exp_data, desc="average_reward"):
        rewards = log[target_name]
        averages.append(np.mean(rewards))

    return averages


def total_reward(exp_data):
    """Total targets found"""

    # Load?
    if isinstance(exp_data, str):
        exp_data = load(exp_data)

    # !
    target_name = "exp_reward"

    totals = []
    for log in tqdm(exp_data, desc="total_reward"):
        rewards = log[target_name]
        # Get stat
        totals.append(np.sum(rewards))

    return totals


def num_death(exp_data):
    """Total number of 'deaths' (aka no targets found)"""
    # Load?
    if isinstance(exp_data, str):
        exp_data = load(exp_data)

    # !
    target_name = "exp_reward"

    # Get totals
    totals = []
    for log in tqdm(exp_data, desc="num_death"):
        rewards = log[target_name]
        # Get stat
        totals.append(np.sum(rewards))

    # Find num of zeros -> deaths
    death = len(totals) - np.count_nonzero(totals)

    return death


def first_reward(exp_data):
    """Find the inverse time to first target.
    
    Citation
    --------
    Bénichou, O., Loverdo, C., Moreau, M. & Voituriez, R. Intermittent 
    search strategies. Rev. Mod. Phys. 83, 81–129 (2011).
    """

    # Load?
    if isinstance(exp_data, str):
        exp_data = load(exp_data)

    # !
    length_name = "agent_step"
    target_name = "exp_reward"

    firsts = []
    for log in tqdm(exp_data, desc="first_reward"):
        # Get
        rewards = log[target_name]
        steps = log[length_name]

        # Short circuit if no targets...
        if np.isclose(np.sum(rewards), 0.0):
            firsts.append(0.0)
            continue

        # Time it takes to find the
        # first target/reward
        time = 1
        for r, l in zip(rewards, steps):
            if np.isclose(r, 0.0):
                time += l
            else:
                firsts.append(1 / time)
                break

        return firsts


def search_efficiency(exp_data):
    """Search efficiency, for each experiment.

    Citation
    --------
    Viswanathan, G. M. et al. Optimizing the success of random searches. Nature 401, 911–914 (1999).
    """

    # Load?
    if isinstance(exp_data, str):
        exp_data = load(exp_data)

    # !
    length_name = "agent_step"
    target_name = "exp_reward"

    effs = []
    for log in tqdm(exp_data, desc="search_efficiency"):
        # Get
        rewards = log[target_name]
        steps = log[length_name]

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


if __name__ == "__main__":
    import fire
    fire.Fire({
        "average_reward": average_reward,
        "total_reward": total_reward,
        "first_reward": first_reward,
        "search_efficiency": search_efficiency
    })
