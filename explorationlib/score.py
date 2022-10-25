import numpy as np
from scipy.stats import entropy
from tqdm.autonotebook import tqdm

from explorationlib.util import load
from explorationlib.util import select_exp


def bandit_rmse(exp_data):
    # Load?
    if isinstance(exp_data, str):
        exp_data = load(exp_data)

    errors = []
    for log in tqdm(exp_data, desc="bandit_rmse"):
        # Predicted
        env = log["exp_env"]
        probs = np.asarray(env.p_dist)
        values = np.asarray(env.r_dist)
        predictions = probs * values

        # Target
        agent = log["exp_agent"]
        targets = list(agent.critic.model.values())
        targets = np.asarray(targets)
        rmse = np.sqrt(((predictions - targets)**2).mean())

        errors.append(rmse)

    return errors


def action_entropy(exp_data, base=None):
    """Entropy of the agent's actions"""

    # Load?
    if isinstance(exp_data, str):
        exp_data = load(exp_data)

    target_name = "exp_action"
    totals = []
    for log in tqdm(exp_data, desc="action_entropy"):
        actions = np.asarray(log[target_name])
        _, counts = np.unique(actions, return_counts=True)
        ent = entropy(counts, base=base)

        totals.append(ent)

    return totals


def total_info_value(exp_data):
    """Total value learning"""

    # Load?
    if isinstance(exp_data, str):
        exp_data = load(exp_data)

    # !
    target_name = "agent_info_value"

    totals = []
    for log in tqdm(exp_data, desc="info_value"):
        infovalues = log[target_name]
        totals.append(np.sum(infovalues))

    return totals


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
    """Number of steps to first reward"""

    # Load?
    if isinstance(exp_data, str):
        exp_data = load(exp_data)

    # !
    target_name = "exp_reward"
    step_name = "exp_step"

    # Get firsts
    firsts = []
    for log in tqdm(exp_data, desc="total_reward"):
        rewards = log[target_name]
        steps = log[step_name]
        t = 0
        for r, s in zip(rewards, steps):
            if r > 0:
                t = s
                break
        firsts.append(t)

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


def on_off_patch_time(exp_data, num_agents, patch_locs, patch_radius):
    """Time steps within vs between patches for multiple runs"""
    
    # Load?
    if isinstance(exp_data, str):
        exp_data = load(exp_data)
    
    on_patch_steps  = []
    off_patch_steps = []
    
    var_name="exp_state"
    
    for log in tqdm(exp_data, desc="on_off_patch_time"):
        
        on_patch_step  = 0
        off_patch_step = 0
    
        # fmt
        states_vec = log[var_name]
        states = [list() for _ in range(num_agents)]
    
        # repack
        states = np.array(states_vec)
        
        xs = states[:, 0]
        ys = states[:, 1]
        
        for x, y in zip(xs, ys):
            for patch_loc in patch_locs:
                if (x - patch_loc[0])**2 + (y - patch_loc[1])**2 < patch_radius:
                    on_patch_steps += 1
                else:
                    off_patch_step += 1
    
        on_patch_steps.append(on_patch_step)
        off_patch_steps.append(off_patch_step)
            
    return on_patch_steps, off_patch_steps


if __name__ == "__main__":
    import fire
    fire.Fire({
        "average_reward": average_reward,
        "total_reward": total_reward,
        "first_reward": first_reward,
        "search_efficiency": search_efficiency
    })
