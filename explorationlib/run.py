import os
import numpy as np
import cloudpickle

from copy import deepcopy
from collections import defaultdict

from explorationlib.util import save
from tqdm.autonotebook import tqdm


def experiment(name, agent, env, num_steps=1, num_experiments=1, seed=None):
    """Run an experiment. 
    
    Note: the experiment log gets saved to 'name'. 
    """

    # Create a log
    log = defaultdict(list)
    base = os.path.basename(name)
    base = os.path.splitext(base)[0]

    # Seed
    agent.seed(seed)
    env.seed(seed)

    # !
    for k in tqdm(range(num_experiments), desc=base):
        # Reset
        agent.reset()
        env.reset()
        state, reward, done, info = env.last()

        # Run episode, for at most num_steps
        for n in range(1, num_steps):
            # Step
            action = agent(state)
            env.step(action)
            state, reward, done, info = env.last()

            # Learn? Might do nothing.
            agent.update(state, reward, info)

            # Log step env
            log["exp_step"].append(n)
            log["num_experiment"].append(k)
            log["exp_state"].append(state.copy())
            log["exp_action"].append(action.copy())
            log["exp_reward"].append(reward)
            log["exp_info"].append(info)

            if done:
                break

        # Log full agent history
        for k in agent.history.keys():
            log[k].extend(deepcopy(agent.history[k]))

    # Save agent and env
    log["exp_name"] = base
    log["num_experiments"] = num_experiments
    log["exp_num_steps"] = num_steps
    log["env"] = env.reset()
    log["agent"] = agent.reset()

    save(log, filename=name)


def multi_experiment(name, agents, env, num_episodes=1, seed=None):
    """Run an experiment, with multiple agents. 
    
    Note: the experiment log gets saved to 'name'. 
    """

    raise NotImplementedError("TODO")