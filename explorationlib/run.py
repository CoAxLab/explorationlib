import os
import numpy as np
import cloudpickle

from copy import deepcopy
from collections import defaultdict

from explorationlib.util import save
from tqdm.autonotebook import tqdm


def experiment(name,
               agent,
               env,
               num_steps=1,
               num_experiments=1,
               seed=None,
               split_state=False,
               dump=True):
    """Run an experiment. 
    
    Note: by default the experiment log gets saved to 'name' and this
    function returns None: To return the exp_data, set dump=False.
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
            log["exp_step"].append(deepcopy(n))
            log["num_experiment"].append(deepcopy(k))
            if split_state:
                pos, obs = state
                log["exp_state"].append(deepcopy(pos))
                log["exp_obs"].append(deepcopy(obs))
            else:
                log["exp_state"].append(deepcopy(state))
            log["exp_action"].append(deepcopy(action))
            log["exp_reward"].append(deepcopy(reward))
            log["exp_info"].append(deepcopy(info))

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

    if dump:
        save(log, filename=name)
    else:
        return log


def multi_experiment(name, agents, env, num_episodes=1, seed=None):
    """Run an experiment, with multiple agents. 
    
    Note: the experiment log gets saved to 'name'. 
    """

    raise NotImplementedError("TODO")