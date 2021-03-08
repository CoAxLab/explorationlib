import os
import numpy as np
import cloudpickle

from copy import deepcopy
from collections import defaultdict

from explorationlib.util import save
from tqdm.autonotebook import tqdm
from explorationlib import local_gym
from explorationlib import agent as agent_gym


def experiment(name,
               agent,
               env,
               num_steps=1,
               num_experiments=1,
               seed=None,
               split_state=False,
               dump=True,
               env_kwargs=None,
               agent_kwargs=None):
    """Run an experiment. 
    
    Note: by default the experiment log gets saved to 'name' and this
    function returns None: To return the exp_data, set dump=False.
    """

    # Parse env
    if isinstance(env, str):
        Env = getattr(local_gym, env)
        if env_kwargs is not None:
            env = Env(**env_kwargs)
        else:
            env = Env()

    # Parse agent
    if isinstance(agent, str):
        Agent = getattr(agent_gym, agent)
        if agent_kwargs is not None:
            agent = Agent(**agent_kwargs)
        else:
            agent = Agent()

    # Pretty name
    base = os.path.basename(name)
    base = os.path.splitext(base)[0]

    # Seed
    agent.seed(seed)
    env.seed(seed)

    # Add one log for each exp
    # to the results list
    results = []

    # !
    for k in tqdm(range(num_experiments), desc=base):
        # Create an exp log
        log = defaultdict(list)

        # Reset world
        agent.reset()
        env.reset()
        state, reward, done, info = env.last()

        # Run experiment, for at most num_steps
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
        # TODO - someday updatee all code to save comp and reg exps the same
        # way, like this
        # log["agent_history"] = []
        # log["agent_history"].append(agent.history)

        # Old fmt. Replace w/ above
        for k in agent.history.keys():
            log[k].extend(deepcopy(agent.history[k]))

        # Save agent and env
        log["exp_name"] = base
        log["num_experiments"] = num_experiments
        log["exp_num_steps"] = num_steps
        log["env"] = env
        log["agent"] = agent

        # Save the log to the results
        results.append(log)

    if dump:
        if not name.endswith(".pkl"):
            name += ".pkl"
        save(results, filename=name)
    else:
        return results


def multi_experiment(name,
                     agents,
                     env,
                     num_steps=1,
                     num_experiments=1,
                     seed=None,
                     split_state=False,
                     dump=True,
                     env_kwargs=None):
    """Run a multi-agent experiment. Targets can also be agents. 
    
    Note: by default the experiment log gets saved to 'name' and this
    function returns None: To return the exp_data, set dump=False.
    """

    # Parse env
    if isinstance(env, str):
        Env = getattr(local_gym, env)
        if env_kwargs is not None:
            env = Env(**env_kwargs)
        else:
            env = Env()

    # Pretty name
    base = os.path.basename(name)
    base = os.path.splitext(base)[0]

    # Seed
    if seed is not None:
        [agent.seed(seed + i) for i, agent in enumerate(agents)]
        env.seed(seed)

    # Add one log for each exp
    # to the results list
    results = []

    # !
    for k in tqdm(range(num_experiments), desc=base):
        # Create an exp log
        log = defaultdict(list)

        # Reset agents...
        [agent.reset() for agent in agents]

        # and the world
        env.reset()
        state, reward, done, info = env.last()

        # Run experiment, for at most num_steps
        for n in range(1, num_steps):
            for i, agent in enumerate(agents):
                # The dead don't step
                if i in env.dead:
                    continue

                # Step the agent
                action = agent(state[i])
                state, reward, done, info = env.step(action, i)

                # Learn? Might do nothing.
                agent.update(state[i], reward, info)

                # Log step env
                log["num_experiment"].append(deepcopy(k))
                log["exp_step"].append(deepcopy(n))
                log["exp_agent"].append(deepcopy(i))
                if split_state:
                    pos, obs = state
                    log["exp_state"].append(deepcopy(pos))
                    log["exp_obs"].append(deepcopy(obs))
                else:
                    log["exp_state"].append(deepcopy(state))
                log["exp_action"].append(deepcopy(action))
                log["exp_reward"].append(deepcopy(reward))
                log["exp_info"].append(deepcopy(info))

                # ?
                if done:
                    break

        # Log agents history
        log["agent_history"] = []
        for agent in agents:
            log["agent_history"].append(agent.history)

        # Save agent and env
        log["exp_name"] = base
        log["num_experiments"] = num_experiments
        log["exp_num_steps"] = num_steps
        log["env"] = env
        log["agent"] = agent

        # Save the log to the results
        results.append(log)

    if dump:
        if not name.endswith(".pkl"):
            name += ".pkl"
        save(results, filename=name)
    else:
        return results


if __name__ == "__main__":
    import fire
    fire.Fire({"experiment": experiment})
