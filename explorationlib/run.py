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
        for k in agent.history.keys():
            log[k].extend(deepcopy(agent.history[k]))

        # Save agent and env
        log["exp_name"] = base
        log["num_experiments"] = num_experiments
        log["exp_num_steps"] = num_steps
        log["env"] = env.reset()
        log["agent"] = agent.reset()

        # Save the log to the results
        results.append(log)

    if dump:
        if not name.endswith(".pkl"):
            name += ".pkl"
        save(results, filename=name)
    else:
        return results


def competitive_experiment(name,
                           preds,
                           preys,
                           pred_states,
                           prey_states,
                           env,
                           num_steps=1,
                           num_experiments=1,
                           seed=None,
                           split_state=False,
                           dump=True,
                           env_kwargs=None):
    """Run an experiment, with multiple agents. 
    
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

    # Seed?
    if seed is not None:
        # Seed agents w/ unique
        for i in range(preds):
            preds[i].seed(seed + i)
        for i in range(preys):
            preys[i].seed(seed + i)
        # Seed world
        env.seed(seed)

    # Add one log for each exp
    results = []

    # -----------------------------------------------------------------------
    # !
    for k in tqdm(range(num_experiments), desc=base):
        # Create an exp log
        log = defaultdict(list)

        # Reset players
        [agent.reset() for agent in preds]
        [agent.reset() for agent in preys]

        # Init locations
        for agent, state in zip(preys, prey_states):
            agent.state = state
        for agent, state in zip(preys, pred_states):
            agent.state = state

        # Reset world
        env.reset()
        state, reward, done, info = env.last()

        # Run experiment, for at most num_steps
        for n in range(1, num_steps):
            # ---------------------------------------------------------------
            # Step prey
            targets, values = [], []
            for i, agent in enumerate(preys):
                # !
                action = agent(state)
                env.step(action)
                state, reward, done, info = env.last()

                # Add new location
                if split_state:
                    targets.append(state[0])
                    values.append(agent.value)
                else:
                    targets.append(state)
                    values.append(agent.value)

                # Learn? Might do nothing.
                agent.update(state, reward, info)

                # Log agent step
                log["exp_prey_code"].append(deepcopy(i))
                log["exp_pred_code"].append(None)

                # Log exp
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

                # Force end?
                if done:
                    break

            # Update targets and values (aka prey)
            env.add_targets(targets,
                            values,
                            detection_radius=env.detection_radius,
                            kd_kwargs=env.kd_kwargs,
                            p_target=env.p_target)

            # ---------------------------------------------------------------
            # Step preds
            for i, agent in enumerate(preds):
                # !
                action = agent(state)
                env.step(action)
                state, reward, done, info = env.last()

                # Learn? Might do nothing.
                agent.update(state, reward, info)

                # Log agent
                log["exp_prey_code"].append(None)
                log["exp_pred_code"].append(deepcopy(i))

                # Log exp
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

        # -------------------------------------------------------------------
        # Log full agent history
        for k in agent.history.keys():
            log[k].extend(deepcopy(agent.history[k]))

        # Save agent and env
        log["multi_exp"] = True  # Tag for analysis
        log["exp_name"] = base
        log["num_experiments"] = num_experiments
        log["exp_num_steps"] = num_steps
        log["env"] = env.reset()
        log["preds"] = [agent.reset() for agent in preds]
        log["preys"] = [agent.reset() for agent in preys]

        # Save the log to the results
        results.append(log)

    # -----------------------------------------------------------------------
    if dump:
        if not name.endswith(".pkl"):
            name += ".pkl"
        save(results, filename=name)
    else:
        return results


if __name__ == "__main__":
    import fire
    fire.Fire({"experiment": experiment})
