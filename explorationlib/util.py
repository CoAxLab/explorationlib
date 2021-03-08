import cloudpickle
import numpy as np

from collections import defaultdict


def save(log, filename='checkpoint.pkl'):
    data = cloudpickle.dumps(log)
    with open(filename, 'wb') as fi:
        fi.write(data)


def load(filename='checkpoint.pkl'):
    with open(filename, 'rb') as fi:
        return cloudpickle.load(fi)


def select_exp(exp_data, num_experiment):
    """Select all data for a single experiment `n`"""
    return exp_data[num_experiment]


def select_agent(exp_data, num_agent, num_experiment):
    """Select an agent's from a competitive experiment"""
    return exp_data[num_experiment]["agent_history"][num_agent]
