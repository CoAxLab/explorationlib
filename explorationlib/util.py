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

    # Get index for n
    var_name = "num_experiment"
    mask = np.asarray(exp_data[var_name]) == num_experiment
    mask = mask.tolist()

    # Don't mask
    exclude = ["agent", "env", "exp_name", "num_experiments", "exp_num_steps"]

    # Select n
    selected = defaultdict(list)
    for k in exp_data.keys():
        if k in exclude:
            continue
        for i, m in enumerate(mask):
            if m:
                selected[k].append(exp_data[k][i])

    # Copy the excluded (in full)
    for k in exclude:
        selected[k] = exp_data[k]

    return selected
