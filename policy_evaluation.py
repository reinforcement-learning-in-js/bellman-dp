import numpy as np

initial = np.zeros(16)
gamma = 1

# ex 4.1. grid transfer example
# since the example on the book is not strictly MDP, it is slightly modified.
def cell_transition(state):
    result = []
    if state == 0:
        return result
    if state == 15:
        return result
    if state > 4:
        result.append(state-4)
    if state % 4 != 3:
        result.append(state+1)
    if state % 4 != 0:
        result.append(state-1)
    if state < 12:
        result.append(state+4)
    return result

def cell_reward(state):
    if state == 0 or state == 15:
        return 0
    else:
        return -1

cell_transition_dict = {}
for i in range(16):
    cell_transition_dict[i] = cell_transition(i)

def greedy_policy(gains):
    return np.amax(gains)

def random_policy(gains):
    return np.average(gains)

def policy_evaluation(values, transition, reward):
    new_values = np.zeros(len(values))
    bellman_rec = lambda x: reward(x) + gamma*values[x]
    for state in range(len(values)):
        if len(transition[state]) == 0:
            new_values[state] = values[state]
        else:
            available_values = bellman_rec(transition[state])
            new_values[state] = random_policy(available_values)
    return new_values

values = initial
for i in range(30):
    values = policy_evaluation(values, cell_transition_dict, cell_reward)
    print(values)
