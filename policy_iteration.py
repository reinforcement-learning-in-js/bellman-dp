from ast import Lambda
from math import factorial
import numpy as np
from scipy.special import factorial

# ex 4.2. car rental example
gamma = 0.9

def poisson(n, lamb):
    return np.power(lamb, n) / factorial(n) * np.exp(-lamb)

# rend/ 
def evaluate_policy(values, gamma):
    new_values = np.zeros(len(values))
    for prev in range(len(new_values)):
        for cur in range(len(new_values)):
            min_rend = prev - cur
            max_rend = prev
            for rend in range(min_rend, max_rend):
                ret = prev - cur - rend
                gain = rend * 10 + gamma * values[prev]
                new_values[cur] += poisson(rend, 3) * poisson(ret, 3) * gain
