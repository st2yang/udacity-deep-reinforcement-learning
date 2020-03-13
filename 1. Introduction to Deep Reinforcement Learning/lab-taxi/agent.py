import numpy as np
import random
from collections import defaultdict


class Agent:

    def __init__(self, nA=6, gamma=0.9, alpha=0.1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.gamma = gamma
        self.alpha = alpha
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state, eps):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > eps:
            return np.argmax(self.Q[state])
        else:
            return random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        current = self.Q[state][action]
        Qsa_next = np.max(self.Q[next_state]) if next_state is not None else 0
        target = reward + self.gamma * Qsa_next
        new_value = current + self.alpha * (target - current)
        self.Q[state][action] = new_value
