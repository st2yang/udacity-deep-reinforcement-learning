# Project 1: Navigation

In this project, we exploit [deep Q-learning (DQN)](https://www.nature.com/articles/nature14236) to navigate an agent in a large, square world. 
The goal of the agent is to collect as many yellow bananas (reward: +1) as possible while avoiding blue bananas (reward: -1).
The state space has 37 dimensions, with which the agent has to learn how to best select one out of four discrete actions.

## Methods
The main idea of Q-learning is to learn action-value function `Q(s, a)` that represents the expected return under some policy. Every Q function for policy π obeys the Bellman equation:

<img src="./assets/equation1.png" width=300>

We train the agent by minimizing the temporal difference (TD) error δ:

<img src="./assets/equation2.png" width=400>

where <img src="https://render.githubusercontent.com/render/math?math=\hat Q"> is function approximator 
with policy networks paramters <img src="https://render.githubusercontent.com/render/math?math=\theta"> or 
target networks paramters <img src="https://render.githubusercontent.com/render/math?math=\theta^{-}">.

DQN method is known to overestimate action values due to `max` operator. [Double DQN](https://arxiv.org/abs/1509.06461) method is thus proposed to alleviate the issue. The idea of Double Q-learning is to decompose the max operation in the target into action
selection and action evaluation. And the TD error for Double DQN is

<img src="./assets/equation3.png" width=500>

To minimise this error, we use the [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) and optimize it with stochastic gradient descent (SGD). The Q function is represented by three fully
connected layers with ReLU.


## Results
We solve the task in 415 episodes and achieve a best average score of 17.17 within 1800 episodes. A plot of score (the average reward over 100 episodes) is included below. 

<p align="center">
    <img src="./assets/ScorePlot.png" width=500 alt="score">
</p>

## Future Work
There are two directions for improvement:

- [Dueling DQN](https://arxiv.org/abs/1511.06581): A dueling network is proposed to represent two separate estimators: one for the state value function and one for
the state-dependent action advantage function. Results show that this architecture leads to better policy evaluation in the presence of many similar-valued actions.

- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952): PER is a framework for prioritizing experience,
so as to replay important transitions more frequently, and therefore learn more
efficiently
