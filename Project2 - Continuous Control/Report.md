# Project 2: Continuous Control

In this project, we exploit [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971) to continuously control a double-jointed arm.
A reward of +0.1 is provided for each step if the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Methods
DDPG is closely connected to Q-learning and specifically adapted for environments with continuous action spaces. 
In DDPG, an actor `μ(s)` is used to propose the optimal action under a given state, and a critic `Q(s, a)` evaluates the state-action pair to approximate optimal Q value.
Thus DDPG concurrently learns a Q-function and a policy using off-policy data.
 
In DDPG, we train the critic by minimizing the temporal difference (TD) error δ:

<img src="./assets/equation1.png" width=500>

and train the actor by maximizing expected Q values:

<img src="./assets/equation2.png" width=250>

## Models and Hyperparameters
Both actor and critic are three-layer fully connected networks, with RELU as activation function. The hyperparameters used in training are as follows

| Hyperparameter                      | Value |
| ----------------------------------- | ----- |
| BUFFER_SIZE                         | 1e5   |
| BATCH_SIZE                          | 128   |
| GAMMA                               | 0.99  |
| TAU                                 | 1e-3  |
| LR_Actor                            | 1e-4  |
| LR_CRITIC                           | 1e-4  |


## Results
We solve the task in 100 episodes. A plot of score (the average reward over 100 episodes) is included below. 

<p align="center">
    <img src="./assets/ScorePlot.png" width=500 alt="score">
</p>

## Future Work
I am interested in the following two directions for future improvements

- Research and compare several policy-based methods, such as [PPO](https://arxiv.org/abs/1707.06347), [TD3](https://arxiv.org/abs/1802.09477), [SAC](https://arxiv.org/abs/1801.01290). It would be interesting to know the performance gaps and advantages of these methods.

- Research effective ways of building models and tuning parameters. 
It seems that policy-based methods (and deep reinforcement learning) are much more sensitive to initialization, normalization and tuning compared to supervised learning.
