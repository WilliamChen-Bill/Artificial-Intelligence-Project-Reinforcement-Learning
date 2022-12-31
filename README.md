# Artificial Intelligence Project: Reinforcement Learning
## 1 Getting Started
* Download the starter code from canvas. It consists of two files: Q-Learning.py and tests.py. You cancreate and activate your virtual environment with the following commands:
```bash
python3 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
```
* Once you have sourced your environment, you can run the following commands to install the necessary dependencies:
```bash
pip install --upgrade pip
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
pip install gym==0.23.1 pygame==2.1.2
```
* You should now have a virtual environment which is fully compatible with the skeleton code. You shouldset up this virtual environment on an instructional machine to do your final testing.
## Q-Learning
* For the Q-learning portion of HW10, we will be using the environment FrozenLake-v1 from OpenAI gym. This is a discrete environment  where the agent can move in the cardinal directions, but is not guaranteed to move in the direction it chooses. The agent gets a reward of 1 when it reaches the tile marked G, and a reward of 0 in all other settings.  You can read more about FrozenLake-v1 (it is the same as FrozenLake-v0) here: [<span style="color:blue">https://www.gymlibrary.dev/environments/toy_text/frozen_lake/</span>](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/). You will not need to change any code outside of the area marked TODO, but youare free to change the hyper-parameters if you want to. For each sampled tuple (s,a,r,s′,done), the update rule for Q-learning is:

$$
Q(s,a) = \left\{\begin{array}{l}(1-\alpha ) Q(s,a)+\alpha (r+\gamma \text{max}_{a'\in A}Q(s',a')) \\
(1-\alpha ) Q(s,a) + \alpha r\end{array}\right.
$$

* The agent should act according to an epsilon-greedy policy as defined in the Reinforcement Learning 1 slides. In this equation, $\alpha$ is the learning rate hyper-parameter, and $\gamma$ is the discount factor hyper-parameter.
    * **HINT:** tests.py is worth looking at to gain an understanding of how to use the OpenAI gym env.
    * **Files to Submit:** For this section, you should submit the files Qlearning.py and QTABLE.pkl.
## OpenAI gym Environment
* You will need to use several OpenAI gym functions in order to operate your gym environment for reinforcement learning. As stated in a previous hint, tests.py has a lot of the function calls you need. Several important functions are as follows:
```python
env.step(action)
```
* Given that the environment is in states, step takes an integer specifiying the chosen action, and returns a tuple of the form (s,r,done,info). 'done' specifies whether or not $s'$ is the final state for that particular episode, and 'info' is unused in this assignment.

```python
env.reset()
```
* Resets the environment to it’s initial state, and returns that state.

```python
env.action_space.sample()
```
* Samples an integer corresponding to a random choice of action in the environment’s action space.

```python
env.actionspace.n
```
* In the setting of the environments we will be working with for these assignments, this is an integer corresponding to the number of possible actions in the environment’s action space.
* You can read more about OpenAI gym here: [<span style="color:blue">https://www.gymlibrary.dev</span>](https://www.gymlibrary.dev).