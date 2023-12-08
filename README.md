# Reinforcement Learning Agents for Lunar Lander

This repository contains the implementation of four different reinforcement learning algorithms designed to operate within the Lunar Lander environment provided by OpenAI Gym. Each algorithm has its own dedicated folder containing the necessary Python scripts for training, testing, and visualizing the performance of the agents.

## Directory Structure

- `DQN/`: Deep Q-Network implementation
- `DDPG/`: Deep Deterministic Policy Gradient implementation
- `PPO_discrete/`: Proximal Policy Optimization for discrete action space
- `PPO_continuous/`: Proximal Policy Optimization for continuous action space

Each folder includes the following files:
- `{algo_name}_agent.py`: The core algorithm and agent logic.
- `train_lunarlander_{algo_name}.py`: Script to train the agent.
- `test_lunarlander_{algo_name}.py`: Script to test the trained agent.
- `plot_figures.py`: Utility script to plot training and testing logs.

## Instructions

To interact with any of the reinforcement learning agents, follow the steps below:

### Training the Agent

Navigate to the directory of the desired algorithm and run the training script:

```bash
cd DQN
python3 train_lunarlander_dqn.py
```

Replace DQN with DDPG, PPO_discrete, or PPO_continuous as needed to train the respective agents.

### Testing the Trained Agent

To test a trained agent, ensure you're in the correct directory and execute the testing script:

```bash
cd DQN
python3 test_lunarlander_dqn.py
```
Make sure to replace DQN with the appropriate folder name as per the agent you wish to test.

### Visualizing Performance
To visualize the training and testing logs with plotted figures, run the following command within any of the algorithm directories:

```bash
python3 plot_figures.py
```
### Note
Ensure you have all the necessary dependencies installed, as specified in requirements.txt, before running the scripts.

