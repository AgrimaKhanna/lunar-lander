# Import necessary libraries
import gym
import torch
import numpy as np
from ddpg_agent import Agent  # Import the Agent from ddpg_vunetid.py

# Define the environment name
environment_name = 'LunarLanderContinuous-v2'

# Create the Lunar Lander environment
env = gym.make(environment_name, render_mode='human')

# Agent initialization parameters
agent_args = {
    'policy_lr': 0.0001,
    'critic_lr': 0.001,
    'input_dims': env.observation_space.shape,
    'tau': 0.001,
    'env_id': environment_name,
    'n_actions': env.action_space.shape[0],
    'hidden_dims': [400, 300],
    'batch_size': 64
}

# Initialize the agent
agent = Agent(**agent_args)

# Load the saved weights into the agent
agent.load_models()  # This assumes you have already trained and saved your model

# Testing loop
for episode in range(10):  # Test for 10 episodes
    state = env.reset()  # Reset the environment to the starting state
    done = False
    while not done:  # Run until the episode is done
        action = agent.choose_action(state)  # Select an action based on the current state
        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = np.squeeze(action)
        state, _, done, _, _ = env.step(action)  # Apply the action to the environment
        env.render()  # Render the environment to visualize the agent's behavior

# Close the environment after testing
env.close()
