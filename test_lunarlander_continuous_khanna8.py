import gym
import numpy as np
import moviepy
from ddpg_khanna8 import DDPG

# Hyperparameters
TEST_EPISODES = 10

# Initialize environment and agent
env = gym.make('LunarLanderContinuous-v2', render_mode ='human')
env = gym.wrappers.RecordVideo(env, 'video')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = DDPG(state_dim, action_dim, max_action)

# Load the trained weights
actor_filename = "lunarlander_ddpg_actor_weights_khanna8.h5"
critic_filename = "lunarlander_ddpg_critic_weights_khanna8.h5"
agent.load_weights(actor_filename, critic_filename)

# Testing loop
test_rewards = []
for episode in range(TEST_EPISODES):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        env.render()  # Visualize the agent's behavior
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        episode_reward += reward

    test_rewards.append(episode_reward)
    print(f"Test Episode {episode + 1}/{TEST_EPISODES}, Reward: {episode_reward:.2f}")

env.close()

# Calculate and print the average test reward
average_test_reward = np.mean(test_rewards)
print(f"Average Test Reward: {average_test_reward:.2f}")
