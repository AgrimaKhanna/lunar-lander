import gym
import numpy as np
import os
import csv
import json
import time
from ddpg_khanna8 import DDPG

# Hyperparameters
EPISODES = 5000
BATCH_SIZE = 64
DISCOUNT = 0.99
TAU = 0.0001

# Initialize environment and agent
env = gym.make('LunarLanderContinuous-v2', render_mode = 'human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = DDPG(state_dim, action_dim, max_action)

# Lists to store metrics
episode_rewards = []
actor_losses = []
critic_losses = []

start_time = time.time()

# Training loop
for episode in range(EPISODES):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.replay_buffer.add((state, action, reward, next_state, done))
        state = next_state
        episode_reward += reward

        # Train agent and capture losses
        if len(agent.replay_buffer.storage) >= BATCH_SIZE:
            actor_loss, critic_loss = agent.train(BATCH_SIZE, DISCOUNT, TAU)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

    episode_rewards.append(episode_reward)
    print(f"Episode {episode + 1}/{EPISODES}, Reward: {episode_reward:.2f}")

    # Termination criteria (average reward over last 100 episodes >= 200)
    if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 200:
        print("Termination criterion reached!")
        break

end_time = time.time()
training_duration = end_time - start_time

# Save the trained weights
actor_filename = "lunarlander_ddpg_actor_weights_khanna8.h5"
critic_filename = "lunarlander_ddpg_critic_weights_khanna8.h5"
agent.save_weights(actor_filename, critic_filename)

# Define directories and filenames for saving metrics
OUTPUT_DIR = "training_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

REWARDS_FILE = os.path.join(OUTPUT_DIR, "episode_rewards_khanna8.csv")
ACTOR_LOSS_FILE = os.path.join(OUTPUT_DIR, "actor_losses_khanna8.csv")
CRITIC_LOSS_FILE = os.path.join(OUTPUT_DIR, "critic_losses_khanna8.csv")

# Save episode rewards to CSV
with open(REWARDS_FILE, 'w', newline='') as file:
    writer = csv.writer(file)
    for reward in episode_rewards:
        writer.writerow([reward])

# Convert actor losses to scalars and save to CSV
actor_losses_scalar = [float(loss.numpy()) for loss in actor_losses]
with open(ACTOR_LOSS_FILE, 'w', newline='') as file:
    writer = csv.writer(file)
    for loss in actor_losses_scalar:
        writer.writerow([loss])

# Convert critic losses to scalars and save to CSV
critic_losses_scalar = [float(loss.numpy()) for loss in critic_losses]
with open(CRITIC_LOSS_FILE, 'w', newline='') as file:
    writer = csv.writer(file)
    for loss in critic_losses_scalar:
        writer.writerow([loss])

# Save training time
TIME_FILE = os.path.join(OUTPUT_DIR, "training_time_khanna8.txt")
with open(TIME_FILE, 'w') as file:
    file.write(str(training_duration))

