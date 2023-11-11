import matplotlib.pyplot as plt

# Load saved training metrics
with open(REWARDS_FILE, 'r') as f:
    episode_rewards = json.load(f)

with open(ACTOR_LOSS_FILE, 'r') as f:
    actor_losses = json.load(f)

with open(CRITIC_LOSS_FILE, 'r') as f:
    critic_losses = json.load(f)

# Calculate the 100-episode running mean for rewards
running_mean_rewards = [np.mean(episode_rewards[max(0, i-99):i+1]) for i in range(len(episode_rewards))]

# Plot training episode rewards
plt.figure(figsize=(12, 6))
plt.plot(episode_rewards, label="Episode Reward")
plt.plot(running_mean_rewards, label="100-Episode Running Mean", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.title("Training Episode Rewards")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_rewards_plot.png")
plt.show()

# Plot training losses
plt.figure(figsize=(12, 6))
plt.plot(actor_losses, label="Actor Loss")
plt.plot(critic_losses, label="Critic Loss")
plt.xlabel("Training Iteration")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Losses")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_losses_plot.png")
plt.show()

training_duration
