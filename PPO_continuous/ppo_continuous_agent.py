import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

# Setting the device for torch tensors to CPU
device = torch.device('cpu')


# RolloutBuffer class to store the experiences during a rollout
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probabilities = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        # Clearing the buffer contents
        del self.actions[:]
        del self.states[:]
        del self.log_probabilities[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


# ActorCritic neural network class
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        # Setting the initial action variance
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # Defining the actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

        # Defining the critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        # Update the action variance
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def act(self, state):
        # Forward pass through actor network to get action mean
        action_mean = self.actor(state)
        # Constructing a multivariate normal distribution to sample action
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        # Sampling an action and its log probability
        action = dist.sample()
        action_log_probability = dist.log_prob(action)
        # Getting state value estimate from critic network
        state_val = self.critic(state)
        return action.detach(), action_log_probability.detach(), state_val.detach()

    def evaluate(self, state, action):
        # Evaluate the policy's output and value for a given state-action pair
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)
        action_log_probabilities = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_log_probabilities, state_values, dist_entropy


# PPO algorithm class
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, action_std_init=0.6):
        # Initializing parameters and networks for PPO
        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MSE_Loss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        # Set new action standard deviation
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        # Decay the action standard deviation
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
        else:
            self.set_action_std(self.action_std)

    def select_action(self, state):
        # Select action for a given state using the old policy
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_log_probability, state_val = self.policy_old.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probabilities.append(action_log_probability)
        self.buffer.state_values.append(state_val)
        return action.detach().cpu().numpy().flatten()

    def update(self):
        # Evaluating the performance of a policy using Monte Carlo estimate of returns.
        # Update the policy based on collected data
        rewards = []
        discounted_reward = 0
        # Calculating the discounted rewards
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # Normalizing rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_log_probabilities = torch.squeeze(torch.stack(self.buffer.log_probabilities, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        advantages = rewards.detach() - old_state_values.detach()
        # PPO update loop
        for _ in range(self.k_epochs):
            log_probabilities, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            # Calculating the PPO loss
            ratios = torch.exp(log_probabilities - old_log_probabilities.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MSE_Loss(state_values, rewards) - 0.01 * dist_entropy
            # Backpropagation
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # Clear the buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        # Save the model state
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        # Load the model state
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))





