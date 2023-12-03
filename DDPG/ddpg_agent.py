import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer():
    def __init__(self, buffer_size, input_shape, n_actions):
        self.buffer_size = buffer_size
        self.mem_count = 0
        self.state_memory = np.zeros((self.buffer_size, *input_shape))
        self.new_state_memory = np.zeros((self.buffer_size, *input_shape))
        self.action_memory = np.zeros((self.buffer_size, n_actions))
        self.reward_memory = np.zeros(self.buffer_size)
        self.terminal_memory = np.zeros(self.buffer_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_count % self.buffer_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = next_state
        self.terminal_memory[index] = done
        self.mem_count += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_count, self.buffer_size)
        batch_indices = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch_indices]
        actions = self.action_memory[batch_indices]
        rewards = self.reward_memory[batch_indices]
        next_states = self.new_state_memory[batch_indices]
        dones = self.terminal_memory[batch_indices]

        return states, actions, rewards, next_states, dones

# Critic Network definition
class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, hidden_dims, n_actions, name, checkpoint_dir='./models'):
        super(CriticNetwork, self).__init__()
        # Initialize input, hidden, and output dimensions
        self.input_dims = input_dims
        self.fc1_dims = hidden_dims[0]
        self.fc2_dims = hidden_dims[1]
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_file = os.path.join(checkpoint_dir, name + '_ddpg.pth')

        # Define network layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        # Initialize weights
        self._initialize_weights()

        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.01)

        # Setup device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _initialize_weights(self):
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        f3 = 0.003
        f4 = 1. / np.sqrt(self.action_value.weight.data.size()[0])
        # Initialize weights uniformly
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

    def forward(self, state, action):
        # Forward pass through the network
        state_value = F.relu(self.bn1(self.fc1(state)))
        state_value = self.bn2(self.fc2(state_value))
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value))
        q_value = self.q(state_action_value)

        return q_value

    def save_checkpoint(self):
        if not os.path.exists('./models'):
            os.makedirs('./models')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_best.pth')
        T.save(self.state_dict(), checkpoint_file)

# Actor Network definition
class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, hidden_dims, n_actions, name, checkpoint_dir='./models'):
        super(ActorNetwork, self).__init__()
        # Initialize input, hidden, and output dimensions
        self.input_dims = input_dims
        self.fc1_dims = hidden_dims[0]
        self.fc2_dims = hidden_dims[1]
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_file = os.path.join(checkpoint_dir, name + '_ddpg.pth')

        # Define network layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        # Initialize weights
        self._initialize_weights()

        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Setup device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _initialize_weights(self):
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        f3 = 0.003
        # Initialize weights uniformly
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

    def forward(self, state):
        # Forward pass through the network
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = T.tanh(self.mu(x))  # tanh activation for bounded action space

        return x

    def save_checkpoint(self):
        if not os.path.exists('./models'):
            os.makedirs('./models')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_best.pth')
        T.save(self.state_dict(), checkpoint_file)

# Ornstein-Uhlenbeck Noise definition
class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

# Agent definition
class Agent():
    def __init__(self, policy_lr, critic_lr, input_dims, tau, env_id, n_actions, gamma=0.99,
               buffer_size=1000000, hidden_dims=[400, 300],
               batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.algo = "DDPG"

        # Initialize ReplayBuffer
        self.buffer = ReplayBuffer(buffer_size, input_dims, n_actions)

        # Initialize Ornstein-Uhlenbeck Noise
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        # Initialize Actor and Critic Networks
        self.actor = ActorNetwork(policy_lr, input_dims, hidden_dims, n_actions=n_actions, name=env_id + '_actor')
        self.critic = CriticNetwork(critic_lr, input_dims, hidden_dims, n_actions=n_actions, name=env_id + '_critic')

        # Initialize Target Networks
        self.target_actor = ActorNetwork(policy_lr, input_dims, hidden_dims, n_actions=n_actions, name=env_id + '_target_actor')
        self.target_critic = CriticNetwork(critic_lr, input_dims, hidden_dims, n_actions=n_actions, name=env_id + '_target_critic')

        # Update target network parameters
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        try:
            observation = observation[np.newaxis, :]  # Add a batch dimension to the state
        except:
            observation = np.array(observation[0])
            observation = observation[np.newaxis, :]  # Handle different state formats and add a batch dimension
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        self.actor.train()

        return mu_prime.cpu().detach().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.store_transition(state, action, reward, next_state, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def save_weights(self, actor_filepath, critic_filepath):
        # Save the weights of the actor network
        T.save(self.actor.state_dict(), actor_filepath)
        # Save the weights of the critic network
        T.save(self.critic.state_dict(), critic_filepath)

    def update(self):
        if self.buffer.mem_count < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        next_states = T.tensor(next_states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)

        target_actions = self.target_actor.forward(next_states)
        target_values = self.target_critic.forward(next_states, target_actions)
        critic_value = self.critic.forward(states, actions)

        target_values[dones] = 0.0
        target_values = target_values.view(-1)
        target = rewards + self.gamma * target_values
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -T.mean(self.critic.forward(states, self.actor.forward(states)))
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()
        return critic_loss.item(), actor_loss.item()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
