import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


# Define the Actor Network
def build_actor_network(input_dim, output_dim, max_action):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(400, activation='relu')(inputs)
    x = tf.keras.layers.Dense(300, activation='relu')(x)
    x = tf.keras.layers.Dense(output_dim, activation='tanh')(x)
    outputs = max_action * x
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# Define the Critic Network
def build_critic_network(input_dim, action_dim):
    state_input = tf.keras.layers.Input(shape=(input_dim,))
    action_input = tf.keras.layers.Input(shape=(action_dim,))
    combined = tf.keras.layers.Concatenate()([state_input, action_input])

    x = tf.keras.layers.Dense(400, activation='relu')(combined)
    x = tf.keras.layers.Dense(300, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)  # Q-value
    model = tf.keras.Model(inputs=[state_input, action_input], outputs=outputs)
    return model


# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        state, action, reward, next_state, done = data

        # Convert states to numpy arrays with specified shape, assuming they are 1D arrays
        try:
            state = np.asarray(state, dtype=np.float32).reshape(1, -1)
            next_state = np.asarray(next_state, dtype=np.float32).reshape(1, -1)
        except:
            state = state[0]
            state = np.asarray(state, dtype=np.float32).reshape(1, -1)
            next_state = np.asarray(next_state, dtype=np.float32).reshape(1, -1)

        # Action should be a 1D array with a single value if it's a scalar or multiple if it's a vector

        action = np.asarray(action, dtype=np.float32).reshape(1, -1)

        # Rewards and dones should be single values, ensure they are floats
        reward = float(reward)
        done = float(done)

        # Add the data to the buffer
        data = (state, action, reward, next_state, done)
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.storage), size=batch_size)
        batch = [self.storage[i] for i in indices]

        states, actions, rewards, next_states, dones = map(lambda seq: np.array(seq).reshape(batch_size, -1),
                                                           zip(*batch))

        rewards = np.array(rewards).reshape(-1, 1)
        dones = np.array(dones).reshape(-1, 1).astype(np.float32)

        return states, actions, rewards, next_states, dones


# DDPG Agent
class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        # Actor and Critic Networks
        self.actor = build_actor_network(state_dim, action_dim, max_action)
        self.actor_target = build_actor_network(state_dim, action_dim, max_action)
        self.actor_optimizer = Adam(learning_rate=1e-4)

        self.critic = build_critic_network(state_dim, action_dim)
        self.critic_target = build_critic_network(state_dim, action_dim)
        self.critic_optimizer = Adam(learning_rate=1e-3)

        # Initialize target weights with model weights
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer(max_size=1e6)

    def select_action(self, state):
        try:
            state = state[np.newaxis, :]
        except:
            state = np.array(state[0])
            state = state[np.newaxis, :]
        state = np.array(state).reshape(1, -1)  # Reshape to (1, num_state_vars)
        return self.actor(state).numpy()[0]

    def train(self, batch_size=64, discount=0.99, tau=0.005):
        # Sample experiences from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Calculate target Q-values using critic target
        target_actions = self.actor_target(next_states)
        target_Q = self.critic_target([next_states, target_actions])
        target_Q = rewards + (1 - dones) * discount * target_Q

        # Update the critic network
        with tf.GradientTape() as tape:
            current_Q = self.critic([states, actions])
            critic_loss = tf.keras.losses.MSE(target_Q, current_Q)
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        # Update the actor network
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic([states, actions]))
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        # Soft update the target networks
        for param, target_param in zip(self.critic.trainable_variables, self.critic_target.trainable_variables):
            target_param.assign(tau * param + (1 - tau) * target_param)
        for param, target_param in zip(self.actor.trainable_variables, self.actor_target.trainable_variables):
            target_param.assign(tau * param + (1 - tau) * target_param)

        return actor_loss, critic_loss

    def save_weights(self, actor_filename, critic_filename):
        self.actor.save_weights(actor_filename)
        self.critic.save_weights(critic_filename)

    def load_weights(self, actor_filename, critic_filename):
        self.actor.load_weights(actor_filename)
        self.critic.load_weights(critic_filename)
