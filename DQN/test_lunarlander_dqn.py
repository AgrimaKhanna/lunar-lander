from dqn_agent import DQNAgent
import gym
import numpy as np
import warnings

warnings.filterwarnings("ignore")

env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
agent.load("./lunarlander_dqn_weights.h5")

for e in range(100):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        env.render()
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        state = np.reshape(next_state, [1, state_size])
        if done:
            break
    print("Episode: {}, score: {}".format(e, time))

env.close()
