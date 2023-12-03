from dqn_agent import DQNAgent
import numpy as np
import gym
import warnings

warnings.filterwarnings("ignore")

env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
episodes = 1000

for e in range(episodes):
    state = env.reset()
    try:
        state = np.array(state)
        state = np.reshape(state, [1, state_size])
    except:
        state = state[0]
        state = np.array(state)
        state = np.reshape(state, [1, state_size])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.memory.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
        if len(agent.memory.memory) > agent.memory.batch_size:
            agent.replay()
    print("Episode: {}/{}, score: {}".format(e, episodes, time))
    if e % 10 == 0:
        agent.save("./lunarlander_dqn_weights.h5")

