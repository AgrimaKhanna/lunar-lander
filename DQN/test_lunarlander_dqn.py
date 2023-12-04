from dqn_agent import Agent
import gym
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
seed = 0
agent = Agent(state_size, action_size, seed)
agent.load("models/LunarLander_dqn_weights.pth")
log_dir = "testing_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)  # Create the directory if it doesn't exist
log_f_name = log_dir + "/testing_log.csv"  # Log file name
log_f = open(log_f_name, "w+")
log_f.write('episode,reward\n')

for e in range (1,101):
    state = env.reset()
    try:
        state = np.reshape(state, [1, state_size])
    except:
        state = state[0]
        state = np.reshape(state, [1, state_size])
    for time in range(500):
        env.render()
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        state = np.reshape(next_state, [1, state_size])
        if done:
            break
    print("Episode: {}, Reward: {}".format(e, time))
    log_f.write('{},{}\n'.format(e, time))
    log_f.flush()

env.close()
