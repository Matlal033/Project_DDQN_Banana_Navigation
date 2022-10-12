import sys
import torch
import numpy as np
from agent import Agent
from unityagents import UnityEnvironment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = UnityEnvironment(file_name='Banana_Windows_x86_64/Banana.exe')

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

action_size = brain.vector_action_space_size

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
seed = 0

try:
    filename = sys.argv[1]
except:
    filename = None

print(filename)
agent = Agent(state_size, action_size, seed, filename)

while True:
    action = agent.act(state,0.01)                      # default epsilon is 0, to take only argmax
    action = action.astype(int)                    # Cast action as np.int32 to avoid "'numpy.int64' object has no attribute 'keys'" error
    print("action: ", action)
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break

print("Score: {}".format(score))
