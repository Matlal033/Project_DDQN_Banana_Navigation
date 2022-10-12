# Project DDQN Banana Navigation
 
### Project Details

This is the first project of the Udacity Deep Reinforcement Learning nanodegree.
For this particular instance, a DDQN is used to solve the Banana Navigation problem from the Unity environment.

The state space has a size of 37, and the action space a size of 4 (move forward, move backward, turn left, turn right).
Yellow bananas are worth 1 point, and blue ones are worth -1 point.
The environment is considered solved when the average score over 100 consecutive episodes reaches 13.0.

### Getting started

To run this code, Python 3.6 is required, along with the dependencies found in [requirements.txt](https://github.com/Matlal033/Project_DDQN_Banana_Navigation/edit/main/requirements.txt)

You will also need to download the unity environnment compressed file from one of the follow links, and extract it inside under the `Project_DDQN_Banana_Navigation/` folder :

- Linux : [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX : [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit) : [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit) : [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

### Instructions

To train the agent from scratch, your can lauch `main.py` from you command line, using the virtual environment with Python 3.6 and the required packaged mentionned earlier.

To watch a trained agent, you can lauch `watch_agent.py [patch_to_checkpoint]`.
