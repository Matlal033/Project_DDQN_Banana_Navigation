### Learning Algorithm

For this project, the **Double Q-learning (DDQN)** algorithm was used, along with the **Huber Loss** method.

The neural network structure is :

| Layer | Input size | Output size | Activation |
|-------|------------|-------------|------------|
| 1 | 37 (state size) | 64 | ReLU |
| 2 | 64 | 64 | ReLU |
| 3 | 64 | 37 (state size) | None |

The hyperparameters used are :

- BUFFER_SIZE = int (2e5)
- BATCH_SIZE = 128 #better than 64 (e499 vs e564)
- GAMMA = 0.98
- TAU = 1e-3
- LR = 5e-4
- UPDATE_EVERY = 5 #better than 15 and 10 with batch_size 128 (e564 vs e706 vs e780)
- max_t=1000       #Maximum timesteps per episode
- eps_start=1.0    #Epsilon start
- eps_end=0.01     #Epsilon end 
- eps_decay=0.995  #Epsilon decay rate
 
### Plot of rewards

![](Figure_1.png)

### Ideas for Future Work

To further improve upon the algorithm used, a few modifications could be easily added, such as :
- Using Prioritized Experience Replay (**PER**) to give more value to good experiences.
- Implementing a **Dueling DDQN** instead of a simple **DDQN** to make the model converge in less episode by separating the state-value function and the action-advantage function.
