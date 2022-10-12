# Project Report

### Learning Algorithm

For this project, the **Double Q-learning (DDQN)** algorithm was used, along with the **Huber Loss** method.

The hyperparameters used are :

- *BUFFER_SIZE* = int (2e5)
- *BATCH_SIZE* = 128
- *GAMMA* = 0.98
- *TAU* = 1e-3
- *LR* = 5e-4
- *UPDATE_EVERY* = 5
- *Maximum timesteps per episode* = 1000
- *Epsilon start* = 1.0
- *Epsilon end* = 0.01
- *Epsilon decay rate* = 0.995

The neural network structure is :

| Layer | Input size | Output size | Activation |
|-------|------------|-------------|------------|
| 1 | 37 (state size) | 64 | ReLU |
| 2 | 64 | 64 | ReLU |
| 3 | 64 | 37 (state size) | None |

### Plot of rewards

#### Average score of 13

In **499 episodes**, the game was solved by achieving an average reward of at least +13 over 100 episodes.

![](Figure_1.png)

```
Episode 100     Average Score: 0.37
Episode 200     Average Score: 2.65
Episode 300     Average Score: 7.37
Episode 400     Average Score: 9.43
Episode 500     Average Score: 11.86
Episode 599     Average Score: 13.00
Environment solved in 499 episodes!     Average Score: 13.00
```

#### Average score of 17

In **701 episodes**, an average score of +17 over 100 episodes could be achieved.


### Ideas for Future Work

To further improve upon the algorithm used, a few modifications could be easily added, such as :
- Using Prioritized Experience Replay (**PER**) to give more value to good experiences.
- Implementing a **Dueling DDQN** instead of a simple **DDQN** to make the model converge in less episode by separating the state-value function and the action-advantage function.
