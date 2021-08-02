# Deep-Q-Learning-mini-game
2D simple mini-game environment to test Deep Reinforcement Learning Algorithms.

The game consists in a 10x10 grid where an agent (purple circle) must reach a goal (green circle) while avoiding an enemy (red circle). Here, we use Deep Q-Learning to find the optimal policy for the player to play the game.

Trained the Agent to find the best path using Tensorflow and Keras. For the CNN architecture, see lines 308-345.

Here is an example of the environment, a 10x10 grid where there is an agent (P, for purple), a goal (G, for Green) and an enemy (R, for Red).
The Agent is able to move up, down, right and left, as well as the enemy. The food is static.

```
   _  _  _  _  _  _  _  _  _  _
  |                             |
  |                         G   |
  |                             |
  |                             |
  |                             |
  |                      R      |
  |                             |
  |                             |
  |     P                       |
  |_  _  _  _  _  _  _  _  _  _ |
```
