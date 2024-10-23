# RL_breakout
# Work in progess!

# Description

This repository contains an implementation of a Deep Q-Network (DQN) to train an agent to play the classic _Breakout_ game. The agent leverages convolutional neural networks to process game frames and utilizes reinforcement learning techniques, such as experience replay and target networks, to learn effective strategies for maximizing rewards. The model is developed using _TensorFlow_ and the _OpenAI Gym_ environment for gameplay simulation.

The main features of the code are as follows:

- _Deep Reinforcement Learning_: utilizes a DQN architecture with convolutional neural networks to play Breakout
- Experience Replay: stores past experiences in a buffer and randomly samples them to break correlation between consecutive experiences and improve learning stability
- Target Network: incorporates a separate target network to stabilize training by updating it less frequently than the main network
- Image Preprocessing: converts game frames to grayscale, resizes them, and normalizes pixel values to feed into the network
- Epsilon-Greedy Policy: balances exploration and exploitation through an epsilon-greedy approach.



