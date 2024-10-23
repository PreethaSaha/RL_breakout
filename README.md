# RL_breakout
# Work in progess!

# Description

This repository contains an implementation of a Deep Q-Network (DQN) to train an agent to play the classic __Breakout__ game. The agent leverages convolutional neural networks to process game frames and utilizes reinforcement learning techniques, such as experience replay and target networks, to learn effective strategies for maximizing rewards. The model is developed using __TensorFlow__ and the __OpenAI Gym__ environment for gameplay simulation.

The main features of the code are as follows:

- __Deep Reinforcement Learning__: utilizes a DQN architecture with convolutional neural networks to play Breakout
- __Experience Replay__: stores past experiences in a buffer and randomly samples them to break correlation between consecutive experiences and improve learning stability
- __Target Network__: incorporates a separate target network to stabilize training by updating it less frequently than the main network
- __Image Preprocessing__: converts game frames to grayscale, resizes them, and normalizes pixel values to feed into the network
- __Epsilon-Greedy Policy__: balances exploration and exploitation through an epsilon-greedy approach.
  

## Installation:
To use this code, please follow these steps:

1. Clone the repository

<div>
  <button class="copy-button" onclick="copyToClipboard(this.parentElement.nextElementSibling.textContent)"></button>
  <pre><code>git clone https://github.com/PreethaSaha/RL_breakout.git
  </code></pre>
</div>

2. Install the required dependencies

<div>
  <button class="copy-button" onclick="copyToClipboard(this.parentElement.nextElementSibling.textContent)"></button>
  <pre><code> pip install -r requirements.txt
  </code></pre>
</div>

## Usage:

<div>
  <button class="copy-button" onclick="copyToClipboard(this.parentElement.nextElementSibling.textContent)"></button>
  <pre><code>python testrun_v5_5k.py

  </code></pre>
</div>

## Future improvements:

- __Parameter Tuning__: experiment with different hyperparameters for better training 
- __Prioritized Experience Replay__: use a prioritized experience replay buffer to enhance learning from more informative experiences

## Benefaction:

Contributions are welcome! Please submit issues and pull requests for improvements or bug fixes.

## License:

This project is licensed under the MIT License - see the [LICENSE](https://github.com/PreethaSaha/RL_breakout/blob/main/LICENSE) file for more details.
