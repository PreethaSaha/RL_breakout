# RL_breakout - Work in progess!

# Description

This repository contains an implementation of a Deep Q-Network (DQN) to train an agent to play the classic __Breakout__ game. The agent leverages convolutional neural networks to process game frames and utilizes reinforcement learning techniques, such as experience replay and target networks, to learn effective strategies for maximizing rewards. The model is developed using __TensorFlow__ and the __OpenAI Gym__ environment for gameplay simulation.

The main features of the code are as follows: 

- __Deep Reinforcement Learning__: utilizes a DQN architecture with convolutional neural networks to play Breakout
- __Experience Replay__: stores past experiences in a buffer and randomly samples them to break correlation between consecutive experiences and improve learning stability
- __Target Network__: incorporates a separate target network to stabilize training by updating it less frequently than the main network
- __Image Preprocessing__: converts game frames to grayscale, resizes them to 84x84 pixels, and normalizes pixel values [0, 1] to feed into the network
- __Epsilon-Greedy Policy__: balances exploration and exploitation through an epsilon-greedy approach.
  
  ## Pre-requisites:
  
  Python 3.7+;
  OpenAI Gym (with the atari package);
  TensorFlow 2.x;
  Numpy;
  PIL (Python Imaging Library);
  Matplotlib

  The appropriate non-conflicting versions of the dependencies used here are quoted in the _requirements.txt_. To install these, please follow __step 2__ of __Usage__. 

## Usage:
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

3. To train the DQN agent, run:

<div>
  <button class="copy-button" onclick="copyToClipboard(this.parentElement.nextElementSibling.textContent)"></button>
  <pre><code>python testrun_v5_5k.py
  </code></pre>
</div>

You can adjust training parameters such as the number of episodes, epsilon decay, and batch size in the testrun_v5_5k.py file

## Results:

  ### Training:
  The training loop runs for a specified number of episodes. During each episode, the agent starts by exploring (random actions) to learn about the environment. As training progresses, it shifts toward exploiting the best-known actions, reducing random actions.

The training progress is saved in a CSV file. The model weights are saved in breakout_model_v5_XX_XX.h5 whenever the agent achieves a predefined reward threshold.

[<img src = "https://github.com/PreethaSaha/RL-breakout/blob/main/breakout_v5_5k_1e-4.png" width = "20%">]: #
![alt text](https://github.com/PreethaSaha/RL_breakout/blob/main/breakout_v5_5k_1e-4.png)

![best-performing test episode](media/best_breakout_episode.gif)

## Future improvements:

- __Parameter Tuning__: experiment with different hyperparameters for better training 
- __Prioritized Experience Replay__: use a prioritized experience replay buffer to enhance learning from more informative experiences

## Benefaction:

Contributions are welcome! Please submit issues and pull requests for improvements or bug fixes.

## License:

This project is licensed under the MIT License - see the [LICENSE](https://github.com/PreethaSaha/RL_breakout/blob/main/LICENSE) file for more details.

### References:

1. [Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). "Human-level control through deep reinforcement learning."](https://www.nature.com/articles/nature14236)
2. [Lin, L. J. (1992). "Self-improving reactive agents based on reinforcement learning, planning and teaching." Machine Learning, 8(3-4), 293-321.](https://link.springer.com/article/10.1007/BF00992699)
3. [Bellemare, M. G., Naddaf, Y., Veness, J., & Bowling, M. (2013). "The Arcade Learning Environment: An Evaluation Platform for General Agents." Journal of Artificial Intelligence Research, 47, 253-279.](https://arxiv.org/abs/1207.4708)
