import numpy as np
import gym
import tensorflow as tf
from collections import deque
import random
import time
import matplotlib.pyplot as plt
from PIL import Image
import os

# Create a directory for saving models if it doesn't exist
if not os.path.exists('saved_models_n'):
    os.makedirs('saved_models_n')
if not os.path.exists('logs'):
    os.makedirs('logs')

# Initialize CSV logging
csv_file = 'logs/training_log_2.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Eval Reward', 'Avg Reward (Last 100)', 'Loss', 'Epsilon', 'Learning Rate'])

# Hyperparameters
NUM_EPISODES = 5000
MAX_NUM_TIMESTEPS = 1000
MEMORY_SIZE = 100000
BATCH_SIZE = 128
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.995
GAMMA = 0.99
NUM_STEPS_FOR_UPDATE = 4
MIN_REPLAY_SIZE = 5000
INITIAL_LEARNING_RATE = 0.0001
MAX_LEARNING_RATE = 0.001
REGULARIZATION = 1e-4

# Create the Breakout environment
env = gym.make('ALE/Breakout-v5')

# Define the Q-Network with Regularization
q_network = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 1)),
    tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(REGULARIZATION)),
    tf.keras.layers.Dense(env.action_space.n, activation='linear', kernel_regularizer=tf.keras.regularizers.L2(REGULARIZATION))
])

# Define the target network
target_q_network = tf.keras.models.clone_model(q_network)
target_q_network.set_weights(q_network.get_weights())

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)

# Experience replay buffer
memory_buffer = deque(maxlen=MEMORY_SIZE)

def preprocess_state(state):
    """Convert state to grayscale, resize, and normalize."""
    if isinstance(state, tuple):  # Handle if the state is a tuple
        state = state[0]
    image = Image.fromarray(state)
    gray_image = image.convert('L')
    gray_resized = gray_image.resize((84, 84))
    gray_normalized = np.array(gray_resized, dtype=np.float32) / 255.0
    return np.expand_dims(gray_normalized, axis=-1)

def get_action(q_values, epsilon):
    """Epsilon-greedy action selection."""
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_values)

def compute_loss(experiences, gamma, q_network, target_q_network):
    """Double DQN loss computation."""
    states, actions, rewards, next_states, done_vals = experiences
    next_q_values = q_network(next_states)  # Use Q-network for action selection
    best_actions = tf.argmax(next_q_values, axis=1)  # Get best actions
    target_q_values = target_q_network(next_states)  # Use Target Network for value evaluation
    max_qsa = tf.gather_nd(
        target_q_values,
        tf.stack([tf.range(tf.shape(best_actions)[0], dtype=tf.int32), tf.cast(best_actions, tf.int32)], axis=1)
    )
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))
    q_values = q_network(states)
    indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
    q_values = tf.gather_nd(q_values, indices)
    loss = tf.reduce_mean(tf.square(y_targets - q_values))  # Mean squared error
    return loss

def agent_learn(experiences, gamma):
    """Updates the weights of the Q networks."""
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)
    gradients = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
    return loss

def update_target_network():
    """Copy the weights from the Q-network to the target network."""
    target_q_network.set_weights(q_network.get_weights())

def process_batch(experiences):
    """Ensure the consistency of the batch components."""
    states, actions, rewards, next_states, done_vals = zip(*experiences)
    return (np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(done_vals, dtype=np.float32))

def evaluate_agent(env, q_network, num_episodes=10, epsilon=0.05):
    """Evaluate the agent on unseen episodes with some exploration."""
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        state = preprocess_state(state)
        episode_reward = 0
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Exploratory action
            else:
                q_values = q_network(np.expand_dims(state, axis=0))
                action = np.argmax(q_values)  # Exploitative action
            step_result = env.step(action)
            next_state, reward, done, truncated, info = step_result if len(step_result) > 4 else (*step_result, False)
            done = done or truncated
            state = preprocess_state(next_state)
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards), np.std(total_rewards)

# Training loop
start = time.time()
reward_data = []
loss_data = []
loss_per_episode = []
epsilon = INITIAL_EPSILON
best_eval_reward = -float('inf')
patience = 100  # Early stopping patience
no_improvement_count = 0

for episode in range(NUM_EPISODES):
    state = env.reset()
    state = preprocess_state(state)
    total_points = 0
    episode_losses = []

    for t in range(MAX_NUM_TIMESTEPS):
        q_values = q_network(np.expand_dims(state, axis=0))
        action = get_action(q_values, epsilon)

        step_result = env.step(action)
        next_state, reward, done, truncated, info = step_result if len(step_result) > 4 else (*step_result, False)
        done = done or truncated
        reward = np.clip(reward, -1, 1)  # Reward Clipping
        next_state = preprocess_state(next_state)

        memory_buffer.append((state, action, reward, next_state, float(done)))
        total_points += reward

        if len(memory_buffer) >= MIN_REPLAY_SIZE and t % NUM_STEPS_FOR_UPDATE == 0:
            experiences = random.sample(memory_buffer, BATCH_SIZE)
            batch = process_batch(experiences)
            loss = agent_learn(batch, GAMMA)
            episode_losses.append(loss.numpy())

        if len(memory_buffer) >= MIN_REPLAY_SIZE and t % (NUM_STEPS_FOR_UPDATE * 10) == 0:
            update_target_network()

        state = next_state
        if done:
            break

    reward_data.append(total_points)
    loss_per_episode.append(np.mean(episode_losses))
    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

    # Adjust learning rate
    #lr_adjustment = INITIAL_LEARNING_RATE + (episode / NUM_EPISODES) * (MAX_LEARNING_RATE - INITIAL_LEARNING_RATE)
    lr_adjustment = INITIAL_LEARNING_RATE 
    optimizer.learning_rate.assign(lr_adjustment)

    if (episode + 1) % 100 == 0:
        eval_reward, eval_std = evaluate_agent(env, q_network)
        avg_reward = np.mean(reward_data[-100:])
        avg_reward_std = np.std(reward_data[-100:])
        print(f"Episode {episode + 1}, Eval Reward: {eval_reward:.2f} ± {eval_std:.2f}, Avg Reward (last 100): {avg_reward:.2f} ± {avg_reward_std:.2f}, LR: {lr_adjustment:.6f}")


        if eval_reward > best_eval_reward:
            best_eval_reward = eval_reward
            no_improvement_count = 0
            q_network.save_weights(f'saved_models_n/best_model_weights_episode_{episode + 1}.h5')
            print("Model weights saved.")
        else:
            no_improvement_count += 1

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode + 1, eval_reward, np.mean(reward_data[-100:]), loss_per_episode, epsilon, optimizer.learning_rate.numpy()])

        if no_improvement_count >= patience:
            print("Early stopping triggered.")
            break
    

tot_time = time.time() - start
print(f"Training completed in {tot_time:.2f} seconds.")

# Plot rewards and losses
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(reward_data, label='Total Rewards')
plt.title('Rewards Over Episodes')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(loss_per_episode, label='Loss Per Episode', color='orange')
plt.title('Loss Over Episodes')
plt.xlabel('Episodes')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
