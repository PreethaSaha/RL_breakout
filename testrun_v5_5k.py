import numpy as np
import gym
import tensorflow as tf
from collections import deque
import random
import time
import matplotlib.pyplot as plt
import csv
from PIL import Image

# Hyperparameters
NUM_EPISODES = 5000
MAX_NUM_TIMESTEPS = 1000
MEMORY_SIZE = 50000
BATCH_SIZE = 128
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.995
GAMMA = 0.99
NUM_P_AV = 10
NUM_STEPS_FOR_UPDATE = 4
MIN_REPLAY_SIZE = 2000

# Create the Breakout environment
env = gym.make('ALE/Breakout-v5')

# Define the Q-Network
q_network = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 1)),
    tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(env.action_space.n, activation='linear')
])

# Define the target network
target_q_network = tf.keras.models.clone_model(q_network)
target_q_network.set_weights(q_network.get_weights())

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)

# Experience replay buffer
memory_buffer = deque(maxlen=MEMORY_SIZE)

def preprocess_state(state):
    """Convert state to grayscale, resize, and normalize using PIL."""
    if isinstance(state, tuple):  # Handle if the state is a tuple
        state = state[0]  # Extract the actual state (assuming the first item is the image)

    image = Image.fromarray(state)  # Convert to PIL Image
    gray_image = image.convert('L')  # Convert to grayscale
    gray_resized = gray_image.resize((84, 84))  # Resize to 84x84
    gray_normalized = np.array(gray_resized, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(gray_normalized, axis=-1)  # Add channel dimension

def get_action(q_values, epsilon):
    """Epsilon-greedy action selection."""
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(q_values)  # Exploit

def compute_loss(experiences, gamma, q_network, target_q_network):
    """Calculate the loss for training."""
    states, actions, rewards, next_states, done_vals = experiences

    # Use TensorFlow's functions to calculate the max Q-values
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=1)  # axis=1 for batch dimension
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))

    q_values = q_network(states)
    indices = tf.stack([tf.range(BATCH_SIZE), actions], axis=1)  # Create indices for gathering
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
    
    # Convert to numpy arrays
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)
    done_vals = np.array(done_vals, dtype=np.float32)

    return states, actions, rewards, next_states, done_vals

# Training loop
start = time.time()
reward_data = []
loss_data = []

epsilon = INITIAL_EPSILON

# Save training data to a CSV file
with open('training_data_v5_5k.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Total Rewards', 'Loss'])

    for episode in range(NUM_EPISODES):
        state = env.reset()
        state = preprocess_state(state)
        total_points = 0

        for t in range(MAX_NUM_TIMESTEPS):
            state_qn = np.expand_dims(state, axis=0)  # Expand dimensions for model input
            q_values = q_network(state_qn)  # Get Q-values
            action = get_action(q_values, epsilon)  # Choose action

            # Step and ensure it returns correctly
            step_result = env.step(action)
            next_state = step_result[0]  # Get next state
            reward = step_result[1]  # Get reward
            done = step_result[2]  # Get done flag
            # info = step_result[3]  # We can ignore info for now

            # Check if next_state is a valid image
            next_state = preprocess_state(next_state)
            memory_buffer.append((state, action, reward, next_state, float(done)))
            total_points += reward

            # Only start training once the memory buffer has enough samples
            if len(memory_buffer) >= MIN_REPLAY_SIZE and t % NUM_STEPS_FOR_UPDATE == 0:
                experiences = random.sample(memory_buffer, BATCH_SIZE)
                batch = process_batch(experiences)
                loss = agent_learn(batch, GAMMA)
                loss_data.append(loss.numpy())

                # Update the target network periodically
                if t % (NUM_STEPS_FOR_UPDATE * 10) == 0:
                    update_target_network()

            state = next_state
            if done:
                break

        reward_data.append(total_points)
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        writer.writerow([episode + 1, total_points, loss.numpy() if len(loss_data) > 0 else 0])

        if (episode + 1) % NUM_P_AV == 0:
            avg_points = np.mean(reward_data[-NUM_P_AV:])
            print(f"Episode {episode + 1} - Average Points: {avg_points:.2f}")

        # Save the model if performance improves
        if total_points >= 8.0:
            fname = f"breakout_model_v5_{total_points}_{episode}.h5"
            q_network.save(fname)

tot_time = time.time() - start
print(f"Training completed in {tot_time:.2f} seconds ({tot_time/60:.2f} minutes).")

# Plot Loss and Rewards
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(reward_data)
plt.title('Rewards Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(1, 2, 2)
plt.plot(loss_data)
plt.title('Loss Over Training Steps')
plt.xlabel('Training Steps')
plt.ylabel('Loss')

plt.show()
