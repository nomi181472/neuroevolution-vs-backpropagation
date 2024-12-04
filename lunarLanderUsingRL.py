import gymnasium as gym  # Use gymnasium instead of gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import imageio
from itertools import product
from IPython.display import Video, display

# Set up the Lunar Lander environment with render_mode="rgb_array"
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# Define the Neural Network for the Q-Function
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Hyperparameters
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
lr = 1e-3
batch_size = 64
target_update_freq = 10
memory_size = 10000
num_episodes = 200

# Replay Memory
memory = deque(maxlen=memory_size)

# Initialize DQN and target network
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Optimizer and Loss
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
criterion = nn.MSELoss()

# Function to select an action (epsilon-greedy policy)
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            q_values = policy_net(state)
            return q_values.argmax().item()

# Function to train the DQN
def train_dqn():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = policy_net(states).gather(1, actions).squeeze()
    next_q_values = target_net(next_states).max(1)[0]
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    loss = criterion(q_values, target_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Function to plot metrics
def plot_metrics(metric_values, epochs, labels, title, ylabel, xlabel="Epoch", linestyle=None):
    plt.figure(figsize=(12, 6))
    for i, values in enumerate(metric_values):
        plt.plot(epochs, values, label=labels[i], marker='o', linestyle=linestyle[i] if linestyle else '-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to save video
def save_video(frames, filename="test_episode_video.mp4", fps=30):
    with imageio.get_writer(filename, fps=fps) as video:
        for frame in frames:
            video.append_data(frame)
    print(f"Video saved as {filename}")

# Hyperparameter Grid
param_grid = {
    "lr": [1e-3, 5e-4],
    "batch_size": [64, 128],
    "epsilon_decay": [0.995, 0.99],
}
param_combinations = list(product(*param_grid.values()))
grid_results = []

# Training Loop with Grid Search
for params in param_combinations:
    lr, batch_size, epsilon_decay = params
    print(f"Training with params: LR={lr}, Batch Size={batch_size}, Epsilon Decay={epsilon_decay}")

    memory = deque(maxlen=memory_size)
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    train_rewards = []
    steps_per_reward = []
    validation_rewards = []
    epsilon = 1.0

    for episode in range(num_episodes):
        state, _ = env.reset(seed=42 + episode)
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            steps += 1

            if len(memory) >= batch_size:
                train_dqn()

        train_rewards.append(total_reward)
        steps_per_reward.append((steps, total_reward))
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if episode % 10 == 0:
            validation_rewards.append(np.mean(train_rewards[-10:]))

    grid_results.append({
        "params": params,
        "train_rewards": train_rewards,
        "steps_per_reward": steps_per_reward,
        "validation_rewards": validation_rewards,
    })

    steps, rewards = zip(*steps_per_reward)
    # plot_metrics([steps, rewards], range(1, len(steps) + 1),
    #              ['Steps', 'Fitness Reward'],
    #              f"Steps vs Fitness Reward (LR={lr}, Batch={batch_size}, Epsilon Decay={epsilon_decay})",
    #              "Fitness Reward")
    #
    # plot_metrics([train_rewards, validation_rewards], range(1, len(train_rewards) + 1),
    #              ['Training Reward', 'Validation Reward'],
    #              f"Training and Validation Rewards (LR={lr}, Batch={batch_size}, Epsilon Decay={epsilon_decay})",
    #              "Reward")

# Plot Results for Each Combination using plot_metrics
for result in grid_results:
    params = result["params"]
    steps, rewards = zip(*result["steps_per_reward"])
    validation_rewards = result["validation_rewards"]
    train_rewards = result["train_rewards"]

    # Adjust epochs to match the lengths
    epochs_steps = range(1, len(steps) + 1)
    epochs_rewards = range(1, len(train_rewards) + 1)
    epochs_val_rewards = range(1, len(validation_rewards) + 1)


    # Plot steps vs fitness reward
    plot_metrics([steps, rewards], epochs_steps, ['Steps', 'Fitness Reward'],
                 f"Steps vs Fitness Reward (LR={params[0]}, Batch={params[1]}, Epsilon Decay={params[2]})",
                 "Fitness Reward")

    # Plot training  rewards
    plot_metrics([train_rewards], epochs_rewards,
                 ['Training Reward'],
                 f"Training and Validation Rewards (LR={params[0]}, Batch={params[1]}, Epsilon Decay={params[2]})",
                 "Reward")

    # Plot validation rewards
    plot_metrics([validation_rewards], epochs_val_rewards,
                 ['Training Reward', 'Validation Reward'],
                 f"Training and Validation Rewards (LR={params[0]}, Batch={params[1]}, Epsilon Decay={params[2]})",
                 "Reward")

# Test the best model
best_result = max(grid_results, key=lambda result: np.mean(result["validation_rewards"]))
best_params = best_result["params"]
best_steps_rewards = best_result["steps_per_reward"]
best_train_rewards = best_result["train_rewards"]
best_validation_rewards = best_result["validation_rewards"]

print(f"Best Model Parameters: LR={best_params[0]}, Batch Size={best_params[1]}, Epsilon Decay={best_params[2]}")

# best_steps, best_rewards = zip(*best_steps_rewards)
# plot_metrics([best_steps, best_rewards], range(1, len(best_steps) + 1),
#              ['Steps', 'Fitness Reward'],
#              f"Best Model: Steps vs Fitness Reward\n(LR={best_params[0]}, Batch Size={best_params[1]}, Epsilon Decay={best_params[2]})",
#              "Fitness Reward")
# plot_metrics([best_train_rewards, best_validation_rewards], range(1, len(best_train_rewards) + 1),
#              ['Training Reward', 'Validation Reward'],
#              f"Best Model: Training and Validation Rewards\n(LR={best_params[0]}, Batch Size={best_params[1]}, Epsilon Decay={best_params[2]})",
#              "Reward")


# Adjust epochs to match lengths
epochs_best_steps = range(1, len(best_steps_rewards) + 1)
epochs_best_train = range(1, len(best_train_rewards) + 1)
epochs_best_valid = range(1, len(best_validation_rewards) + 1)


# Plot Steps vs Fitness Reward for Best Model
best_steps, best_rewards = zip(*best_steps_rewards)
plot_metrics([best_steps, best_rewards], epochs_best_steps,
             ['Steps', 'Fitness Reward'],
             f"Best Model: Steps vs Fitness Reward\n(LR={best_params[0]}, Batch Size={best_params[1]}, Epsilon Decay={best_params[2]})",
             "Fitness Reward")

# Plot Training and Validation Rewards for Best Model
plot_metrics([best_train_rewards], epochs_best_train,
             ['Training Reward'],
             f"Best Model: Training and Validation Rewards\n(LR={best_params[0]}, Batch Size={best_params[1]}, Epsilon Decay={best_params[2]})",
             "Reward")

plot_metrics([best_validation_rewards], epochs_best_valid,
             ['Validation Reward'],
             f"Best Model: Training and Validation Rewards\n(LR={best_params[0]}, Batch Size={best_params[1]}, Epsilon Decay={best_params[2]})",
             "Reward")


# Test phase with videos
num_test_episodes = 10
test_rewards = []
test_steps = []

best_model = DQN(state_dim, action_dim)
best_model.load_state_dict(policy_net.state_dict())
best_model.eval()

for test_episode in range(num_test_episodes):
    state, _ = env.reset(seed=2024 + test_episode)
    total_reward = 0
    steps = 0
    done = False
    frames = []

    while not done:
        frame = env.render()
        frames.append(frame)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = best_model(state_tensor).argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        steps += 1

    test_rewards.append(total_reward)
    test_steps.append(steps)
    save_video(frames, filename=f"test_episode_{test_episode + 1}.mp4")
    print(f"Test Episode {test_episode + 1}: Total Reward: {total_reward:.2f}, Steps: {steps}")

# Plot test rewards vs. episodes
plot_metrics(
    [test_rewards],
    range(1, num_test_episodes + 1),
    ["Test Rewards"],
    "Test Rewards vs Episodes",
    ylabel="Rewards",
    xlabel="Episode"
)

# Plot test steps vs. episodes
plot_metrics(
    [test_steps],
    range(1, num_test_episodes + 1),
    ["Test Steps"],
    "Test Steps vs Episodes",
    ylabel="Steps",
    xlabel="Episode"
)

# Scatter plot for steps per reward during test episodes
plt.figure(figsize=(12, 6))
plt.scatter(test_rewards, test_steps, color="blue", label="Steps per Reward", marker="o")
plt.xlabel("Test Rewards")
plt.ylabel("Steps")
plt.title("Steps vs Rewards (Test Phase)")
plt.grid()
plt.legend()
plt.show()

# Calculate and print average metrics
avg_test_reward = np.mean(test_rewards)
avg_test_steps = np.mean(test_steps)
print(f"Average Test Reward: {avg_test_reward:.2f}")
print(f"Average Test Steps: {avg_test_steps:.2f}")


env.close()
video_filenames = [f"test_episode_{i + 1}.mp4" for i in range(num_test_episodes)]
for video_filename in video_filenames:
    display(Video(video_filename, embed=True))
