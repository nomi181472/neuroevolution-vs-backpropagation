import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from evotorch import Problem
from evotorch.algorithms import CMAES
from evotorch.neuroevolution import GymNE
from evotorch.logging import  StdOutLogger
# Define the environment
class PointMazeEnv(gym.Env):
    def __init__(self):
        super(PointMazeEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        self.goal = np.array([9.0, 9.0], dtype=np.float32)
        self.max_steps = 100
        self.current_step = 0

    def reset(self,options,seed=3):
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        self.current_step = 0
        return self.state

    def step(self, action):
        self.state = np.clip(self.state + action, 0, 10)
        distance_to_goal = np.linalg.norm(self.state - self.goal)
        reward = -distance_to_goal
        self.current_step += 1
        done = distance_to_goal < 0.1 or self.current_step >= self.max_steps
        return self.state, reward, done, {}

    def render(self, mode="human"):
        print(f"State: {self.state}")

    def close(self):
        pass

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Tanh to keep actions within [-1, 1]
        return x

# Register the environment
env_name = "PointMazeEnv-v0"
gym.envs.registration.register(id=env_name, entry_point=PointMazeEnv)

# Define the problem
input_size = 2
output_size = 2
network = PolicyNetwork(input_size, output_size)

problem = GymNE(
    env_name,
    network=network,
    num_actors=1,

)

# Define the optimizer
searcher = CMAES(problem, popsize=50,stdev_init=0.1)

StdOutLogger(searcher)
# Run the optimization
searcher.run(100)

# Print the best solution
print(f"Best solution: {searcher.status['best']}")
