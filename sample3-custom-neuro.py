from src.evotorch.algorithms import PGPE
from src.evotorch.logging import StdOutLogger
from src.evotorch.neuroevolution import GymNE
import os
import gymnasium as gym
# Specialized Problem class for RL
# Function to evaluate the policy and save video with iteration number
def evaluate_and_record(policy, env_name, save_path, iteration):
    # Create a unique directory for each iteration
    iteration_path = os.path.join(save_path, f"iteration_{iteration}")
    os.makedirs(iteration_path, exist_ok=True)

    # Wrap the environment with RecordVideo
    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, iteration_path, episode_trigger=lambda _: True)  # Always save this episode
    obs,_ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = policy(obs)
        obs, reward, done, info,_ = env.step(action)
        total_reward += reward

    env.close()
    return total_reward
    
def get_problem(env_name,path_video,isRecord=False):
    def create_env():
        env:gym.Env
        if isRecord:
            env=gym.make(env_name)
        else:
            env = gym.make(env_name, render_mode="rgb_array",)
            # Wrap the environment with RecordVideo
            env = gym.wrappers.RecordVideo(env, path_video, episode_trigger=lambda x: x % 100 == 0)
        return env
    return GymNE(

        env=create_env,  # ame of the environment
        network="Linear(obs_length, act_length)",  # Linear policy that we defined earlier
        network_args={'bias': False},  # Linear policy should not use biases
        num_actors='max',  # Use 4 available CPUs. You can modify this value, or use 'max' to exploit all available CPUs
        observation_normalization=True,  # Observation normalization was not used in Lunar Lander experiments
    )

problem =get_problem("Humanoid-v4","./data")

searcher = PGPE(
    problem,
    popsize=200,
    center_learning_rate=0.01125,
    stdev_learning_rate=0.1,
    optimizer_config={"max_speed": 0.015},
    radius_init=0.27,
    num_interactions=150000,
    popsize_max=3200,
)
logger = StdOutLogger(searcher)
searcher.run(10)

population_center = searcher.status["center"]
policy = problem.to_policy(population_center)
problem.visualize(policy)