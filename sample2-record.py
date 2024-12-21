from src.evotorch.algorithms import PGPE
from src.evotorch.logging import StdOutLogger
from src.evotorch.neuroevolution import GymNE

import gymnasium as gym
# Specialized Problem class for RL

def get_problem(env_name,path_video):
    def create_env():
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

problem = GymNE(
    env_name="Humanoid-v4",
    # Linear policy
    network="Linear(obs_length, act_length)",
    observation_normalization=True,
    decrease_rewards_by=5.0,
    # Use all available CPU cores
    num_actors="max",
)

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