from src.evotorch.algorithms import Cosyne
from src.evotorch.logging import StdOutLogger
from src.evotorch.neuroevolution import GymNE

import gymnasium as gym
# Specialized Problem class for RL
print("video records")
def get_problem(env_name,path_video):
    def create_env():
        env = gym.make(env_name, render_mode="rgb_array",)

        # Wrap the environment with RecordVideo
        print("Configuring record video wrapper")
        env = gym.wrappers.RecordVideo(env,video_folder=path_video, episode_trigger=lambda x:True)
        return env
    return GymNE(

        env=create_env,  # ame of the environment
        network="Linear(obs_length, act_length)",  # Linear policy that we defined earlier
        network_args={'bias': False},  # Linear policy should not use biases
        num_actors='max',  # Use 4 available CPUs. You can modify this value, or use 'max' to exploit all available CPUs
        observation_normalization=True,  # Observation normalization was not used in Lunar Lander experiments
    )

problem =get_problem("LunarLander-v3","./data2")




print("problem is created")
searcher = Cosyne(

                problem,
                popsize=6,
    **{
        "num_elites": 1,

        "tournament_size": 10,
        "mutation_stdev": 0.3,
        "mutation_probability": 0.5,
        "permute_all": True,
    }
            )
logger = StdOutLogger(searcher)
print("Algorithm added")
searcher.run(1000)
print("Iteration completed")
population_center = searcher.status["best"]
policy = problem.to_policy(population_center)
