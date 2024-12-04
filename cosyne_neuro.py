import os.path
import copy
import torch
from torch import nn
from evotorch.decorators import pass_info
from evotorch.neuroevolution import GymNE
#from evotorch.logging import StdOutLogger
from custom_logger import CSVLogger
from evotorch.algorithms import Cosyne
import gymnasium as gym
import  pandas as pd

file= "metrics_rl.csv"

columns=["Id","epoch",
         "parameters","population_size", "algorithm"
         "best_eval",
         "mean_eval","median_eval",
         "pop_best_eval",

                                   ]

def get_df(file,columns):
    df=None
    if os.path.exists(file):
        df = pd.read_csv(file, index_col=0)

    else:
        df = pd.DataFrame(columns=columns)
        df.to_csv(file)
    return df

global neurons, num_of_layers
@pass_info
class LinearPolicy(nn.Module):
    def __init__(
            self,
            obs_length: int,  # Number of observations from the environment
            act_length: int,  # Number of actions of the environment
            bias: bool = True,  # Whether the policy should use biases
            **kwargs  # Anything else that is passed
    ):
        super().__init__()  # Always call super init for nn Modules

        layers = []
        input_size = obs_length

        if num_of_layers == 1:
            # Directly map input to output without hidden layers
            layers.append(nn.Linear(input_size, act_length, bias=bias))
        else:
            # Add hidden layers
            for _ in range(num_of_layers - 1):
                layers.append(nn.Linear(input_size, neurons, bias=bias))
                layers.append(nn.ReLU())
                input_size = neurons

            # Add the final output layer
            layers.append(nn.Linear(input_size, act_length, bias=bias))

        # Register layers as a ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Pass through each layer
        x = obs
        for layer in self.layers:
            x = layer(x)
        return x

# Example usage
neurons = 10
num_of_layers = 10

def check_and_create(path):
    if not os.path.exists(path):
        os.mkdir(path)

env_name = 'LunarLander-v3'

folder_name = f"./data/{env_name}"
check_and_create(folder_name)
weights_path = f"{folder_name}/weights"

metrics={
    "best_eval":-100000,
    "worst_eval":100000000,
    "iter":1,
    "median_eval":-100000,
    "pop_best_eval":-100000,
    "mean_eval":-10000
}





check_and_create(weights_path)




def get_problem(env_name):
    def create_env():
        env = gym.make(env_name, render_mode="rgb_array")
        # Wrap the environment with RecordVideo
        env = gym.wrappers.RecordVideo(env, f"{folder_name}/videos", episode_trigger=lambda x: x % 100 == 0)
        return env
    return GymNE(

        env=create_env,  # ame of the environment
        network=LinearPolicy,  # Linear policy that we defined earlier
        network_args={'bias': False},  # Linear policy should not use biases
        num_actors=4,  # Use 4 available CPUs. You can modify this value, or use 'max' to exploit all available CPUs
        observation_normalization=False,  # Observation normalization was not used in Lunar Lander experiments
    )


def get_searcher(problem, **kwargs):
    return Cosyne(

        problem,
        **kwargs
    )

def check_points():

    global searcher,metrics,problem,weights_path,save_weights_after_iter,check_and_create


    current_status=copy.deepcopy(searcher.status)
    if not "iter" in current_status:
        return
    iter=current_status['iter']
    if not iter >=save_weights_after_iter:
        return


    def save_weights_based_on_metrics(metric_name, sol_name):
        if metric_name in current_status and current_status[metric_name] >= metrics[metric_name]:
            current_score = current_status[metric_name]


            solution = current_status[sol_name]
            best_policy = problem.parameterize_net(solution.access_values())
            path=f"{weights_path}/{metric_name}"
            check_and_create(path)
            file_name=f"{path}/iter_{iter}_score_{current_score}.pth"
            torch.save(best_policy.state_dict(), file_name)
            metrics[metric_name] = current_score
            print(f"saved {file_name}")

    m_name = "pop_best_eval"
    s_name = "pop_best"
    save_weights_based_on_metrics( m_name, s_name)
    m_name = "best_eval"
    s_name = "best"
    save_weights_based_on_metrics( m_name, s_name)
    m_name = "median_eval"
    s_name = "pop_best"
    save_weights_based_on_metrics( m_name, s_name)
    m_name = "mean_eval"
    s_name = "pop_best"
    save_weights_based_on_metrics( m_name, s_name)

def save_metrics():

    pass


# Setting up the GymNE problem
problem = get_problem(env_name)


save_weights_after_iter=20
searcher = get_searcher(problem,
                       ** {
                            "num_elites": 1,
                            "popsize": 50,
                            "tournament_size": 10,
                            "mutation_stdev": 0.3,
                            "mutation_probability": 0.5,
                            "permute_all": True,
                        })



searcher.after_step_hook.append(save_metrics)
searcher.before_step_hook.append(check_points)

CSVLogger(searcher)
searcher.run(1000)
