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

from src.evotorch import SolutionBatch


# The decorator `@pass_info` tells the problem class `GymNE`
# to pass information regarding the gym environment via keyword arguments
# such as `obs_length` and `act_length`.


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
        self.linear1 = nn.Linear(obs_length, obs_length, bias=bias)
        self.linear2 = nn.Linear(obs_length, act_length, bias=bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Forward pass of model simply applies linear layer to observations
        return self.linear2(self.linear1(obs))

#env = gym.wrappers.RecordVideo(env, "videos/", episode_trigger=lambda x: x % 10 == 0)

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
    return GymNE(

        env=env_name,  # ame of the environment
        network=LinearPolicy,  # Linear policy that we defined earlier
        network_args={'bias': False},  # Linear policy should not use biases
        num_actors=4,  # Use 4 available CPUs. You can modify this value, or use 'max' to exploit all available CPUs
        observation_normalization=False,  # Observation normalization was not used in Lunar Lander experiments
    )


# Setting up the GymNE problem
problem = get_problem(env_name)


def get_searcher(problem, **kwargs):
    return Cosyne(

        problem,
        **kwargs
    )

save_weights_after_iter=300
searcher = get_searcher(problem,
                       ** {
                            "num_elites": 1,
                            "popsize": 200,
                            "tournament_size": 10,
                            "mutation_stdev": 0.3,
                            "mutation_probability": 0.5,
                            "permute_all": True,
                        })

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


            solution = searcher.status[sol_name]
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



searcher.after_step_hook.append(save_metrics)
searcher.before_step_hook.append(check_points)

CSVLogger(searcher)
searcher.run(1000)
pop_best = searcher.status['pop_best']
best_weights = problem.parameterize_net(pop_best.access_values())
torch.save(best_weights.state_dict(), f"{weights_path}/pop_best.pth")

best = searcher.status['best']
best_weights = problem.parameterize_net(best.access_values())
torch.save(best_weights.state_dict(), f"{weights_path}/best.pth")

problem = GymNE(
    env=env_name,  # ame of the environment
    network=LinearPolicy,  # Linear policy that we defined earlier
    network_args={'bias': False},  # Linear policy should not use biases
    num_actors=4,  # Use 4 available CPUs. You can modify this value, or use 'max' to exploit all available CPUs
    observation_normalization=False,
    # Observation normalization was not used in Lunar Lander experiments,render_mode='human'
)

print('evaluating')
for i in range(100):
    best_weights = None
    if i % 2 == 0:
        best_weights = searcher.status['best']
        print('best')
    else:
        best_weights = searcher.status['pop_best']
        print("pop_best")
    print(problem.visualize(policy=problem.parameterize_net(best_weights), num_episodes=1))
