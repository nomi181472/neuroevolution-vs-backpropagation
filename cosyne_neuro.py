import os.path
import copy
import torch
from torch import nn
from evotorch.decorators import pass_info
from evotorch.neuroevolution import GymNE
from evotorch.logging import PandasLogger, StdOutLogger
#from evotorch.logging import StdOutLogger
from custom_logger import CSVLogger
from evotorch.algorithms import Cosyne,PGPE


import gymnasium as gym
import  pandas as pd

file= "metrics_rl.csv"
def get_id_value(epoch, parameters, size,algorithm,num_of_l):
    return f"epoch:{epoch},parameters:{parameters},pop_size:{size},algorithm:{algorithm},num_of_layer:{num_of_l}"

def get_df(file,columns):
    df=None
    if os.path.exists(file):
        df = pd.read_csv(file, index_col=0)

    else:
        df = pd.DataFrame(columns=columns)
        df.to_csv(file)
    return df


columns=["Id","epoch",
         "parameters","population_size",
         "algorithm",
         "num_of_layer"
         "best_eval",
         "mean_eval","median_eval",
         "pop_best_eval",
         ]
df=get_df(file,columns)
seed=42

def get_df(file,columns):
    df=None
    if os.path.exists(file):
        df = pd.read_csv(file, index_col=0)

    else:
        df = pd.DataFrame(columns=columns)
        df.to_csv(file)
    return

global neurons, num_of_layers,pop_size,algorithm,epochs,epoch,parameters
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
num_of_layers = 1

def check_and_create(path):
    print(f"creating:{path}")
    if not os.path.exists(path):
        os.mkdir(path)

env_name = 'Swimmer-v5'
check_and_create("data")
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
        env = gym.make(env_name, render_mode="rgb_array",)

        # Wrap the environment with RecordVideo
        path_video=f"{folder_name}/videos/{algorithm}/{num_of_layers}/{pop_size}/{parameters}/"
        env = gym.wrappers.RecordVideo(env, path_video, episode_trigger=lambda x: x % 100 == 0)
        return env
    return GymNE(

        env=create_env,  # ame of the environment
        network=LinearPolicy,  # Linear policy that we defined earlier
        network_args={'bias': False},  # Linear policy should not use biases
        num_actors='max',  # Use 4 available CPUs. You can modify this value, or use 'max' to exploit all available CPUs
        observation_normalization=True,  # Observation normalization was not used in Lunar Lander experiments
    )


def get_searcher(problem,algo,p_size, **kwargs):
    if algo=="Cosyne":
        return Cosyne(

                problem,
                popsize=p_size,
                **kwargs
            )
    elif algo=="PGPE":
        return PGPE(
            problem,
            popsize=p_size,
            center_learning_rate=0.01125,
            stdev_learning_rate=0.1,
            optimizer_config={"max_speed": 0.015},
            radius_init=0.27,
            num_interactions=150000,
            popsize_max=3200,
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
            file_name=f"{path}/iter_{iter}_score_{current_score}_{algorithm}{num_of_layers}_{pop_size}_{parameters}.pth"
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
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_metrics():

    pass
def cal_score():
    global searcher, val_loader, \
        train_loader, compute_metrics,\
        get_id_value,count_parameters,pop_size,algorithm,num_of_layers


    epoch= copy.deepcopy(searcher.status["iter"])
    loss = copy.deepcopy(searcher.status["best"])


    # Calculate metrics for the training set

    id={"Id":get_id_value(epoch,parameters,pop_size,algorithm,num_of_layers),
        "population_size":pop_size,"parameters":parameters,
        "loss":loss,
        "algorithm":algorithm,
        "num_of_layer":num_of_layers,
        "epoch":epoch}


    # Combine train and validation metrics
    metrics = {**id}
    return metrics

neuron_list=[10]
num_of_layers_list=[2]
pop_sizes=[100]
algorithms=["Cosyne",]
total_epochs=1000
epoch=0
save_weights_after_iter=20

for neurons in neuron_list:
    for num_of_layers in num_of_layers_list:
        for pop_size in pop_sizes:
            for algorithm in algorithms:
                network=LinearPolicy(8,4)
                parameters = count_parameters(network)

                id = get_id_value(total_epochs, parameters, pop_size, algorithm,num_of_layers)
                if ((df["Id"] == id)).any():
                    print(f"already done:{id}")
                    continue

                # Setting up the GymNE problem
                problem = get_problem(env_name)



                searcher = get_searcher(problem,algorithm,pop_size,
                       ** {
                            "num_elites": 1,

                            "tournament_size": 10,
                            "mutation_stdev": 0.3,
                            "mutation_probability": 0.5,
                            "permute_all": True,
                        })

                searcher.after_step_hook.append(cal_score)
                searcher.after_step_hook.append(save_metrics)
                searcher.before_step_hook.append(check_points)

                CSVLogger(searcher)
                # StdOutLogger(searcher)
                searcher.run(total_epochs)
                # new_df = pandas_logger.to_dataframe()
                # df = pd.concat([df, new_df[columns]], ignore_index=True)
                # print(f"done:{id}")
                # df.to_csv(file)
                # problem.kill_actors()
