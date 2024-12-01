import evotorch.operators
import torch
from torch import nn
from evotorch.decorators import pass_info
from evotorch.neuroevolution import GymNE
from evotorch.logging import StdOutLogger
from evotorch.algorithms import Cosyne

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
        self.linear = nn.Linear(obs_length, act_length, bias=bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Forward pass of model simply applies linear layer to observations
        return self.linear(obs)

# Setting up the GymNE problem
problem = GymNE(
    env="LunarLanderContinuous-v3",  # Name of the environment
    network=LinearPolicy,  # Linear policy that we defined earlier
    network_args={'bias': False},  # Linear policy should not use biases
    num_actors=4,  # Use 4 available CPUs. You can modify this value, or use 'max' to exploit all available CPUs
    observation_normalization=False,  # Observation normalization was not used in Lunar Lander experiments
)

# Setting up the PGPE searcher
radius_init = 4.5  # (approximate) radius of initial hypersphere that we will sample from
max_speed = radius_init / 15.  # Rule-of-thumb from the paper
center_learning_rate = max_speed / 2.

searcher = Cosyne(
    problem,
    num_elites=1,
    popsize=50,
    tournament_size=4,
    mutation_stdev=0.3,
    mutation_probability=0.5,
    permute_all=True,
)


StdOutLogger(searcher)
searcher.run(50)
