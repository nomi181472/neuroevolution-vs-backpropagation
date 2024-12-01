from src.evotorch import operators
import torch
from copy import deepcopy
from torch import nn
from src.evotorch.decorators import pass_info
from src.evotorch.neuroevolution import GymNE
from src.evotorch.logging import StdOutLogger
from src.evotorch.algorithms import GeneticAlgorithm

from evotorch.core import Problem, SolutionBatch
class CustomMutationOperator(operators.CopyingOperator):
    def __init__(self, problem: Problem, mean: float = 0.0, std_dev: float = 0.1):
        super().__init__(problem,)
        self.mean = mean
        self.std_dev = std_dev
        self.prob:Problem=problem

    @torch.no_grad()
    def _do(self, solutions: SolutionBatch) -> SolutionBatch:


        result = deepcopy(solutions)
        data = result.access_values()
        noise = torch.normal(mean=self.mean, std=self.std_dev, size=solutions.values.shape)
        data.add_(noise)
        data[:] = self._respect_bounds(data)
        self.prob.evaluate(result)

        return result







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

searcher = GeneticAlgorithm(
    problem,
    popsize=200, # Static population size
    operators=[
CustomMutationOperator(problem,0.1,0.1),
        operators.TwoPointCrossOver(problem=problem, tournament_size=10),
    operators.GaussianMutation(problem=problem,stdev=0.1,mutation_probability=0.1),
    ]
   # radius_init=radius_init,  # Initial radius
   # center_learning_rate=center_learning_rate,
   # stdev_learning_rate=0.1,  # Stdev learning rate of 0.1
   # optimizer="clipup",  # Using the ClipUp optimiser
   # optimizer_config={
   #     'max_speed': max_speed,  # Defined max speed
   #     'momentum': 0.9,  # Momentum fixed to 0.9
   # }
)

# Logging and running the searcher
StdOutLogger(searcher)
searcher.run(50)
