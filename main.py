import torch
from torch import nn
from evotorch.decorators import pass_info
from evotorch.neuroevolution import GymNE
from evotorch.logging import StdOutLogger
from evotorch.algorithms import PGPE

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

searcher = PGPE(
    problem,
    popsize=200,  # Static population size
    radius_init=radius_init,  # Initial radius
    center_learning_rate=center_learning_rate,
    stdev_learning_rate=0.1,  # Stdev learning rate of 0.1
    optimizer="clipup",  # Using the ClipUp optimiser
    optimizer_config={
        'max_speed': max_speed,  # Defined max speed
        'momentum': 0.9,  # Momentum fixed to 0.9
    }
)

# Logging and running the searcher
StdOutLogger(searcher)
searcher.run(50)

# Visualizing the trained policy
#center_solution = searcher.status["center"]  # Get mu
#policy_net = problem.to_policy(center_solution)  # Instantiate a policy from mu

# Save the weights of the trained policy network
#torch.save(policy_net.state_dict(), 'trained_policy_weights.pth')
#print('Weights of the trained policy network have been saved to "trained_policy_weights.pth".')


# Create a new policy network and load the saved weights
#loaded_policy_net = LinearPolicy(obs_length=problem.get_observation_stats(), act_length=4, bias=False)
#loaded_policy_net.load_state_dict(torch.load('trained_policy_weights.pth'))
#print('Loaded the trained policy network weights from "trained_policy_weights.pth".')

# Visualize 10 episodes with the loaded policy network
#for _ in range(10):
#    result = problem.visualize(loaded_policy_net)
#    print('Visualised episode has cumulative reward:', result['cumulative_reward'])
