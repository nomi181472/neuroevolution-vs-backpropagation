import gymnasium as gym
import torch
from torch import nn
from evotorch.decorators import pass_info
from evotorch.neuroevolution import GymNE
#from evotorch.logging import StdOutLogger
from custom_logger import CSVLogger
from evotorch.algorithms import Cosyne
env_name='LunarLander-v3'
test_env = gym.make(env_name,render_mode='human')


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
num_of_layers = 2


actions=4
print(f"actions:{actions}")
policy = LinearPolicy(obs_length=test_env.observation_space.shape[0], act_length= actions,bias=False)

problem = GymNE(
    env=env_name,  # ame of the environment
    network=LinearPolicy,  # Linear policy that we defined earlier
    network_args={'bias': False},  # Linear policy should not use biases
    num_actors=4,  # Use 4 available CPUs. You can modify this value, or use 'max' to exploit all available CPUs
    observation_normalization=False,
    # Observation normalization was not used in Lunar Lander experiments,render_mode='human'
)
path="data/LunarLander-v3/weights/mean_eval/iter_765_score_280.3769226074219.pth"
best_weights = torch.load(path, weights_only=True)
print('evaluating')
policy.load_state_dict(best_weights)
for i in range(10):
    print(problem.visualize(policy=policy, num_episodes=1))
