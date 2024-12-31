from evotorch.decorators import pass_info
import torch.nn as nn
@pass_info
class LinearPolicy(nn.Module):
    def __init__(self, obs_length, act_length, num_layers=2, neurons=8, bias=True, **kwargs):
        super().__init__()

        layers = []
        input_size = obs_length

        if num_layers == 1:
            layers.append(nn.Linear(input_size, act_length, bias=bias))
        else:
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(input_size, neurons, bias=bias))
                layers.append(nn.ReLU())
                input_size = neurons

            layers.append(nn.Linear(input_size, act_length, bias=bias))

        self.layers = nn.ModuleList(layers)
    def calculate_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    def forward(self, obs):
        x = obs
        for layer in self.layers:
            x = layer(x)
        return x
