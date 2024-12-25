import torch
import torch.nn as nn


class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        # Store the models
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]

        to_return = x
        map = dict()
        for idx, y_hat in enumerate(outputs):

            action = torch.argmax(y_hat).item()
            if action in map:
                map[action] = (idx, map[action][-1] + 1)
            else:
                map[action] = (idx, 1)
                # Find the action with the highest count (i.e., most votes)
        highest_same_action = max(map.items(), key=lambda x: x[1][-1])  # Get the action with max votes

        # Get the model index with the most votes for the action
        highest_same_action_idx = highest_same_action[1][0]
        return outputs[highest_same_action_idx]

        # Get predictions from each model



