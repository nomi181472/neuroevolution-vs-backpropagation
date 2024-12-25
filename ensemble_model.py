import torch
import torch.nn as nn


class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        # Store the models
        self.models = nn.ModuleList(models)

    def forward(self, x):
        # Get predictions from each model
        outputs = [model(x) for model in self.models]

        # Apply softmax to get probabilities (assuming the models are classification models)
        outputs = [torch.softmax(output, dim=0) for output in outputs]

        # Get predicted class labels for each model
        preds = [output.argmax(dim=0) for output in outputs]

        # Stack the predictions to create a tensor of shape (num_models, batch_size)
        preds = torch.stack(preds, dim=0)

        # Perform majority voting along the model dimension
        majority_preds = torch.mode(preds, dim=0)[0]  # Mode along the model dimension

        return majority_preds