import numpy as np
import torch

class PluralityRecurrentAgent(torch.nn.Module):
    def __init__(self, config:dict, num_actions, obs_size=None):
        super().__init__()

        # recurrent layer
        # TODO: Do I want 2 here?
        self.recurrent_layer = self._rec_layer_init(torch.nn.LSTM(obs_size, config['rec_hidden_size'], num_layers=config['rec_layers'], batch_first=True))

        # hidden layers
        self.fc_joint = self._layer_init(torch.nn.Linear(config['rec_hidden_size'], config['hidden_mlp_size']))
        self.policy_hidden = self._layer_init(torch.nn.Linear(config['hidden_mlp_size'], config['hidden_mlp_size']))
        self.value_hidden = self._layer_init(torch.nn.Linear(config['hidden_mlp_size'], config['hidden_mlp_size']))

        # policy output
        self.policy_out = self._layer_init(torch.nn.Linear(config['hidden_mlp_size'], num_actions), std=0.01)

        # value output
        self.value_out = self._layer_init(torch.nn.Linear(config['hidden_mlp_size'], 1), std=1.0)
    
    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        # torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def _rec_layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        for name, param in layer.named_parameters():
            # if "bias" in name:
                # torch.nn.init.constant_(param, bias_const)
            if "weight" in name:
                torch.nn.init.orthogonal_(param, std)
        return layer
    
    
    def forward(self, x, recurrent_cell: torch.tensor):

        # pass  through the Recurrence Layer
        h, recurrent_cell = self.recurrent_layer(torch.unsqueeze(x,1), recurrent_cell)
        h = torch.squeeze(h,1)

        # Pass through a hidden layer
        h = torch.relu(self.fc_joint(h))

        # Split for Value and Policy
        h_value = torch.relu(self.value_hidden(h))
        h_policy = torch.relu(self.policy_hidden(h))

        # value
        value = self.value_out(h_value).reshape(-1)

        # policy
        policy = self.policy_out(h_policy)
        policy = torch.distributions.Categorical(logits=policy)

        return policy, value, recurrent_cell