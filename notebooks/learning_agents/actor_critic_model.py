import numpy as np
import torch

class ApprovalRecurrentAgent(torch.nn.Module):
    def __init__(self, config:dict, num_players, obs_size=None):
        super().__init__()

        # recurrent layer
        # TODO: Do I want 2 here?
        self.recurrent_layer = self._rec_layer_init(torch.nn.LSTM(obs_size, config['rec_hidden_size'], num_layers=config['rec_layers'], batch_first=True))

        # hidden layers
        self.fc_joint = self._layer_init(torch.nn.Linear(config['rec_hidden_size'], config['hidden_mlp_size']))
        self.policy_hidden = self._layer_init(torch.nn.Linear(config['hidden_mlp_size'], config['hidden_mlp_size']))
        self.value_hidden = self._layer_init(torch.nn.Linear(config['hidden_mlp_size'], config['hidden_mlp_size']))

        # policy output
        self.policies_out = torch.nn.ModuleList()
        for _ in range(num_players):
            actor_branch = self._layer_init(torch.nn.Linear(in_features=config['hidden_mlp_size'], out_features=config['approval_states']), std=0.01)
            self.policies_out.append(actor_branch)
        # value output
        self.value_out = self._layer_init(torch.nn.Linear(config['hidden_mlp_size'], 1), std=1.0)

        self.num_players = num_players

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
        policies = [torch.distributions.Categorical(logits=branch(h_policy)) for branch in self.policies_out]

        return policies, value, recurrent_cell
    
    def _convert_policy_action_to_game_action(self, policy_action):
        """
        0 -> -1 
        1 -> 0
        2 -> 1

        We will just subtract 1 from the items as the predicted classes are 0,1,2
        """
        return torch.tensor(policy_action) - 1
    
    def get_action_from_policies(self, policies):
        policy_action = [policy.sample() for policy in policies]
        game_action = self._convert_policy_action_to_game_action(policy_action)

        return policy_action, game_action