import numpy as np
import torch


class SequentialAgent(torch.nn.Module):
    def __init__(self, approval_states, num_players, obs_size):
        super().__init__()

        self.critic = torch.nn.Sequential(
            self._layer_init(torch.nn.Linear(obs_size+1, 256)),
            torch.nn.Tanh(),
            self._layer_init(torch.nn.Linear(256,512)),
            torch.nn.Tanh(),
            self._layer_init(torch.nn.Linear(512,256)),
            torch.nn.Tanh(),
            self._layer_init(torch.nn.Linear(256,1), std=1.0),
        )

        self.actor = torch.nn.Sequential(
            self._layer_init(torch.nn.Linear(obs_size+1, 256)),
            torch.nn.Tanh(),
            self._layer_init(torch.nn.Linear(256,512)),
            torch.nn.Tanh(),
            self._layer_init(torch.nn.Linear(512,256)),
            torch.nn.Tanh(),
            self._layer_init(torch.nn.Linear(256, approval_states), std=0.01),
        )

        self.num_players = num_players
    
    def get_value(self, x):
        # TODO: We need torch.mean because PPO will use value, and we have a bunch here. 
        #       Do we need to change PPO here?
        return torch.mean(self.critic(torch.stack([torch.cat((torch.tensor([i]), x)) for i in range(self.num_players)])))
    
    # only doing this for the PPO batched call so I don't need extra logic in the regular get action and value
    def get_batched_action_and_value(self, x, actions=None):

        if actions is None:
            raise ValueError("We need batched actions here")

        log_probs = []
        entropies = []
        critics = []
        for current_obs, action in zip(x, actions):
            updated_obs = torch.stack([torch.cat((torch.tensor([i]), current_obs)) for i in range(self.num_players)])

            logits = self.actor(updated_obs)
            probs = torch.distributions.categorical.Categorical(logits=logits)
            
            # update our return tensors
            log_probs.append(torch.sum(probs.log_prob(action)))
            entropies.append(torch.prod(probs.entropy()))
            critics.append(torch.mean(self.critic(updated_obs)))
            
        return actions, torch.stack(log_probs), torch.stack(entropies), torch.stack(critics)

    def convert_actions_to_approvals(self, actions):
        return [-1 if a == 2 else a.item() for a in actions]

    def get_action_and_value(self, x, action=None):
        # could call the network each time, with a different integer for each player?  get approvals that way
        # x is the flattened observation. we should go ahead and run each of the player_ids appended to full obs to get multiple classifications
        # how  to handle entropy here? maybe we multiply all the probs, and then calculate the overall entropy
        # self.critic needs to be changed too, to return an array

        # option to have critic/actors for every single player?

        # option to also delevt n-1 * n-2 for -1s on the wolf
        
        # get logits for every single player in the game.
        x = torch.stack([torch.cat((torch.tensor([i]), x)) for i in range(self.num_players)])
        logits = self.actor(x)
        probs = torch.distributions.categorical.Categorical(logits=logits)

        if action is None:
            action = probs.sample()
        
        # we multiply the entropy, and we add the log_probs together
        # TODO: multiple values for critic. should I average?
        return action, torch.sum(probs.log_prob(action)), torch.prod(probs.entropy()), torch.mean(self.critic(x))
    

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
