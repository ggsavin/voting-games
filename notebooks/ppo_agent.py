
import torch
import numpy as np

class Agent(torch.nn.Module):
    def __init__(self, num_actions, obs_size):
        super().__init__()

        self.critic = torch.nn.Sequential(
            self._layer_init(torch.nn.Linear(obs_size, 64)),
            torch.nn.Tanh(),
            self._layer_init(torch.nn.Linear(64,64)),
            torch.nn.Tanh(),
            self._layer_init(torch.nn.Linear(64,1), std=1.0),
        )

        self.actor = torch.nn.Sequential(
            self._layer_init(torch.nn.Linear(obs_size, 64)),
            torch.nn.Tanh(),
            self._layer_init(torch.nn.Linear(64,64)),
            torch.nn.Tanh(),
            self._layer_init(torch.nn.Linear(64, num_actions), std=0.01),
        )
    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)

        probs = torch.distributions.categorical.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)
    obs = torch.tensor(obs).to(device)

def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x

def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x
