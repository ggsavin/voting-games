import torch
import numpy as np

class RecurrentRolloutBuffer():
    
    def __init__(self, buffer_size: int, gamma: float, gae_lambda: float):
        '''
            @bufffer_size: This is the number of trajectories
        '''
        
        self.rewards = None
        self.actions = None
        self.dones = None
        self.observations = None

        # do we want these for both actor and critic?
        self.hcxs = None 

        self.log_probs = None
        self.values = None
        self.advantages = None

        self.buffer_size = buffer_size
        self.gamma = gamma 
        self.gae_lambda = gae_lambda

        self.reset(gamma=gamma, gae_lambda=gae_lambda)

    def reset(self, gamma: float, gae_lambda: float):
        self.rewards = []
        self.actions = []
        self.dones = []
        self.observations = []

        # do we want these for both actor and critic?
        self.hcxs = []

        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []

        self.gamma = gamma 
        self.gae_lambda = gae_lambda

    def add_replay(self, game) -> bool:
         
         self.rewards.append(game['rewards'])
         self.actions.append(game['actions'])
         self.dones.append(game["terms"])
         self.observations.append(game["obs"])
         self.log_probs.append(game["logprobs"])
         self.values.append(game["values"])
         self.hcxs.append(game["hcxs"][:-1])
        
         advantages, returns = self._calculate_advantages(game)
             
         self.advantages.append(advantages)
         self.returns.append(returns)

         return True
    
    @torch.no_grad()
    def _calculate_advantages(self, game):
        """Generalized advantage estimation (GAE)
        """
        advantages = torch.zeros_like(torch.tensor(game['rewards']))

        for t in reversed(range(len(game['rewards']))):
             delta = game['rewards'][t] + self.gamma * game['values'][max((t+1)%len(game['rewards']),t)] - game['values'][t]
             advantages[t] = delta + self.gamma * self.gae_lambda * advantages[max((t+1)%len(game['rewards']),t)]

        # adv and returns
        return advantages, advantages + torch.tensor(game['values'])
    
    def get_minibatch_generator(self, batch_size):

        # fold and stack observations
        actions = torch.cat([item for sublist in self.actions for item in sublist])
        logprobs = torch.cat([item for sublist in self.log_probs for item in sublist])
        returns = torch.cat(self.returns)
        values = torch.cat([item for sublist in self.values for item in sublist])
        advantages = torch.cat(self.advantages).float()

        # TODO : Gotta update these to work with a single set of hxs, rxs
        hxs, cxs = zip(*[(hxs, cxs) for hxs, cxs in [item for sublist in self.hcxs for item in sublist]])
        observations = torch.cat([item for sublist in self.observations for item in sublist])

        index = np.arange(len(observations))

        np.random.shuffle(index)

        # We do not handle remaining stuff here
        for start in range(0,len(observations), batch_size):
            end = start + batch_size
            batch_index = index[start:end].astype(int)

            yield {
                "actions": actions[batch_index],
                "logprobs": logprobs[batch_index],
                "returns": returns[batch_index],
                "values": values[batch_index],
                "advantages": advantages[batch_index],
                # we are using sequence lengths of 1, because everything should be encoded in 
                "hxs": torch.swapaxes(torch.cat(hxs)[batch_index],0,1),
                "cxs": torch.swapaxes(torch.cat(cxs)[batch_index],0,1),
                "observations": observations[batch_index]
            } 