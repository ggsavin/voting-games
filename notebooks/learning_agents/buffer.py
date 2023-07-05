import torch
import numpy as np
import copy

class ReplayBuffer():
    
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
        actions = torch.stack([torch.tensor(item) for sublist in self.actions for item in sublist])
        logprobs = torch.stack([item.reshape(-1) for sublist in self.log_probs for item in sublist])
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


@torch.no_grad()
def fill_recurrent_buffer(buffer, env, config:dict, wolf_policy, villager_agent, voting_type=None):

    buffer.reset(gamma=config["training"]["gamma"], gae_lambda=config["training"]["gae_lambda"])
    
    for _ in range(config["training"]["buffer_games_per_update"]):
        ## Play the game 
        next_observations, rewards, terminations, truncations, infos = env.reset()
        # init recurrent stuff for actor and critic to 0 as well
        magent_obs = {agent: {'obs': [], 
                              'rewards': [], 
                              'actions': [], 
                              'logprobs': [], 
                              'values': [], 
                              'terms': [],

                              # obs size, and 1,1,64 as we pass batch first
                              'hcxs': [(torch.zeros((1,1,config["model"]["recurrent_hidden_size"]), dtype=torch.float32), 
                                        torch.zeros((1,1,config["model"]["recurrent_hidden_size"]), dtype=torch.float32))]
                    } for agent in env.agents if not env.agent_roles[agent]}
        
        wolf_brain = {'day': 1, 'phase': 0, 'action': None}
        while env.agents:
            observations = copy.deepcopy(next_observations)
            actions = {}

            villagers = set(env.agents) & set(env.world_state["villagers"])
            wolves = set(env.agents) & set(env.world_state["werewolves"])

            # villager steps
                # villagers actions
            for villager in villagers:
                #torch.tensor(env.convert_obs(observations['player_0']['observation']), dtype=torch.float)
                torch_obs = torch.tensor(env.convert_obs(observations[villager]['observation']), dtype=torch.float)
                obs = torch.unsqueeze(torch_obs, 0)

                # TODO: Testing this, we may need a better way to pass in villagers
                recurrent_cell = magent_obs[villager]["hcxs"][-1]
                
                # ensure that the obs is of size (batch,seq,inputs)
                # we only have one policy, but to keep the same agent structure, we have "policies"
                policies, value, recurrent_cell = villager_agent(obs, recurrent_cell)
                policy_action, game_action = villager_agent.get_action_from_policies(policies, voting_type=voting_type)
                
                # only difference is the game_action here between this and approval
                if voting_type == "plurality":
                    actions[villager] = game_action.item()
                elif voting_type == "approval":
                    actions[villager] = game_action.tolist()

                # can store some stuff 
                magent_obs[villager]["obs"].append(obs)
                magent_obs[villager]["actions"].append(policy_action)

                # how do we get these
                magent_obs[villager]["logprobs"].append(torch.stack([policy.log_prob(action) for policy, action in zip(policies, policy_action)], dim=1).reshape(-1))
                magent_obs[villager]["values"].append(value)

                #store the next recurrent cells
                magent_obs[villager]["hcxs"].append(recurrent_cell)


            # wolf steps
            day = observations[list(observations)[0]]['observation']['day']
            phase = observations[list(observations)[0]]['observation']['phase']

            if wolf_brain['day'] != day or wolf_brain['phase'] == plurality_Phase.NIGHT:
                wolf_brain = {'day': day, 'phase': phase, 'action': None}

            for wolf in wolves:
                action = wolf_policy(env, wolf, action=wolf_brain['action'])
                wolf_brain['action'] = action
                actions[wolf] = action

            # actions = actions | wolf_policy(env)
        
            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            for villager in villagers:
                magent_obs[villager]["rewards"].append(rewards[villager])
                magent_obs[villager]["terms"].append(terminations[villager])

        ## Fill bigger buffer, keeping in mind sequence
        for agent in magent_obs:
            buffer.add_replay(magent_obs[agent])
    
    return buffer