
import torch
import random

### WEREWOLVES ###

def random_plurality_wolf(env, agent, action=None):
    villagers_remaining = set(env.world_state["villagers"]) & set(env.world_state['alive'])
    return action if action != None else int(random.choice(list(villagers_remaining)).split("_")[-1])

def random_approval_wolf(env, agent, action=None):
    if action != None:
        return action

    villagers_remaining = set(env.world_state["villagers"]) & set(env.world_state['alive'])
    wolves_remaining = set(env.world_state["werewolves"]) & set(env.world_state['alive'])

    # pick a living target
    target = random.choice(list(villagers_remaining))

    action = [0] * len(env.possible_agents)
    action[int(target.split("_")[-1])] = -1
    for curr_wolf in wolves_remaining:
        action[int(curr_wolf.split("_")[-1])] = 1

    return action


def revenge_plurality_wolf(env, agent, action=None):
    villagers_remaining = set(env.world_state["villagers"]) & set(env.world_state['alive'])

    prev_votes = env.history[-1]['votes']
    villagers_targetting_you = [player for player in prev_votes if f'player_{prev_votes[player]}' == agent]
    if len(villagers_targetting_you) > 0:
        return int(random.choice(list(villagers_targetting_you)).split("_")[-1])
    
    return int(random.choice(list(villagers_remaining)).split("_")[-1])

def coordinated_revenge_plurality_wolf(env, agent, action=None):
    villagers_remaining = set(env.world_state["villagers"]) & set(env.world_state['alive'])
    wolves_remaining = set(env.world_state["werewolves"]) & set(env.world_state['alive'])

    # who tried to vote out a wolf last time?
    # TODO : just get this from the observations
    prev_votes = env.history[-1]['votes']
    villagers_targetting_wolves = [player for player in prev_votes if f'player_{prev_votes[player]}' in wolves_remaining]
    # target = random.choice(list(villagers_remaining))
    # pick 
    if action != None:
        return action
    
    if len(villagers_targetting_wolves) == 0:
        return int(random.choice(list(villagers_remaining)).split("_")[-1])
    
    return int(random.choice(list(villagers_targetting_wolves)).split("_")[-1])


### VILLAGERS ###

def random_approval_villager(env, agent, action=None):
    targets = set(env.world_state["alive"]) - set([agent])
    action = [0] * len(env.possible_agents)
    action[int(agent.split("_")[-1])] = 1
    action[int(random.choice(list(targets)).split("_")[-1])] = -1
    return action

def random_coordinated_approval_villager(env, agent, action=None):
    return action if action != None else \
        random_approval_villager(env, agent)

def random_plurality_villager(env, agent, action=None):
    targets = set(env.world_state["alive"]) - set([agent])
    return int(random.choice(list(targets)).split("_")[-1])

def random_coordinated_plurality_villager(env, agent, action=None):
    return action if action != None else \
        random_plurality_villager(env, agent)

def random_agent(env, agent, action=None):
   return env.action_space(agent).sample()

    
class RandomRecurrentPluralityAgent(torch.nn.Module):
    def __init__(self, voting_type=None):
        super().__init__()

        assert voting_type != None, "Static agent needs a voting type"
        self.voting_type = voting_type
    
    def forward(self, x, recurrent_cell: torch.tensor):
        #TODO: do something about observation
        return x, recurrent_cell

    def get_action_from_policies(self, policies, voting_type=None):
        policy_action = [policy.sample() for policy in policies]
        if voting_type == "approval":
            game_action = self._convert_policy_action_to_game_action(policy_action)
        elif voting_type == "plurality":
            game_action = policy_action[0]
        else:
            raise ValueError("voting type not implemented!")

        return policy_action, game_action