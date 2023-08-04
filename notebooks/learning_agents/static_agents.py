
import torch
import random

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

def random_approval_villager(env, agent, action=None):
    if action != None:
        return action
    
    targets = set(env.world_state["alive"]) - set([agent])
    action = [0] * len(env.possible_agents)
    action[int(agent.split("_")[-1])] = 1
    action[int(random.choice(list(targets)).split("_")[-1])] = -1

    return action

def random_plurality_villager(env, agent, action=None):
    if action != None:
        return action

    targets = set(env.world_state["alive"]) - set([agent])
    return int(random.choice(list(targets)).split("_")[-1])


    
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