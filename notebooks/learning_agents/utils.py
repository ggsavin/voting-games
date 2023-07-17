import torch
import numpy as np

# def convert_obs(self, observation):
        
#         if len(observation["votes"].keys()) != len(observation["player_status"]):
#             raise Exception()
        
#         # plurality
#         return  np.asarray([observation['day']] + \
#         [observation['phase']] + \
#         [observation['self_id']] + \
#         [int(status) for status in observation['player_status']] + \
#         [role for role in observation['roles']] + \
#         list(observation["votes"].values()))

#         # approval 
#         return  np.asarray([observation['day']] + \
#         [observation['phase']] + \
#         [observation['self_id']] + \
#         [int(status) for status in observation['player_status']] + \
#         [role for role in observation['roles']] + \
#         [i for sublist in observation["votes"].values() for i in sublist])


def convert_obs(observation, voting_type=None, one_hot=False):
    '''
    observation will have the following information
        day (int)
        phase (int) 
        self_id (int)
        player_status (list) - 0/1 for alive or dead
        roles (list) - 0/1 for villager or werewolf
        votes (dict) - dict with player and associated vote
    '''

    if len(observation["votes"].keys()) != len(observation["player_status"]):
        raise Exception()
    
    # phase length
    day = torch.tensor([observation['day']])

    # we can make the phase a one hot, hardcoded 3 phases
    if one_hot:
        phase = torch.nn.functional.one_hot(torch.tensor(observation['phase']), num_classes=3)
        self_id = torch.nn.functional.one_hot(torch.tensor(observation['self_id']), num_classes=len(observation['roles']))

        if voting_type == "approval":
            votes = torch.nn.functional.one_hot(torch.tensor(list(observation['votes'].values())) + 1, num_classes=3).reshape(-1)
        elif voting_type == "plurality":
            votes = torch.nn.functional.one_hot(torch.tensor(list(observation['votes'].values())), num_classes=len(observation['roles'])+ 1).reshape(-1)

    else:

        if voting_type == "approval":
            votes = torch.tensor(list(observation['votes'].values())).reshape(-1)
        elif voting_type == "plurality":
            votes = torch.tensor(list(observation['votes'].values()))

        phase = torch.tensor([observation['phase']])
        self_id = torch.tensor([observation['self_id']])


    # PLAYER STATUS (ALIVE OR DEAD)
    player_status = torch.tensor(observation['player_status'], dtype=torch.int)
    player_roles = torch.tensor(observation['roles'], dtype=torch.int)

    return torch.cat((day, phase, self_id, player_status, player_roles, votes)).float()
