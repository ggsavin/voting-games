import torch
import numpy as np

def convert_obs(self, observation):
        
        if len(observation["votes"].keys()) != len(observation["player_status"]):
            raise Exception()
        
        # plurality
        return  np.asarray([observation['day']] + \
        [observation['phase']] + \
        [observation['self_id']] + \
        [int(status) for status in observation['player_status']] + \
        [role for role in observation['roles']] + \
        list(observation["votes"].values()))

        # approval 
        return  np.asarray([observation['day']] + \
        [observation['phase']] + \
        [observation['self_id']] + \
        [int(status) for status in observation['player_status']] + \
        [role for role in observation['roles']] + \
        [i for sublist in observation["votes"].values() for i in sublist])


def convert_obs_to_one_hot(observation, voting_type=None):
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
    day = torch.tensor(observation['day'])

    # we can make the phase a one hot, hardcoded 3 phases
    # phase = torch.tensor(observation['phase'])
    phase = torch.nn.functional.one_hot(torch.tensor(observation['phase']), num_classes=3)

    # we can make the self_id a one hot
    # self_id = torch.tensor(observation['self_id'])
    self_id = torch.nn.functional.one_hot(torch.tensor(observation['self_id']), num_classes=len(observation['roles']))

    # player status is bools
    player_status = torch.tensor(observation['player_status'], dtype=torch.int)
    
    # votes can* be a one hot, but 
    if voting_type == "approval":
        votes = torch.nn.functional.one_hot(torch.tensor(list(observation['votes'].values())) + 1, num_classes=3).reshape(-1)
    elif voting_type == "plurality":
        votes = torch.nn.functional.one_hot(torch.tensor(list(observation['votes'].values())), num_classes=len(observation['roles'])).reshape(-1)

    return torch.cat(day, phase, self_id, player_status, votes)
