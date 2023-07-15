import torch

def convert_as_much_one_hot(observation, voting_type=None):

    if len(observation["votes"].keys()) != len(observation["player_status"]):
        raise Exception()
    
    # phase length

    # how many agents are there?
    num_agents = len(observation['roles'])
    # 

    # we can make the phase a one hot, hardcoded 3 phases
    torch.nn.functional.one_hot(torch.tensor(observation['phase']), num_classes=3)

    # we can make the self_id a one hot
    torch.nn.functional.one_hot(torch.tensor(observation['self_id']), num_classes=len(num_agents))

    # votes can* be a one hot, but 
    if voting_type == "approval":
        torch.nn.functional.one_hot(torch.tensor(list(observation['votes'].values())) + 1, num_classes=3).reshape(-1)
    elif voting_type == "plurality":
        torch.nn.functional.one_hot(torch.tensor(list(observation['votes'].values())), num_classes=10).reshape(-1)
