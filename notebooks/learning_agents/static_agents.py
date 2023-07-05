
import torch

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