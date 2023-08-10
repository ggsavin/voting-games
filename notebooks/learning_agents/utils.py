import torch
import numpy as np
import copy
import enum

class Roles(enum.IntEnum):
    VILLAGER = 0
    WEREWOLF = 1

class Phase(enum.IntEnum):
    ACCUSATION = 0
    VOTING = 1
    NIGHT = 2

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

@torch.no_grad()
def play_recurrent_game(env, wolf_policy, villager_agent, num_times=10, hidden_state_size=None, voting_type=None):

    wins = 0
    game_replays = []

    for _ in range(num_times):
        next_observations, _, _, _, _ = env.reset()
        # init recurrent stuff for actor and critic to 0 as well
        magent_obs = {agent: {'obs': [], 
                              # obs size, and 1,1,64 as we pass batch first
                              'hcxs': [(torch.zeros((1,1,hidden_state_size), dtype=torch.float32), 
                                        torch.zeros((1,1,hidden_state_size), dtype=torch.float32))],
                    } for agent in env.agents if not env.agent_roles[agent]}

        wolf_action = None

        while env.agents:
            observations = copy.deepcopy(next_observations)
            actions = {}

            villagers = set(env.agents) & set(env.world_state["villagers"])
            wolves = set(env.agents) & set(env.world_state["werewolves"])

            ## VILLAGER LOGIC ##
            v_obs = torch.cat([torch.unsqueeze(torch.tensor(env.convert_obs(observations[villager]['observation']), dtype=torch.float), 0) for villager in villagers])

            # TODO: maybe this can be sped up? 
            hxs, cxs = zip(*[(hxs, cxs) for hxs, cxs in [magent_obs[villager]["hcxs"][-1] for villager in villagers]])
            hxs = torch.swapaxes(torch.cat(hxs),0,1)
            cxs = torch.swapaxes(torch.cat(cxs),0,1)

            policies, _ , cells = villager_agent(v_obs, (hxs, cxs))
            v_actions = torch.stack([p.sample() for p in policies], dim=1)

            hxs_new, cxs_new = cells
            hxs_new = torch.swapaxes(hxs_new,1,0)
            cxs_new = torch.swapaxes(cxs_new,1,0)

            for i, villager in enumerate(villagers):
                if voting_type == "plurality":
                    actions[villager] = v_actions[i].item()
                elif voting_type == "approval":
                    actions[villager] = (v_actions[i] - 1).tolist()
                magent_obs[villager]['hcxs'].append((torch.unsqueeze(hxs_new[i], 0), torch.unsqueeze(cxs_new[i], 0)))

            ## WOLF LOGIC ## 
            phase = env.world_state['phase']
            for wolf in wolves:
                wolf_action = wolf_policy(env, wolf, action=wolf_action)
                actions[wolf] = wolf_action

            next_observations, _, _, _, _ = env.step(actions)

            ## UPDATED WOLF VARIABLE FOR WOLVES THAT COORDINATE ## 
            if env.world_state['phase'] == Phase.NIGHT:
                wolf_action = None
            
            if env.world_state['phase'] == Phase.ACCUSATION and phase == Phase.NIGHT:
                wolf_action = None
            
        ## Fill bigger buffer, keeping in mind sequence
        winner = env.world_state['winners']
        if winner == Roles.VILLAGER:
            wins += 1

        game_replays.append(copy.deepcopy(env.history))
    
    return wins, game_replays

def play_static_game(env, wolf_policy, villager_policy, num_times=100):

    villager_wins = 0
    game_replays = []
    for _ in range(num_times):
        observations, rewards, terminations, truncations, infos = env.reset()
        
        wolf_action = None
        while env.agents:
            actions = {}

            villagers = set(env.agents) & set(env.world_state["villagers"])
            wolves = set(env.agents) & set(env.world_state["werewolves"])

            # villager steps
            for villager in villagers:
                actions[villager] = villager_policy(env, villager)


            # wolf steps
            phase = env.world_state['phase']
            for wolf in wolves:
                wolf_action = wolf_policy(env, wolf, action=wolf_action)
                actions[wolf] = wolf_action
        
            observations, rewards, terminations, truncations, infos = env.step(actions)


            if env.world_state['phase'] == Phase.NIGHT:
                wolf_action = None
            
            if env.world_state['phase'] == Phase.ACCUSATION and phase == Phase.NIGHT:
                wolf_action = None

        winner = env.world_state['winners']
        if winner == Roles.VILLAGER:
            villager_wins += 1

        game_replays.append(copy.deepcopy(env.history))

    return villager_wins, game_replays