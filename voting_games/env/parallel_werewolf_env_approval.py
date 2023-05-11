from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.test import api_test
from gymnasium.spaces import Discrete, MultiDiscrete, Dict, Box, Space
import enum
import random
import numpy as np
import collections
import json
import os
import copy 

class Roles(enum.IntEnum):
    VILLAGER = 0
    WEREWOLF = 1

class Phase(enum.IntEnum):
    ACCUSATION = 0
    VOTING = 1
    NIGHT = 2

REWARDS = {
    "day": -1,
    "death": -5,
    "win": 50,
    "loss": -30,
    "vote_miss": -1,
    "self_vote": -3,
    "dead_vote": -5,
}

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)

class raw_env(ParallelEnv):

    metadata = {
        "render_modes" : ["human"],
        "name": "werewolf_v1"
    }

    def __init__(self, num_agents=5, werewolves=1):
        super().__init__()

        assert werewolves < num_agents, f"The number of werewolves should be less than the number of players ({num_agents})"
        assert werewolves <= np.sqrt(num_agents), f"The number of werewolves should be less than the square root of agents ({num_agents})"       

        self.agents = [f"player_{i}" for i in range(num_agents)]
        self.possible_agents = self.agents[:]
        self.possible_roles = [Roles.WEREWOLF] * werewolves + [Roles.VILLAGER] * (num_agents - werewolves)
        self.agent_roles = { name : role for name, role in zip(self.agents, self.possible_roles)}

        self.world_state = {
            "day": 1,
            "phase": Phase.ACCUSATION,
            "alive": self.agents.copy(),
            "killed": [],
            "executed": [],
            "werewolves_remaining": [],
            "villagers_remaining": [],
            "votes": {},
            "winners": None,
        }
        self.history = [copy.deepcopy(self.world_state)]

        # Action and Observation Spaces
        self.action_spaces = { name: Box(low=-1, high=1, shape=(num_agents,), dtype=int) for name in self.agents }
        self.observation_spaces = {
            name: Dict(
                {
                    "observation": Dict({
                        "day": Discrete(int(num_agents/2), start=1),
                        "phase": Discrete(3),
                        "self_id": Discrete(num_agents), # TODO: FINISH THIS
                        "player_status": Box(low=0, high=1, shape=(num_agents,), dtype=bool), # Is player alive or dead
                        "roles": Box(low=0, high=1, shape=(num_agents,), dtype=int), # 0 - villager, 1 - werewolf 
                        "votes" : Dict({
                            name: Box(low=-1, high=1, shape=(num_agents,), dtype=int) for name in self.agents}),
                    }),
                    "action_mask": Box(low=0, high=1, shape=(num_agents,), dtype=bool)
                }
            )
            for name in self.agents
        }

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def observation_space(self, agent: str) -> Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> Space:
        return self.action_spaces[agent]
    
    def reset():
        pass

    # we are keeping track of a random vote needing to be triggered
    def _get_player_to_be_killed(self, actions) -> tuple[int, bool, bool]:
        # we want to vote out the player with the lowest score.
        # an approval should not count against a low score
        votes = [[0 if i == 1 else i for i in p_actions] for p_actions in actions]
        votes = np.sum(votes, axis=1)

        max_indices = np.where(votes == min(votes))[0]
        dead_vote_flag = False
        tie_vote_flag = False

        # if we have a tie, keep the living players and randomly choose between them, report back a tie
        living_selections = [player for player in max_indices if player not in self.dead_agents]

        if len(max_indices) > 1:
            tie_vote_flag = True


        if len(living_selections) < len(max_indices):
            dead_vote_flag = True

        # If we have any living selections, lets sampple one
        if len(living_selections) > 0:
            return random.choice(living_selections), dead_vote_flag, tie_vote_flag

        # keep going down the chain
        for next_best in np.argsort(votes)[len(max_indices):]:
            if next_best not in self.dead_agents:
                return next_best, dead_vote_flag, tie_vote_flag
        
    
    
    def step(self, actions):
        # get votes, kill a target or random person if no legitimate player was voted for.
        # TODO : Penalty for no legitimate target
        # TODO : Get stats on if we had to randomly kill a living player
        # TODO : A dead werewolf will still get -1. how do we ignore this

        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        self.world_state['votes'] = copy.deepcopy(actions)
        target, dead_vote, tie_vote  = self._get_player_to_be_killed(list(actions.values()))

        if self.world_state['phase'] != Phase.ACCUSATION:
            
            print(f'About to kill {target}')
            self.dead_agents.append(f'player_{target}')

            # updating these lists
            self.world_state['alive'].remove(f'player_{target}')
            
            if self.world_state['phase'] == Phase.NIGHT:
                self.world_state['killed'].append(f'player_{target}')
            elif self.world_state['phase'] == Phase.VOTING:
                self.world_state['executed'].append(f'player_{target}')

        # WIN CONDITIONS #
        terminations = {agent: agent == f'player_{target}' for agent in actions.keys()}

        if not set(self.world_state["werewolves"]) & set(self.world_state['alive']):
            # print("Villagers WIN!!!!!")
            self.world_state['winners'] = Roles.VILLAGER
            terminations = {agent: True for agent in actions.keys()}

        elif len(set(self.world_state["werewolves"]) & set(self.world_state['alive'])) >= \
            len(set(self.world_state["villagers"]) & set(self.world_state['alive'])):
            # print("Werewolves WIN!!!!")
            self.world_state['winners'] = Roles.WEREWOLF
            terminations = {agent: True for agent in actions.keys()}

        # votes are in, append snapshot of world state to history
        self.history.append(copy.deepcopy(self.world_state))

        # UPDATE TIME OF DAY AND PHASE # 
        if self.world_state['phase'] == Phase.NIGHT:
            self.world_state['day'] += 1
        self.world_state['phase'] =  (self.world_state['phase'] + 1) % 3

        # hand out rewards conditions
        rewards = {a: 0 for a in self.agents}

        # Override with truncations if needed
        truncations = {a: False for a in self.agents}

        # if self.agent_roles[agent] == Roles.VILLAGER and self.history[-1]['phase'] == Phase.NIGHT:
        #     votes = [len(self.possible_agents)] * len(self.possible_agents)
        # elif len(self.history) == 1:
        #     votes = [0] * len(self.possible_agents)
        # else: 
        #     votes = [0 if agent not in prev_state['votes'] else prev_state['votes'][agent] for agent in self.possible_agents]

        # BUILD OUT OBSERVATIONS #
        action_mask = [agent not in self.dead_agents for agent in self.possible_agents]
        observations = {
            agent: {
                    "observation" : {
                    "day" : self.history[-1]["day"],
                    "phase": self.history[-1]["phase"],
                    "self_id": int(agent.split('_')[-1]),
                    "player_status": action_mask,
                    "roles": [Roles.VILLAGER] * len(self.possible_agents) if self.agent_roles[agent] == Roles.VILLAGER else list(self.agent_roles.values()),
                    "votes": self.history[-1]['votes']
                },
                "action_mask": action_mask
            }
            for agent in self.agents
        }

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        # take out the dead agent
        if self.history[-1]['phase'] != Phase.ACCUSATION:
            self.agents.remove(f'player_{target}')

        if self.world_state['winners'] != None:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self, mode: str = "human"):
        """
        Renders the environment. In human mode, it prints to the terminal
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print(json.dumps(self.world_state, indent=4))
    
    def close():
        """
        Needed for games with a render function
        """
        pass

    def reset(self, seed=None, return_info=False, options=None):
        self.agents = self.possible_agents[:]
        self.dead_agents = []
        random.shuffle(self.possible_roles)
        self.agent_roles = { name : role for name, role in zip(self.agents, self.possible_roles)}

        self.world_state = {
            "day": 1,
            "phase": Phase.ACCUSATION,
            "alive": self.agents.copy(),
            "killed": [],
            "executed": [],
            "werewolves": [agent for agent in self.agents if self.agent_roles[agent] == Roles.WEREWOLF],
            "villagers": [agent for agent in self.agents if self.agent_roles[agent] == Roles.VILLAGER],
            "votes": {agent: [] for agent in self.agents},
            "winners": None,
        }
        self.history = [copy.deepcopy(self.world_state)]

        self.votes = {agent: [] for agent in self.agents}

        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}


def random_policy(observation, agent):
    # these are the other wolves. we cannot vote for them either
    available_actions = list(range(len(observation['observation']['player_status'])))
    # dead players
    action_mask = observation['action_mask']

    #TODO: random.choice a full array at once
    action = [random.choice([0,1,-1]) for _ in action_mask]
    return action

if __name__ == "__main__":

    # api_test(raw_env(), num_cycles=100, verbose_progress=True)

    env = raw_env()
    env.reset()

    while env.agents:
        actions = {agent: env.action_space(agent).sample().tolist() for agent in env.agents}  # this is where you would insert your policy
        env.render()
        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.render() # post game render
    print("hello")

print("Done")