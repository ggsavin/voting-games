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
        self.action_spaces = { name: Discrete(num_agents) for name in self.agents }
        self.observation_spaces = {
            name: Dict(
                {
                    "observation": Dict({
                        "day": Discrete(int(num_agents/2), start=1),
                        "phase": Discrete(3),
                        "self_id": Discrete(num_agents), # TODO: FINISH THIS
                        "player_status": Box(low=0, high=1, shape=(num_agents,), dtype=bool), # Is player alive or dead
                        "roles": Box(low=0, high=1, shape=(num_agents,), dtype=int), # 0 - villager, 1 - werewolf 
                        "votes": Box(low=0, high=num_agents+1, shape=(num_agents,)) # num_agents + 1 means a no-vote because most likely dead
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
    def _get_player_to_be_killed(self, votes) -> tuple[int, bool]:
        vote_counts = collections.Counter(votes)
        flag = 0
        for (player_id,_) in vote_counts.most_common():
            if f'player_{player_id}' not in self.dead_agents:
                return player_id, bool(flag)
            flag += 1
        # no legitimate player was voted for, kill a random living player
        # TODO : Consider another a seperate flag if we ever end up here?
        player = random.choice(self.world_state['alive'])
        return int(player.split('_')[-1]), True
    
    
    def step(self, actions):
        # get votes, kill a target or random person if no legitimate player was voted for.
        # TODO : Penalty for no legitimate target
        # TODO : Get stats on if we had to randomly kill a living player

        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        target, bad_vote = self._get_player_to_be_killed(actions.values())

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
                    "votes": [len(self.possible_agents)] * len(self.possible_agents)
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
            "votes": {},
            "winners": None,
        }
        self.history = [copy.deepcopy(self.world_state)]

        self.votes = {agent: 0 for agent in self.agents}

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

    legal_actions = [action for action,is_alive,is_wolf in zip(available_actions, action_mask, observation['observation']['roles']) if is_alive and not is_wolf]
    # wolves don't vote for other wolves. will select another villager at random
    action = random.choice(legal_actions)
    return action

if __name__ == "__main__":

    # api_test(raw_env(), num_cycles=100, verbose_progress=True)

    env = raw_env()
    env.reset()

    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
        env.render()
        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.render() # post game render
    print("hello")

print("Done")