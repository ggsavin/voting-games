from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.test import api_test
import gym
import enum
import numpy as np

class Roles(enum.IntEnum):
    VILLAGER = 0
    WEREWOLF = 1

REWARDS = {
    "day": -1,
    "death": -5,
    "win": 25,
    "loss": -25,
    "vote_miss": -1
}

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)

class raw_env(AECEnv):

    metadata = {
        "render_modes" : ["human"],
        "name": "werewolf_v1"
    }

    def __init__(self, num_agents=5, werewolves=1):
        super().__init__()

        assert werewolves < num_agents, f"The number of werewolves should be less than the number of players ({num_agents})"
        assert werewolves <= np.sqrt(num_agents), f"The number of werewolves should be less than the square root of agents ({num_agents})"       

        # Action and Observation Spaces
        self.action_spaces = { name: gym.spaces.Discrete(num_agents) for name in self.agents }
        self.observation_spaces = {
            name: gym.spaces.Dict(
                {
                    "observation": gym.spaces.Dict({
                        "day": gym.spaces.Discrete(int(num_agents/2), start=1),
                        "time_of_day": gym.spaces.Discrete(2),
                        "player_status": gym.spaces.Box(low=0, high=1, shape=(num_agents,), dtype=bool), # Is player alive or dead
                        "roles": gym.spaces.Box(low=0, high=1, shape=(num_agents,), dtype=int), # 0 - villager, 1 - werewolf 
                        "votes": gym.spaces.Box(low=0, high=num_agents+1, shape=(num_agents,))
                }),
                    "action_mask": gym.spaces.Box(low=0, high=1, shape=(num_agents,), dtype=bool)
                }
            )
            for name in self.agents
        }

    def reset(self, seed=None, return_info=False, options=None):
        pass 

    def step(self, action):
        pass
    
    def observation_space(self, agent: str) -> gym.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        return self.action_spaces[agent]


if __name__ == "__main__":

    api_test(raw_env(), num_cycles=100, verbose_progress=True)