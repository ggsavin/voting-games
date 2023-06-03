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
    "death": -3,
    "win": 25,
    "loss": -20,
    # "dead_wolf": 10, # Maybe not needed ?
    "self_vote": -3,
    "dead_vote": -5,
    "no_viable": -5,
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

    def __init__(self, num_agents=5, werewolves=1, num_accusations=1):
        super().__init__()

        assert werewolves < num_agents, f"The number of werewolves should be less than the number of players ({num_agents})"
        assert werewolves <= np.sqrt(num_agents), f"The number of werewolves should be less than the square root of agents ({num_agents})"       

        self.agents = [f"player_{i}" for i in range(num_agents)]
        self.possible_agents = self.agents[:]
        self.possible_roles = [Roles.WEREWOLF] * werewolves + [Roles.VILLAGER] * (num_agents - werewolves)
        self.agent_roles = { name : role for name, role in zip(self.agents, self.possible_roles)}
        self.num_accusation_steps = num_accusations

        self.world_state = {
            "day": 1,
            "phase": Phase.ACCUSATION,
            "step": 0,
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
        self.action_spaces = { name: Discrete(num_agents)  for name in self.agents }
        self.observation_spaces = {
            name: Dict(
                {
                    "observation": Dict({
                        "day": Discrete(int(num_agents/2), start=1),
                        # "time_step": Discrete(int(num_agents/2), start=1), TODO: Make this depend on a phase amount, so we can have
                        # multiple accusation phases if needed
                        "phase": Discrete(3),
                        "self_id": Discrete(num_agents), # TODO: FINISH THIS # now hot encode this
                        "player_status": Box(low=0, high=1, shape=(num_agents,), dtype=bool), # Is player alive or dead
                        "roles": Box(low=0, high=1, shape=(num_agents,), dtype=int), # 0 - villager, 1 - werewolf 
                        "votes" : Dict({
                            name: Box(low=0, high=num_agents, shape=(num_agents,)) for name in self.agents}),
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

    # 
    # TODO: return the target and voting information on each player
    def _get_player_to_be_killed(self, actions) -> tuple[int, object]:
        '''
        Plurality Voting
        '''
        # TODO : We want to know if agents voted for a dead player (that was not an already dead werewolf)
        #        or 

        # no_viable_vote : no viable living target (i.e just voted for a dead wolf if villager)
        # dead_vote : voted for a dead player, (does not count if the ggent keeps a negative opinion of a dead player) 
        #   TODO: maybe this should still give a penalty
        infos = {a: {"self_vote" : False, "dead_vote": False, "viable_vote": 0} for a in actions.keys()}

        votes = np.array([0] * len(self.possible_agents))

        for player, action in actions.items():
            pid = int(player.split("_")[-1])

            if action == pid:
                infos[player]["self_vote"] = True

            # dead_vote
            if f'player_{action}' in self.dead_agents:
                infos[player]["dead_vote"] = True
            
            # maybe we want to sum how many viable votes are done
            if f'player_{action}' in actions.keys():
                infos[player]["viable_vote"] = True

            votes[action] += 1

        # if we have a tie, keep the living players and randomly choose between them, report back a tie
        # TODO: Can we do this calculation in the loop above? Is it better to do it in the loop above?
        max_indicies = np.where(votes == max(votes))[0]
        living_selections = [player for player in max_indicies if f'player_{player}' not in self.dead_agents]

        # If we have any living selections, lets sampple one
        if len(living_selections) > 0:
            return random.choice(living_selections), infos

        # keep going down the chain if we have no living selections
        for next_best in np.argsort(votes)[::-1][len(max_indicies):]:
            if f'player_{next_best}' not in self.dead_agents:
                return next_best, infos
        
    def _check_for_winner(self):
        winners = None

        if not set(self.world_state["werewolves"]) & set(self.world_state['alive']):
            winners = Roles.VILLAGER

        # if there is an equal or more amount of werewolves than villagers, the wolves win
        elif len(set(self.world_state["werewolves"]) & set(self.world_state['alive'])) >= \
            len(set(self.world_state["villagers"]) & set(self.world_state['alive'])):
            # print("Werewolves WIN!!!!")
            winners = Roles.WEREWOLF

        return winners
    
    def _get_roles(self, agent):
        if self.agent_roles[agent] == Roles.VILLAGER:
            # If a werewolf is dead, then reveal their role
            roles = [Roles.VILLAGER] * len(self.possible_agents)
            for agent in self.possible_agents:
                if agent in self.dead_agents and agent in self.world_state["werewolves"]:
                    roles[int(agent.split("_")[-1])] = Roles.WEREWOLF
        else:
            # werewolves know the true roles of everyone
            roles = list(self.agent_roles.values())
        return roles
    
    # TODO: make these the plurality votes
    def _get_votes(self, agent):
        # important point : num_agents + 1 is the null vote
        if self.agent_roles[agent] == Roles.VILLAGER and self.history[-1]['phase'] == Phase.NIGHT:
            votes = {agent: len(self.possible_agents) for agent in self.possible_agents}
        else:
            votes = {agent: len(self.possible_agents) if agent not in self.history[-1]['votes'] else self.history[-1]['votes'][agent] for agent in self.possible_agents}
        
        return votes
    def step(self, actions):
        # get votes, kill a target or random person if no legitimate player was voted for.
        # TODO : Penalty for no legitimate target
        # TODO : Get stats on if we had to randomly kill a living player
        # TODO : A dead werewolf will still get -1. how do we ignore this

        # TODO : do we end up here if a werewolf dies?
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # set up some returns early 
        rewards = {a: 0 for a in self.agents}
        infos = {a: {} for a in self.agents}
        truncations = {a: False for a in self.agents}
        terminations = {a: False for a in self.agents}

        # if its nighttime, we will add 0 votes for all agents. All asleep
        # if its nighttime, villagers do not see votes
        self.world_state['votes'] = copy.deepcopy(actions)

        target, infos  = self._get_player_to_be_killed(actions)

        if self.world_state['phase'] != Phase.ACCUSATION:
            
            # add target to the dead agents
            self.dead_agents.append(f'player_{target}')
            # hand out dead reward to the agent this round
            rewards[f'player_{target}'] += REWARDS["death"]

            # updating these lists
            self.world_state['alive'].remove(f'player_{target}')
            
            if self.world_state['phase'] == Phase.NIGHT:
                self.world_state['killed'].append(f'player_{target}')
            elif self.world_state['phase'] == Phase.VOTING:
                self.world_state['executed'].append(f'player_{target}')

        # WIN CONDITIONS #
        terminations = {agent: agent == f'player_{target}' for agent in self.agents}

        # Do we have winners
        winners = self._check_for_winner()
        if winners != None:
            self.world_state['winners'] = winners
            terminations = {agent: True for agent in self.agents}

            # give out winning rewards to winners, and losing rewards to losers
            for agent in self.agents:
                rewards[agent] += REWARDS["win"] if self.agent_roles[agent] == winners else REWARDS["loss"]

        # votes are in, append snapshot of world state to history
        self.history.append(copy.deepcopy(self.world_state))

        # UPDATE TIME OF DAY AND PHASE # 
        if self.world_state['phase'] == Phase.NIGHT:
            self.world_state['day'] += 1
        self.world_state['phase'] =  Phase((self.world_state['phase'] + 1) % len(Phase))

        # FINISH Rewards
        # Reminder object of infos is : "self_vote" : False, "dead_vote": 0, "viable_vote": 0
        for agent, info in infos.items():

            if self.agent_roles[agent] == Roles.VILLAGER and self.history[-1]['phase'] == Phase.NIGHT:
                raise Exception("Villager should not have voted during the night")
            
            if self.history[-1]['phase'] != Phase.ACCUSATION:
                if info["self_vote"]:
                    rewards[agent] += REWARDS["self_vote"]
                
                if info["viable_vote"] == 0:
                    rewards[agent] += REWARDS["no_viable"]

                if info["dead_vote"] > 0:
                    # TODO: Is this too punishing?
                    rewards[agent] += info["dead_vote"]*REWARDS["dead_vote"]
                
                # if self.agent_roles[f'player_{target}'] == Roles.WEREWOLF and self.agent_roles[agent] == Roles.VILLAGER:
                #     rewards[agent] += REWARDS["dead_wolf"]

                 #  TODO: should we give this every step? or every day shift. and do we want to give i
                if not winners:
                    rewards[agent] += REWARDS["day"]

        # BUILD OUT OBSERVATIONS #
        action_mask = [agent not in self.dead_agents for agent in self.possible_agents]

        # villagers only see everyone voting 0 at night
        observations = {
            agent: {
                    "observation" : {
                    "day" : self.world_state["day"],
                    "phase": self.world_state["phase"],
                    "self_id": int(agent.split('_')[-1]),
                    "player_status": action_mask,
                    "roles": self._get_roles(agent),
                    "votes": self._get_votes(agent)
                },
                "action_mask": action_mask
            }
            for agent in self.agents
        }

        # take out the dead agent. Needs to be done for future loops
        if self.history[-1]['phase'] != Phase.ACCUSATION:
            self.agents.remove(f'player_{target}')

        if self.world_state['winners'] != None:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _game_phase_iterator(self):
        phase = Phase(0)
        while True:
            yield phase
            phase = Phase((phase + 1) % len(Phase))

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
            "votes": {agent: len(self.possible_agents) for agent in self.agents},
            "winners": None,
        }
        self.history = [copy.deepcopy(self.world_state)]
        self.votes = {agent: len(self.possible_agents)+1 for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # lets create observations here and return a whole step return
        action_mask = [agent not in self.dead_agents for agent in self.possible_agents]
        observations = {
            agent: {
                    "observation" : {
                    "day" : self.history[-1]["day"],
                    "phase": self.history[-1]["phase"],
                    "self_id": int(agent.split('_')[-1]),
                    "player_status": action_mask,
                    "roles": self._get_roles(agent),
                    "votes": self.history[-1]['votes']
                },
                "action_mask": action_mask
            }
            for agent in self.agents
        }

        return observations, self.rewards, self.terminations, self.truncations, self.infos


    def convert_obs(self, observation):
        
        if len(observation["votes"].keys()) != len(observation["player_status"]):
            raise Exception()
        
        return  np.asarray([observation['day']] + \
        [observation['phase']] + \
        [observation['self_id']] + \
        [int(status) for status in observation['player_status']] + \
        [role for role in observation['roles']] + \
        [i for sublist in observation["votes"].values() for i in sublist])
    
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

    observations, rewards, terminations, truncations, infos = env.reset()
    env.render()
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents if not (env.world_state["phase"] == Phase.NIGHT and env.agent_roles[agent] == Roles.VILLAGER)}  # this is where you would insert your policy
        observations, rewards, terminations, truncations, infos = env.step(actions)
        env.render()
    env.render() # post game render

print("Done")