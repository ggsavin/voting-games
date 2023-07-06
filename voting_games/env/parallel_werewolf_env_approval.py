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
    "player_death": -1,
    "player_win": 10,
    "player_loss": -5,
    "dead_wolf": 5, # Maybe not needed ?
    "dead_villager": -1,
    "self_vote": -1,
    "dead_vote": -1,
    "no_viable_vote": -1,
}


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)

class raw_env(ParallelEnv):

    metadata = {
        "render_modes" : ["human"],
        "name": "werewolf_approval_v1"
    }

    def __init__(self, num_agents=5, werewolves=1, num_accusations=1, rewards=REWARDS):
        super().__init__()

        assert werewolves < num_agents, f"The number of werewolves should be less than the number of players ({num_agents})"
        assert werewolves <= np.sqrt(num_agents), f"The number of werewolves should be less than the square root of agents ({num_agents})"       

        self.agents = [f"player_{i}" for i in range(num_agents)]
        self.possible_agents = self.agents[:]
        self.possible_roles = [Roles.WEREWOLF] * werewolves + [Roles.VILLAGER] * (num_agents - werewolves)
        self.agent_roles = { name : role for name, role in zip(self.agents, self.possible_roles)}
        self.num_accusation_steps = num_accusations

        self.game_phase_tracker = self._game_phase_iterator()
        day, phase, accusation_round = next(self.game_phase_tracker)

        self.world_state = {
            "day": day,
            "phase": phase,
            "round": accusation_round,
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

        self.rewards = rewards
        assert all(k in rewards for k in ("day","player_death", "player_win", "player_loss", "dead_wolf", "dead_villager", "self_vote", "dead_vote", "no_viable_vote"))

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def observation_space(self, agent: str) -> Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> Space:
        return self.action_spaces[agent]

    # we are keeping track of a random vote needing to be triggered
    # TODO: return the target and voting information on each player
    def _get_player_to_be_killed(self, actions) -> tuple[int, object]:
        # TODO : We want to know if agents voted for a dead player (that was not an already dead werewolf)
        #        or 

        # no_viable_vote : no viable living target (i.e just voted for a dead wolf if villager)
        # dead_vote : voted for a dead player, (does not count if the ggent keeps a negative opinion of a dead player) 
        #   TODO: maybe this should still give a penalty
        infos = {a: {"self_vote" : False, "dead_vote": 0, "viable_vote": 0, "did_not_sleep": False} for a in actions.keys()}

        votes = [0] * len(self.possible_agents)

        for player, action in actions.items():
            pid = int(player.split("_")[-1])

            for i, opinion in enumerate(action):
                
                # self_vote
                if i == pid and opinion == -1:
                    infos[player]["self_vote"] = True

                # dead_vote
                if f'player_{i}' in self.dead_agents and opinion == -1:
                    # maybe we want to possibly have a penalty here, or maybe we dont care
                    if self.agent_roles[f'player_{i}'] != Roles.WEREWOLF:
                        infos[player]["dead_vote"] += 1
                
                # maybe we want to sum how many viable votes are done
                if f'player_{i}' in actions.keys() and opinion == -1:
                    infos[player]["viable_vote"] += 1

                # only vote out 
                if opinion == -1 and not (self.agent_roles[player] == Roles.VILLAGER and self.world_state['phase'] == Phase.NIGHT):
                    votes[i] += opinion

        # if we have a tie, keep the living players and randomly choose between them, report back a tie
        # TODO: Can we do this calculation in the loop above? Is it better to do it in the loop above?
        min_indices = np.where(votes == min(votes))[0]
        living_selections = [player for player in min_indices if f'player_{player}' not in self.dead_agents]

        # If we have any living selections, lets sampple one
        if len(living_selections) > 0:
            return random.choice(living_selections), infos

        # keep going down the chain
        for next_best in np.argsort(votes)[len(min_indices):]:
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
    
    def _get_votes(self, agent):
        if self.agent_roles[agent] == Roles.VILLAGER and self.history[-1]['phase'] == Phase.NIGHT:
            votes = {agent: [0 for _ in range(0,len(self.possible_agents))] for agent in self.possible_agents}
        else: 
            votes = {agent: [0 for _ in range(0, len(self.possible_agents))] if agent not in self.history[-1]['votes'] else self.history[-1]['votes'][agent] for agent in self.possible_agents}
        
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

        target, infos = self._get_player_to_be_killed(actions)

        if self.world_state['phase'] != Phase.ACCUSATION:
            
            # add target to the dead agents
            self.dead_agents.append(f'player_{target}')
            # hand out dead reward to the agent this round
            rewards[f'player_{target}'] += self.rewards["player_death"]

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
                rewards[agent] += self.rewards["player_win"] if self.agent_roles[agent] == winners else self.rewards["player_loss"]

        # votes are in, append snapshot of world state to history
        self.history.append(copy.deepcopy(self.world_state))

        # UPDATE TIME OF DAY AND PHASE # 
        # TODO : Maybe just update these in the game phase tracker directly>
        day, phase, accusation_round = next(self.game_phase_tracker)
        self.world_state['phase'] = phase
        self.world_state['day'] = day
        self.world_state['round'] = accusation_round

        # FINISH Rewards
        # Reminder object of infos is : "self_vote" : False, "dead_vote": 0, "viable_vote": 0
        for agent, info in infos.items():

            # if self.agent_roles[agent] == Roles.VILLAGER and self.history[-1]['phase'] == Phase.NIGHT:
            #     raise Exception("Villager should not have voted during the night")
            
            if self.history[-1]['phase'] != Phase.ACCUSATION:

                if not (self.agent_roles[agent] == Roles.VILLAGER and self.history[-1]['phase'] == Phase.NIGHT):
                    if info["self_vote"]:
                        rewards[agent] += self.rewards["self_vote"]
                    
                    if info["viable_vote"] == 0:
                        rewards[agent] += self.rewards["no_viable_vote"]

                    if info["dead_vote"] > 0:
                        # TODO: Is this too punishing?
                        rewards[agent] += info["dead_vote"]*self.rewards["dead_vote"]
                    
                    if self.agent_roles[f'player_{target}'] == Roles.WEREWOLF and self.agent_roles[agent] == Roles.VILLAGER:
                        rewards[agent] += self.rewards["dead_wolf"]

                    # maybe too punishing?
                    if self.agent_roles[f'player_{target}'] == Roles.VILLAGER and self.agent_roles[agent] == Roles.VILLAGER:
                        rewards += self.rewards["dead_villager"]
                        
                #  TODO: Do this every day, not every phase
                if not winners and (self.world_state['day'] != self.history[-1]['day']):
                    rewards[agent] += self.rewards["day"]

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
        day = 1
        phase = Phase(0)
        accusation_round = 0
        while True:
            yield day, phase, accusation_round

            # do not roll over Phase until accusation roles over in accusation
            if phase == Phase.ACCUSATION:
                accusation_round = (accusation_round + 1) % self.num_accusation_steps

            # we looped out of accusation rounds, should move the phase out of accusation
            if accusation_round == 0:
                # if the next phase out of here is 
                if phase + 1 == len(Phase):
                    day += 1
            
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

        self.game_phase_tracker = self._game_phase_iterator()
        day, phase, accusation_round = next(self.game_phase_tracker)
        
        self.world_state = {
            "day": day,
            "phase": phase,
            "round": accusation_round,
            "alive": self.agents.copy(),
            "killed": [],
            "executed": [],
            "werewolves": [agent for agent in self.agents if self.agent_roles[agent] == Roles.WEREWOLF],
            "villagers": [agent for agent in self.agents if self.agent_roles[agent] == Roles.VILLAGER],
            "votes": {agent: [0]*len(self.possible_agents) for agent in self.agents},
            "winners": None,
        }
        self.history = [copy.deepcopy(self.world_state)]

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

        self.game_phase_tracker = self._game_phase_iterator()

        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos


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

    #TODO: random.choice a full array at once
    action = [random.choice([0,1,-1]) for _ in action_mask]
    return action

if __name__ == "__main__":

    # api_test(raw_env(), num_cycles=100, verbose_progress=True)

    env = raw_env(num_accusations=5)
    env.reset()

    while env.agents:
        actions = {agent: env.action_space(agent).sample().tolist() for agent in env.agents if not (env.world_state["phase"] == Phase.NIGHT and env.agent_roles[agent] == Roles.VILLAGER)}  # this is where you would insert your policy
        env.render()
        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.render() # post game render