from pettingzoo import AECEnv
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
    "win": 25,
    "loss": -25,
    "wolf_vote": 5,
    "vote_miss": -1,
    "self_vote": -3,
    "dead_vote": -5,
}

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)

class raw_env(AECEnv):

    metadata = {
        "render_modes" : ["human"],
        "name": "werewolf_approval_v1"
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
        self.infos = {agent: {} for agent in self.agents}
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def render(self, mode: str = "human"):
        """
        Renders the environment. In human mode, it prints to the terminal
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        if mode == "ret":
            return self.world_state
        print(json.dumps(self.world_state, indent=4))

    
    def close():
        """
        Needed for games with a render function
        """
        pass

    def _is_game_over(self) -> bool: 
        pass

    def _get_player_to_be_killed(self) -> tuple[int, bool, bool]:
        # we now get an approval chain, we want to vote out the agent with the lowest approval rating?
        # maybe we want to 
        # we want to vote out the player with the lowest score.
        # an approval should not count against a low score
        votes = [[0 if i == 1 else i for i in p_actions] for p_actions in self.votes.values()]
        votes = np.sum(votes, axis=0)

        min_indices = np.where(votes == min(votes))[0]
        dead_vote_flag = False
        tie_vote_flag = False

        # if we have a tie, keep the living players and randomly choose between them, report back a tie
        living_selections = [player for player in min_indices if f'player_{player}' not in self.dead_agents]

        if len(min_indices) > 1:
            tie_vote_flag = True


        if len(living_selections) < len(min_indices):
            dead_vote_flag = True

        # If we have any living selections, lets sampple one
        if len(living_selections) > 0:
            return random.choice(living_selections), dead_vote_flag, tie_vote_flag

        # keep going down the chain
        for next_best in np.argsort(votes)[len(min_indices):]:
            if f'player_{next_best}' not in self.dead_agents:
                return next_best, dead_vote_flag, tie_vote_flag
        print("Did I make it here?")
        

    def _step_day(self, action):
        return

    def _step_night(self, action):
        return 

    def step(self, action):
        ## everyone votes
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
            # self._was_done_step(action) does not actually set the next agent, so we should do it here
            # this also fails if everyone had DONES before hand
            # self.agent_selection = self._agent_selector.next()

        # Villagers cannot vote during nightime
        if self.world_state['phase'] == Phase.NIGHT:
             if self.agent_roles[self.agent_selection] != Roles.VILLAGER:
                    self.votes[self.agent_selection] = action
                    self.world_state['votes'][self.agent_selection] = action
        else:
            self.votes[self.agent_selection] = action
            self.world_state['votes'][self.agent_selection] = action

        # What is needed here?
        self._cumulative_rewards[self.agent_selection] = 0

        # if this is the last agent to go, kill whomever we need to kill
        if self._agent_selector.is_last():
            
            if self.world_state['phase'] != Phase.ACCUSATION:
                agent_id_to_die, was_dead_vote, was_tie_vote = self._get_player_to_be_killed()
                agent_to_die = self.possible_agents[agent_id_to_die]

                self.terminations[agent_to_die] = True
                self.dead_agents.append(agent_to_die)
                self.world_state['alive'].remove(agent_to_die)
            else:
                agent_id_to_die = -1
                agent_to_die = -1

            if self.world_state['phase'] == Phase.NIGHT:
                self.world_state['killed'].append(agent_to_die)
            elif self.world_state['phase'] == Phase.VOTING:
                self.world_state['executed'].append(agent_to_die)

            if not set(self.world_state["werewolves"]) & set(self.world_state['alive']):
                # print("Villagers WIN!!!!!")
                self.world_state['winners'] = Roles.VILLAGER
                self.terminations = {agent: True for agent in self.terminations}

            elif len(set(self.world_state["werewolves"]) & set(self.world_state['alive'])) >= \
                len(set(self.world_state["villagers"]) & set(self.world_state['alive'])):
                # print("Werewolves WIN!!!!")
                self.world_state['winners'] = Roles.WEREWOLF
                self.terminations = {agent: True for agent in self.terminations}

            # votes are in, append snapshot of world state to history
            self.history.append(copy.deepcopy(self.world_state))

            # figure out rewards for everyone 
            # extra logic for night turn with villagers not voting, so not getting rewards
            for agent in self.agents:
                if agent == agent_to_die:
                    self.rewards[agent] = REWARDS["death"]
                # elif agent in self.votes and agent_id_to_die != self.votes[agent] and self.world_state['phase'] != Phase.ACCUSATION:
                #     self.rewards[agent] = REWARDS["vote_miss"]
                
                # TODO: handle villager night votes better
                # Right now we will go ahead and just ignore anything a villager does at night
                # if not (self.world_state['phase'] == Phase.NIGHT and self.agent_roles[agent] == Roles.VILLAGER) and agent in self.votes:
                #     voted_for = f'player_{self.votes[agent]}'
                #     if self.world_state['phase'] != Phase.ACCUSATION:
                #         # determine if the agent voted for an already dead player
                #         if (voted_for in self.dead_agents) and (voted_for != agent_to_die):
                #             self.rewards[agent] += REWARDS["dead_vote"]
                    
                #         # penalize if agent voted for themselves
                #         if voted_for == agent:
                #             self.rewards[agent] += REWARDS["self_vote"]

            if False not in self.terminations.values():
                if self.world_state['winners'] != None:
                    for agent in self.agents:
                        if self.agent_roles[agent] == self.world_state['winners']:
                            self.rewards[agent] += REWARDS["win"]
                        else:
                            self.rewards[agent] += REWARDS["loss"]
            elif self.world_state['phase'] == Phase.NIGHT:
                for agent in self.agents:
                    self.rewards[agent] += REWARDS["day"]



            # phase is over, set votes to nothing, increment time_of_day and day accordingly
            self.votes = {}
            self.world_state['votes'] = {}

            if self.world_state['phase'] == Phase.NIGHT:
                self.world_state['day'] += 1
            self.world_state['phase'] =  (self.world_state['phase'] + 1) % 3

            # re-init logic here might be still needed
            self._agent_selector.reinit(self.world_state['alive'])
        else:
            # no rewards are allocated until all players give an action
            self._clear_rewards()

            # let user know 
            # if f'player_{action + 1}' in self.dead_agents:
            #     self.infos[self.agent_selection] = 

        if len(self._agent_selector.agent_order):
            self.agent_selection = self._agent_selector.next()
        
        self._accumulate_rewards()

        # option to call deads step first here possiblu
        # self._deads_step_first()


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

        self.votes = {agent: [] for agent in self.agents}

        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()

    
    def observation_space(self, agent: str) -> Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> Space:
        return self.action_spaces[agent]

    def observe(self, agent: str):
        
        # action: action 0 : no vote - cannot vote against a dead Agent
        action_mask = [agent not in self.dead_agents for agent in self.possible_agents]

        # player roles 
        if self.agent_roles[agent] == Roles.VILLAGER:
            # TODO : If a werewolf is dead, then reveal their role
            # villagers think everyone is a villager
            roles = [Roles.VILLAGER] * len(self.possible_agents)
        else:
            # werewolves know the true roles of everyone
            roles = list(self.agent_roles.values())

        # TODO: hide previous votes of werewolves if it was nighttime from villagers
        prev_state = self.history[-1]

        # Determine what should be shown for previous votes
        if self.agent_roles[agent] == Roles.VILLAGER and prev_state['phase'] == Phase.NIGHT:
            votes = {agent: [0 for _ in range(0, self.num_agents)] for agent in self.possible_agents}
        elif len(self.history) == 1:
            votes = {agent: [0 for _ in range(0, self.num_agents)] for agent in self.possible_agents}
        else: 
            votes = {agent: [0 for _ in range(0, self.num_agents)] if agent not in prev_state['votes'] else prev_state['votes'][agent] for agent in self.possible_agents}

        observation = {
            "day" : prev_state["day"],
            "phase": prev_state["phase"],
            "self_id": int(agent.split('_')[-1]),
            "player_status": action_mask,
            "roles": roles,
            "votes": votes
        }

        return {"observation": observation, "action_mask": action_mask}
    
    def convert_obs(self, observation):
        return  np.asarray([observation['day']] + \
        [observation['phase']] + \
        [observation['self_id']] + \
        [int(status) for status in observation['player_status']] + \
        [role for role in observation['roles']] + \
        [vote for vote in observation['votes']])


def random_policy(observation, agent):
    # these are the other wolves. we cannot vote for them either
    player_status = list(range(len(observation['observation']['player_status'])))
    # dead players
    action_mask = observation['action_mask']
    me = observation['observation']['self_id']

    legal_actions = [action for action,is_alive,is_wolf in zip(player_status, action_mask, observation['observation']['roles']) if is_alive and not is_wolf]
    # wolves don't vote for other wolves. will select another villager at random
    player = random.choice(legal_actions)

    action = [0] * len(action_mask)
    action[me] = 1
    action[player] = -1
    return action

def revenge_wolf_policy(observation, agent, action=None):
    # we already know the agent is a werewolf
    me = observation['observation']['self_id']

    # who voted for me 
    votes_against_me = [i for i, x in enumerate(observation['observation']['votes']) if x == -1 and i == me]

    # remove any wolves who voted for me (they should not have)
    wolf_ids = [i for i, x in enumerate(observation['observation']['roles']) if x == 1 and i != me]
    votes_against_me = list(set(votes_against_me)^set(wolf_ids))

    # remove any players who voted for me but are dead now
    votes_against_me = [i for i in votes_against_me if observation['observation']['player_status'][i] == True]

    villagers_alive = [i for i, x in enumerate(observation['observation']['roles']) \
        if observation['observation']['player_status'][i] == True and x == 0]

    # if there are no votes against me, pick a random villager that is alive
    choice = random.choice(votes_against_me) if len(votes_against_me) > 0 else random.choice(villagers_alive)
    choice = [-1] * len(observation['action_mask'])

    choice[me] = 1
    for wid in wolf_ids:
        choice[wid] = 1

    return action if action != None else choice

def random_wolf_policy(observation, agent, action=None):
    # pick a villager to vote for that is alive
    villagers_alive = [i for i, x in enumerate(observation['observation']['roles']) \
        if observation['observation']['player_status'][i] == True and x == 0]
    return action if action != None else random.choice(villagers_alive)


if __name__ == "__main__":

    # api_test(raw_env(), num_cycles=100, verbose_progress=True)

    env = raw_env(num_agents=10, werewolves=2)
    env.reset()

    wolf_brain = {'day': 1, 'phase': 0, 'action': None}

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        day = observation['observation']['day']
        phase = observation['observation']['phase']

        if wolf_brain['day'] != day or wolf_brain['phase'] != phase:
            wolf_brain = {'day': day, 'phase': phase, 'action': None}

        role = observation['observation']['roles'][observation['observation']['self_id']]

        if role == Roles.WEREWOLF:
            if wolf_brain['action'] != None:
                action = wolf_brain['action'] if not termination or truncation else None
            else:
                action = revenge_wolf_policy(observation, agent) if not termination or truncation else None
                wolf_brain['action'] = action
        else:
            action = random_policy(observation, agent) if not termination or truncation else None

        env.render()
        env.step(action)
    env.render()
    
    print("Done")