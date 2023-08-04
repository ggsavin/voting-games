# Werewolf - The Implementation

## Pettingzoo

![PettingZoo logo](https://pettingzoo.farama.org/_images/pettingzoo-text.png)

While there are many routes to take when creating a custom environment, using a popular underlying framework makes reproducibility and reusability trivial. For this werewolf game a simple Multi-Agent Reinforcement Learning (MARL) API standard provided by [PettingZoo](https://pettingzoo.farama.org/) {cite}`terry2021pettingzoo` was chosen. The [Farama Foundation](https://farama.org/) which oversees PettingZoo's development is also in charge of mainting the popular [Gym (now Gymnasium)](https://gymnasium.farama.org/) {cite}`towers_gymnasium_2023` RL framework.


## Werewolf

(game-flow)=
### Game flow overview

![gameplay flowchart](images/werewolf-flowchart.svg)

### Start of the game
```python
def __init__(self, num_agents=5, werewolves=1, num_accusations=1, rewards=REWARDS)
```

We allow for a variable number of agents as well as number of accusation rounds before an execution vote. We also allow for custom reward values along with [default values](game-rewards)

```{note}
square root of the number of agents is the max allowed werewolf count given game theoretic advantages for werewolves {cite}`braverman2008mafia`
```

A permutation of roles given the above choice is taken, and assigned to players.

### Roles and Phases
There are only two roles, werewolves and villagers

```python
class Roles(enum.IntEnum):
    VILLAGER = 0
    WEREWOLF = 1
```

Currently, the day consists of $n$ accusation phases and one voting phase, while the night is its own phase as wolf policies are static.

```python
class Phase(enum.IntEnum):
    ACCUSATION = 0
    VOTING = 1
    NIGHT = 2
```

During accusastion phases, agents vote as usual, but nobody dies. It is a way to "finger-point" and allow agents to broadcast their intent.

(env-spaces)=
### Action Space and Observation Space

The action space is simply the vote of the agent. In the plurality version, it is the player ID the agent wishes to taget. In the approval setting, it is a multi-discrete array of length #-players, where each value is (-1 - dissaproval, 0 - neutral, 1 - like)

```python
# plurality
self.action_spaces = { name: Discrete(num_agents)  for name in self.agents }

# approval
self.action_spaces = { name: Box(low=-1, high=1, shape=(num_agents,), dtype=int) for name in self.agents }
```

The observation space returned has quite a bit more information, and a seperate [utility function](convert-obs) is used to convert this observation in the training of agents.

The longest possible run-time of the game is chosen as the upper-bound for days

A self-id is also provided, as the agent has to have a way to know which id is theirs.

The votes are the previous phases votes, and for villagers after a night round, hides the information as an invalid target.

Roles are also returned as all villagers for other villagers, except for instances were a wolf was killed, then their role is revealed as per the rules of the game.

```python
        self.observation_spaces = {
            name: Dict(
                {
                    "observation": Dict({
                        "day": Discrete(int(num_agents/2), start=1),
                        "phase": Discrete(3),
                        "self_id": Discrete(num_agents), # TODO: FINISH THIS
                        "player_status": Box(low=0, high=1, shape=(num_agents,), dtype=bool),
                        "roles": Box(low=0, high=1, shape=(num_agents,), dtype=int), 
                        "votes" : Dict({
                            name: Box(low=-1, high=1, shape=(num_agents,), dtype=int) for name in self.agents}), # approval
                            name: Box(low=0, high=num_agents, shape=(num_agents,)) for name in self.agents}), # plurality
                    }),
                    "action_mask": Box(low=0, high=1, shape=(num_agents,), dtype=bool)
                }
            )
            for name in self.agents
        }
```

### Game state

```python
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
```

The world state tracks everything happening in the game and is used for both the history and for rendering the game. In the case of game history for [analysis purposes](game-analysis-methodology) the votes are the actual votes that occured in the current phase, unlike observations which contain the previous phases votes.

### Voting Mechanisms
The following function returns the target id as an integer as well as an information object detailing which agent voted for themselves, voted for a dead player, had a viable vote, etc...

```python
def _get_player_to_be_killed(self, actions) -> tuple[int, object]:
```

For both approval and plurality voting, if ties exist, a target is chosen by breaking ties randomly. If however the target with the majority of votes is a dead player, the next highest targetted agent is taken.


(game-rewards)=
### Rewards

Reward designing to try to shape behavior and outcomes is very hard, and can usally lead to unintended behavior that still optimizes *cite* (look at inverse RL paper for these citations, use the upside-down helicopter picture)

The following object is the default reward structure for the game, and it was based on reward structures found in other Werewolf focused papers {cite}`Brandizzi2021RLupusCT, Velikov2021-vt`.


```python
REWARDS = {
    "day": -1,
    "player_death": -1,
    "player_win": 10,
    "player_loss": -5,
    "dead_wolf": 5,
    "dead_villager": -1,
    "self_vote": -1,
    "dead_vote": -1,
    "no_viable_vote": -1,
    "no_sleep": -1,
}
```

### Game loop

As seen in the [flowchart](game-flow), the logic is all implemented in the `env.step()` function which expects a dictionary of agent ids and their respective actions.

For every phase, the target is determined via `self._get_player_to_be_killed(actions)`. If this was a voting or night-time phase, the target is added to executed (voting) or killed (night-time) lists, and this dying agent is given a negative reward for dying.

```{warning}
During the night-time, we still allow actions from villagers, however these are completely ignored. 
```

We then check for winners and reward players accordingly, before appending the history and possibly incrementing the game round, phase and day. 

Finally the remainder of rewards based on agent voting information is distributed