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

The initialization signature aboe highlights the $K$ amount of  accusation phases set to a default value of 1, along with 
a custom reward object that [defaults to the following values](game-rewards). These values were partially based off of work done in RLupus {cite}`Brandizzi2021RLupusCT`

```{note}
square root of the number of agents is the max allowed werewolf count given game theoretic advantages for werewolves {cite}`braverman2008mafia`
```

A permutation of roles (villager or werewolf) based on the initialization values is selected and assigned to the $N$ agents playing the game. 


(roles-phases)=
### Roles and Phases

We currently implement only two roles, villager or werewolf to simplify the environment for analysis.

```python
class Roles(enum.IntEnum):
    VILLAGER = 0
    WEREWOLF = 1
```

Currently, the day consists of $K$ accusation phases and one voting phase, while the night is its own phase as wolf policies are static.

```python
class Phase(enum.IntEnum):
    ACCUSATION = 0
    VOTING = 1
    NIGHT = 2
```

During accusastion phases, agents vote as usual, but nobody dies. It is a way to "finger-point" and allow agents to broadcast their intent.

(env-spaces)=
### Action Space
For both plurality and approval voting mechanisms, the action space is their vote.

```python
# plurality
self.action_spaces = { name: Discrete(num_agents)  for name in self.agents }
```

In the case of plurality voting, the action space is an integer between $[0,N]$ where a vote represented by the integer $N$ represents a null vote. This is because agent IDs start at $0$ and the vote is simply their ID represented as an integer.

```python
# approval
self.action_spaces = { name: Box(low=-1, high=1, shape=(num_agents,), dtype=int) for name in self.agents }
```

In approval voting, the action space is now represented as a multi-discrete integer array of length $N$, where elements can have one of three values:
- $-1$ : dissaproval
- $0$: neutral
- $1$ : approval

```{warning}
For in-game logic and calculation, the $-1$ acts like a plurality vote, where the index of the $-1$ is the integer identifier of the taget. Other values are not used to determine voting outcomes.
```

### Observation Space 

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

Every environment step, players are provided an observation object and an action mask representing dead or alive players. The player status and action masks are identical, however we have chosen to seperate them like this for more clarity and to conform to more training APIs.

As we have implemented our own PPO loop, we have a seperate [utility function](convert-obs) that we use to convert the observations into an input vector for our neural networks.

In the observation object we return:
- `day`: an integer representing the current day, starting at $1$

```{note}
Because at the end of a day $t$ we have $N_t = N - 2t$ players left, after half the agents die, one of our win conditions will be met, so we can assume the maximum amount of days will be roughly $N/2$.
```

- `phase`: an integer, corresponding to the current [phase](roles-phases) agents are acting in.
- `self_id`: an integer between $0,N-1$ representing the current agents ID.
- `player_status`: an array where each index corresponds to an agent ID, and the value, either $0$ or $1$ signifies if that agent is alive or dead
- `roles`: an array where each index corresponds to an agent ID, and the value is represented by our [Roles enum](roles-phases). $0$ represents a villager, $1$ a werewolf.

```{note}
As wolves have a role represented by $1$, the only time a villager will see an array with anything other than $0$'s is when a wolf gets killed and has to reveal their role.
```

- `votes`: this is a dictionary keyed by a string `player_{n}` where $n$ is the integer corresponding to the player ID. Each value is the vote of that corresponding player, which is either an integer in the case of plurality voting, or an array in the case of approval voting. These values are derived from the action space of agents.

```{warning}
During the first accusation phase of a day, villagers are not shown the true votes the wolves have taken overnight. Instead they are shown either all votes being $N$, the null vote in the case of plurality voting, or an array of all $0$ for approval voting, representing neutral feelings.
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

The state object is more comprehensive than the observation an agent receives given that this game is represented by a POMDP. It was also designed to make parsing game history easier and to render on-screen via `env.render` while debugging early on in the project.

This state object tracks the following:
- `day`: an integer representing the current day, starting at $1$
- `phase`: an integer, corresponding to the current [phase](roles-phases) agents are acting in.
- `round`: an integer between $[0,K)$ representing the current accusation round out of $K$ repititions.
```{warning}
We chose **not** to return this as an observation as it was added later in the project and our results did not change. In fact, it could act as a reason to remove the day value as well, since actions should be tied more strongly to the `phase`
```
- `alive`: an array containing all agents that are still alive
- `killed`: an array representing agents killed by werewolves
- `executed`: an array representing agents executed by group consensus
- `werewolves_remaining`: an array containing the remaining werewolves
- `villagers_remaining`: an array containing the remaining villagers, excluding werewolves
- `votes`: this is a dictionary keyed by a string `player_{n}` where $n$ is the integer corresponding to the player ID. Each value is the vote of that corresponding player, which is either an integer in the case of plurality voting, or an array in the case of approval voting. These values are derived from the action space of agents.
- `winners`: If the game is over, this will either be a $0$ for villagers, or $1$ for werewolves. Otherwise, it will be `null`. It is mainly used as a flag for the main game loop in `env.step`.


### Voting Mechanisms

Eliminating a player requires gathering and determining a group consensus. Each environment (plurality and approval) has their own function to determine who that target consensus is. 

```python
def _get_player_to_be_killed(self, actions) -> tuple[int, object]:
```

This function also returns an information object with flags for each agent if they voted for themselves, a dead player, or if they had a viable vote. It is this object that is used further inform the rewards individual agents receive each step taken in the environment.

```{admonition} Ties
For both approval and plurality voting, if ties exist, a target is chosen by breaking ties randomly. If however the target with the majority of votes is a dead player, the next highest targetted agent is taken.
```

(game-rewards)=
### Rewards

Reward shaping in RL (to bias towards certain desired behaviors) is very hard, and can usally lead to unintended results.
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

The rewards object is comrpised of:
- `day`: a negative reward trying to incentize agents to complete the game faster
- `player_death`: a negative reward given to the player who was either killed or executed. _This could be further seperated based on killing or execution._
- `player_win`: the reward given to a player who has won the game
- `player_loss`: the reward given to losing players.
- `dead_wolf`: a reward given to villagers who were able to execute a werewolf
- `dead_villager`: a negative reward given to villagers if an actual villager was executed
- `self_vote`: a negative reward given if a player targets themselves
- `dead_vote`: a negative reward given if a player targets another dead player.
- `no_viable_vote`: An additional penalty if the vote was not a viable one
- `no_sleep`: This penalty is not currently implemented, but was intended to shape villager learning towards selecting the appropriate `null` action during night votes.

```{admonition} Training Rewards
Currently, we only further distribute a fractional win/loss reward to dead players during [training](agent-training-rewards), however targetting dead players may be another reward we want to punish with a multiplicative effect proportional to the amount of villagers voting for a dead player.
```

### Game loop

As seen in the [flowchart](game-flow), the logic is all implemented in the `env.step()` function which expects a dictionary of agent ids and their respective actions.

For every phase, the target is determined via `self._get_player_to_be_killed(actions)`. If this was a voting or night-time phase, the target is added to executed (voting) or killed (night-time) lists, and this dying agent is given a negative reward for dying.

```{warning}
During the night-time, we still allow actions from villagers, however these are completely ignored. 
```

We then check for winners and reward players accordingly, before appending the history and possibly incrementing the game round, phase and day. 

Finally the remainder of rewards based on agent voting information is distributed to the agents that provided actions during this current environment step.