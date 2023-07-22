# Werewolf - The Game

Mechanisms employed:
- Hidden Roles
- Negotiation
- Roles with Asymmetric information
- Voting
- Elimination

## Description

Werewolf is a multi player game between two competing teams comprised of villagers and werewolves. The werewolf team is a hidden subset of the villager team as true villagers cannot tell who the werewolves are, and must try and deduce their identities.

Gameplay alternates between day and night phases. During the day, villagers vote on who amongst them should be executed for the crime of being a werewolf, and at night, wolves vote on which helpless villager to kill.

Once a player dies, they reveal their role to the rest of the team.

Different versions of the game slightly alter rules, or introduce various new roles with their own special abilities. In our case, we have simplified the game as much as possible to focus on the voting mechanisms used by agents to determine who to execute and kill.

The game ends when either all the werewolves have been executed, or there are more werewolves than villagers remaining.

## History

Also known as MAFIA, it is a social game
https://boardgamegeek.com/boardgame/925/werewolf

https://en.wikipedia.org/wiki/Mafia_(party_game)

Created by Dimitry Davidoff in 1986 in Russia. Moscow State University, Psychology Department


Mafia is one of the 50 most historically and culturally significant tabletop games since 1800 according to about.com


Andrew Plotkin put the werewolf spin on the game in 1997

Deduction, Negotiation, Bluffing

The game models a conflict between two groups: an informed minority (the mafiosi or the werewolves) and an uninformed majority (the villagers)


## Implementation

### Pettingzoo

![PettingZoo logo](https://pettingzoo.farama.org/_images/pettingzoo-text.png)

While there are many routes to take when creating a custom environment, using a popular underlying framework makes reproducibility and reusability trivial. For this werewolf game a simple Multi-Agent Reinforcement Learning (MARL) API standard provided by [PettingZoo](https://pettingzoo.farama.org/) {cite}`terry2021pettingzoo` was chosen. The [Farama Foundation](https://farama.org/) which oversees PettingZoo's development is also in charge of mainting the popular [Gym (now Gymnasium)](https://gymnasium.farama.org/) {cite}`towers_gymnasium_2023` RL framework.


### Werewolf

#### 

(game-rewards)=
#### Rewards


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
Plurality

Approval

