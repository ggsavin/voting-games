# The Game - Werewolf

Mechanisms employed:
- Hidden Roles
- Negotiation
- Roles with Asymmetric information
- Voting
- Elimination


## Description

Werewolf is a multi player game between two teams comprised of villagers and werewolves. The werewolf team is a hidden subset of the villager team as true villagers cannot tell who the werewolves are. 

Each day players (villagers and werewolves) decide on who amongst them should be executed for the crime of being a werewolf, and at night, the true werewolves kill a helpless villager.

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

While there are many routes to take when creating a custom environment, using a popular underlying framework makes reproducibility and reusability trivial. For this werewolf game a simple API standard provided by [PettingZoo](https://pettingzoo.farama.org/) {cite}`terry2021pettingzoo` was chosen. The Farama Foundation which oversees PettingZoo's development is also in charge of mainting the popular Gym (now Gymnasium) RL framework.

### Pettingzoo

### Werewolf

Plurality

Approval