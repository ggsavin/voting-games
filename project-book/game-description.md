(game-desc)=
# Werewolf - The Game

## Quick History
Werewolf is a 1997 re-imagining by Andrew Plotkin [^werewolf-boardgame] of a game known as mafia[^mafia-wikipedia], credited to Dimitry Davidoff in the Psychology Department of Moscow State University in 1986.

The game models a conflict between an informed minority (mafiosi) and an unaware majority(villagers).

Originally created for psychological research, it has spread all over the world and is now in the top 50 most historically and culturally significant games published[^top50]

## Description

Werewolf is a multi player game between two competing teams comprised of villagers and werewolves. The werewolf team is a hidden subset of the villager team as true villagers cannot tell who the werewolves are, and must try and deduce their identities.

```{admonition} Team Composition
There are a total of $N$ players, where $M$ belong to the werewolf team and the remaining $N-M$ belong to the villager team. At the beginning of the game, each player is given their own identity (werewolf or villager) and every werewolf is also given the identity of all other werewolves.
```

Gameplay alternates between day and night phases. During the day, villagers vote on who amongst them should be executed for the crime of being a werewolf, and at night, wolves vote on which helpless villager to kill. Communication between agents either takes place during the voting round, or precedes it in multiple communication rounds. Once a player dies, they reveal their role to the rest of the players.

```{admonition} Gameplay Loop
The way we have implemented Werewolf consists of interations of the following three phases until the game terminates.

1. **Accusation Phase**: Villagers and Werewolves make their vote known and broadcast it to all other players. There are $K$ accusation rounds before the final decision and vote is taken. Because we have removed any explicit communication channel, we allow for this variable length of accusation rounds so agents can focus on the votes.

2. **Vote Phase**: All players vote, and based on the chosen voting mechanism (plurality or approval), a player is selected to be eliminated, and will reveal their role to the rest of the players. 


3. **Night Phase**: Wolves select which player to kill without leaking any information to the rest of the villagers. All villagers see is a null action, but the identity of the dead player is revealed along with their game role.


A round $t$ starts in the accusation phase where there are $N_t = N - 2t$ villagers. $W_t$ and $V_t$ denote the numbers of werewolves and villagers at the start of round $t$ respectively.


**Win conditions**:
- Villagers win when all Werewolves are killed, $W_t = 0$
- Werewolves win when there are more werewolves than villagers $W_t \geq V_t$.

Whenever $W_t = V_t$, villagers need to force a tie and get a lucky break to avoid a loss.
```

Different versions of the game slightly alter rules, or introduce various new roles with their own special abilities. In our case, we have simplified the game as much as possible to focus on the voting mechanisms used by agents to determine who to execute and kill.

The game ends when either all the werewolves have been executed, or there are more werewolves than villagers remaining.




[^werewolf-boardgame]: https://boardgamegeek.com/boardgame/925/werewolf
[^mafia-wikipedia]: https://en.wikipedia.org/wiki/Mafia_(party_game)
[^top50]:https://www.thesprucecrafts.com/board-and-card-games-timeline-409387