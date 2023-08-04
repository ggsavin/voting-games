# COMP 5903 - Voting in Hidden Role Games

## Abstract

Social Deduction Games (SDGs), and especially Hidden Role variants such as Werewolf provide players with challenging and dynamic gameplay as teams with asymmetric information try to win. This imbalance of information favors the smaller, deceptive team as the majority try to figure out who the traitors are before it is too late. While player communication and elucidation of traitors has been a main focus of research, the voting mechanism underpinning the games is quite overlooked and is in general computationally challenging in Multi-Agent settings {cite}`Dodevska2019ComputationalSC`. In this project, we substitute the usual plurality scheme with an approval one in a custom Werewolf environment, and show that it is possible to train approval agents without using communication. We hope this spurs more social choice theory research in SDGs.


## Introduction

Social Deduction Games model tentative cooperation between groups with uncertain motives. In these games, different players and teams/coalitions have access to different information, with their goals ranging from cooperative to atagonistic{cite}`Dafoe2020-ds`. A special type of SDG is a Hidden Role game where the smaller team of players with an information advantage over the majority group also have their roles masked to "blend" into this majority. It is then up to the uniformed majority to build trust amongst themselves while ferreting out the deceptors. Well known games such as Amongst Us, Avalon, Secret Hitler and Mafia fall in this category. 

```{admonition} Mechanisms employed in Mafia/Werewolf
{bdg-primary}`Hidden Roles`, {bdg-primary}`Negotiation`, {bdg-primary}`Asymmetric information`, {bdg-primary}`Voting`, {bdg-primary}`Elimination`
```
Given this dichotomy and duplicity, SDGs are ripe for psychological analysis, and indeed, mafia was first played in the Psychology Department of Moscow State University[^mafia-wikipedia]. A gamut of emotions are experienced playing these types of games, from catharsis when identifying deceptors, to perverse joy from lying, cheating and manipulating[^amongst-us-article]. Other psychological uses of SDGs have been to treat anxiety through symbolic interaction {cite}`Conner2022-iu`, quantifying entertainment dynamics of SDGs{cite}`Ri2022-ih` and studying non-verbal communication {cite}`katagami2014investigation`.

```{admonition} What is Mafia/Werewolf?
Mafia is a hidden role game where a team of villagers try to guess who the mafia members are (by voting to eliminating players), while the actual mafia team who are posing as villagers try to eliminate all the villagers. Mafia players have an advantage because they can both vote with the villagers as well as during their own hidden turn. Villagers on the other hand do not know who the mafia members, so they need to communicate and synthesize misleading information to identify mafiosos and win the game.

Werewolf is simply a more recent re-imagining of the mafia game where the mafia members are swapped with werewolves, and the hidden turn is during the night when villagers sleep, and werewolves plot who to kill.

More information can be found in the game [description](game-desc).
```

IN SDGs, communication and deception present unique challenges for AI agents trying to learn optimal strategies on top of the already challenging Multi-Agent-RL (MARL) setting {cite}`Dafoe2020-ds`. To better study these topics along with machine learning, multi-agent simulation, human-agent interaction and cognitive science, an yearly competition called AIWolf {cite}`toriumi2017ai` was started that is still being held to this day. Werewolf, the game played in this competition is simply a re-imagining of Mafia where the mafia are now werewolves, and villagers vote to eliminate who they believe are werewolves during the day.

## Motivation

Social Choice Theory[^soc-choice] underpins a core mechanism of all SDGs in the gathering and enforcment of choices made by each individual agent. In all the works we have looked through, it seems as though this aspect of SDGs have not been permuted and explored. *By changing the underlying voting mechanism, we are interested in seeing how the dynamics between groups change*

Approval voting has more implicit and explicit information, and allows for more expression when voting. Naturally, we ask ourselves what kind of impact changing the voting mechanism underlying werewolf would have on the ability of the simulated agents to learn and their chances of winning.

## Contribution

- Creation of a custom environment using PettingZoo[^petting-zoo] for the purpose of voting mechanism research within the Werewolf SDG. 
- Sucessfully implemented approval voting as the voting mechanism in our environment
- Successfully trained plurality and approval RL agents in our environment
- Designed a wide range of indicators to explore behavior and voting patterns of agents


[^mafia-wikipedia]: https://en.wikipedia.org/wiki/Mafia_(party_game)
[^amongst-us-article]: https://www.sportskeeda.com/among-us/the-psychology-among-us
[^soc-choice]: https://en.wikipedia.org/wiki/Social_choice_theory
[^petting-zoo]: https://pettingzoo.farama.org/index.html