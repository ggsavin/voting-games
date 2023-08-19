


# Conclusion and Future Directions

## Conclusion

Hidden Role games allow for rich interactions between competing groups of players with varying access to information. This uneven playing field leads to uneasy cooperatiion and perverse attempts at deception and guile. The identification of traitors and allies by the uninformed majority player group has so far focused on their communication, overlooking the powerful contribution voting mechanisms play in revealing the true intentions of their voters. Our approval voting mechanism is a novel twist to the voting mechanics of werewolf, and hidden role games in general. We were able train agents to play successfully, and in tandem with our heuristic indicators, were able to classify their voting behaviors. 

In a recent case for approval voting {cite}`Hamlin2022TheCF`, albeit in an elector setting, the arguments were multi-faceted. It is simplistic enough to supercede plurality, with almost no additional mental load on voters, while also doing a better job at selecting a strong winner and has been proven to be able to select Condorcet winners. It also has a higher accuracy relative to honest assessments of voters. Our project extends this claim to hidden role games, and gives a less informed majority a better chance against an informed antagonistic minority group.

Given our findings and our easy to use and up to date environment, we hope future researchers can extend or incorporate ideas found here in their own work.

## Future Directions

There are many different avenues that can be taken but not to limited extending the environment with different voting mechanisms, exploring and creating more indicators and focusing on the neural network model architecture and associated hyperparameter choices. 
We list out a couple of ideas in no particular order: 
- What are optimal strategies in these self-playing scenarios where wolves and villagers learn to fool and indentify themselves? What voting mechanisms make this easier for villagers and which ones give werewolves an even greater advantage?
- How do we improve our observation and state space to produce more robust and consistent training?
- Make the environment work well with a training framework such as Tianshou[^Tianshou] so more MARL algorithms can be employed and tested.
- Add roles that have access to more information, or have different actions available to them.
- Extend the game to allow explicit communication.
    - How would this impact the use of signaling in voting? Do they work in tandem, or are there novel interactions to bypass detection?
- How do these findings scale with larger game sizes?
- Mix different villager and werewolf policies in a game, and see how trained agents perform. Also train agents in these mixed settings.
- Adding more indicators
    - Measuring commitment. how likely a villager is to stay with a vote throughout phases.
    - Dynamics of how a vote transitions from approval to neutral to dissaproval. 

[^Tianshou]:https://github.com/thu-ml/tianshou