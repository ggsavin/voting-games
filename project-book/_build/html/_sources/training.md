# Training 

## Agent Design
We use Independent Learners with a shared policy for all the agents vs a JAL because we  dont want action space to scale

Use pytorchviz or torchview to visualize the pytorch graph



### Utilities

(convert-obs)=
#### Convert observation to vector


(game-analysis-methodology)=
## Analysis

With MLP sizes of 128, 128, hard to learn

We need at least an LSTM of 256 and 256 MLP hidden

We use Independent Learners with a shared policy for all the agents vs a JAL because we  dont want action space to scale

INitializations and hyper-parameter choices were greatly influenced by the exhaustive work in {cite}`Andrychowicz2020-fs` and {cite}`pleines2022memory` for the LSTM portion.

### Hyperparameters

### Reward Shaping
