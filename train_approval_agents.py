
import torch 
import numpy as np 
import copy
import mlflow
import sys
sys.path.append('../')
import random

from notebooks.learning_agents.trainer import PPOTrainer
from voting_games.werewolf_env_v0 import pare

def random_wolf(env, agent, action=None):
    if action != None:
        return action

    villagers_remaining = set(env.world_state["villagers"]) & set(env.world_state['alive'])
    wolves_remaining = set(env.world_state["werewolves"]) & set(env.world_state['alive'])

    # pick a living target
    target = random.choice(list(villagers_remaining))

    action = [0] * len(env.possible_agents)
    action[int(target.split("_")[-1])] = -1
    for curr_wolf in wolves_remaining:
        action[int(curr_wolf.split("_")[-1])] = 1

    return action

def aggressive_wolf(env, agent, action=None):
    wolves_remaining = set(env.world_state["werewolves"]) & set(env.world_state['alive'])
    action = [-1] * len(env.possible_agents)
    for curr_wolf in wolves_remaining:
        action[int(curr_wolf.split("_")[-1])] = 1

    return action


def revenge_coordinated_wolf(env, actions = None):
    villagers_remaining = set(env.world_state["villagers"]) & set(env.world_state['alive'])
    wolves_remaining = set(env.world_state["werewolves"]) & set(env.world_state['alive'])

    # who tried to vote out a wolf last time?
    # TODO:
    return None
    # for wolf in env.werewolves_remaining:

def random_single_target_villager(env, agent):
    targets = set(env.world_state["alive"]) - set([agent])
    action = [0] * len(env.possible_agents)
    action[int(agent.split("_")[-1])] = 1
    action[int(random.choice(list(targets)).split("_")[-1])] = -1

    return action
    # for villager in env.villagers_remaining:

# random_coordinated_wolf(env)
def random_agent_action(env, agent, action=None):
   return env.action_space(agent).sample().tolist()

config_training = {
    "model": {
        "recurrent_layers": 1, # 1,2 (2)
        "recurrent_hidden_size": 256, # 64-128-256-512 (4)
        "mlp_size": 256, # 64-128-256-512 (4)
        "num_votes": 10,
        "approval_states": 3,
    },
    "training" : {
        "batch_size": 256, # 32-64-128-256-512-1024 (6)
        "epochs": 5, # 4,5,6,7,8,9,10 (7)
        "updates": 501, # 1000 (1)
        "buffer_games_per_update": 500, # 50-100-200 (3)
        "clip_range": 0.1, # 0.1,0.2,0.3 (3)
        "value_loss_coefficient": 0.1, # 0.1, 0.05, 0.01, 0.005, 0.001 (5)
        "max_grad_norm": 0.5, 
        "beta": 0.01, # entropy loss multiplier # 0.1, 0.05, 0.01, 0.005, 0.001
        "learning_rate": 0.0001, # 0.001, 0.0005, 0.0001, 0.00005, 0.00001
        "adam_eps": 1e-5, # 1e-8, 1e-7. 1e-6, 1e-5
        "gamma": 0.99, # 0.99
        "gae_lambda": 0.95, #0.95-0.99
    }
}


config_game = {
    "rewards": {
        "day": -1,
        "player_death": -1,
        "player_win": 10,
        "player_loss": -5,
        "self_vote": -1,
        "dead_villager": -1,
        "dead_vote": -1,
        "dead_wolf": 5,
        "no_viable_vote": -1,
        "no_sleep": -1,
    },
    "gameplay": {
        "accusation_phases": 3, # 2,3
        "num_agents": 10,
        "num_werewolves": 2,
    }
}

config = {
    "config_game": config_game,
    "config_training": config_training,
}

env = pare(num_agents=config["config_game"]["gameplay"]["num_agents"],
            werewolves=config["config_game"]["gameplay"]["num_werewolves"],
            num_accusations=config["config_game"]['gameplay']["accusation_phases"], 
            #rewards=self.config["config_game"]['rewards']
            )

finished_one = False
for _ in range(50):
    try:
        trainer = PPOTrainer(env,
                             config=config,
                             wolf_policy=random_wolf,
                             run_id="Approval_256",
                             device=torch.device("cpu"),
                             mlflow_uri="http://mlflow:5000",
                             )
        trainer.train(voting_type="approval", save_threshold=25)
        finished_one = True
    except ValueError as e:
        if ("nan" in str(e)):
            print("It was value errors, trying again")
    finally:
        if finished_one == True:
            break