
import torch 
import numpy as np 
import copy
import mlflow
import sys
sys.path.append('../')
import random

from notebooks.learning_agents.trainer import PPOTrainer
from voting_games.werewolf_env_v0 import plurality_env

def random_coordinated_wolf(env, action=None):
    villagers_remaining = set(env.world_state["villagers"]) & set(env.world_state['alive'])
    wolves_remaining = set(env.world_state["werewolves"]) & set(env.world_state['alive'])

    target = random.choice(list(villagers_remaining))
    return {wolf: int(target.split("_")[-1]) for wolf in wolves_remaining}

def random_wolfs(env):
    return {wolf: env.action_space(wolf).sample() for
            wolf in set(env.world_state["werewolves"]) & set(env.world_state['alive'])}

def revenge_coordinated_wolf(env, actions = None):
    villagers_remaining = set(env.world_state["villagers"]) & set(env.world_state['alive'])
    wolves_remaining = set(env.world_state["werewolves"]) & set(env.world_state['alive'])

    # who tried to vote out a wolf last time?
    
    target = random.choice(list(villagers_remaining))
    # pick 
    for wolf in wolves_remaining:
        actions[wolf] = [0] * len(env.possible_agents)
        actions[wolf][int(target.split("_")[-1])] = -1
        for curr_wolf in wolves_remaining:
            actions[wolf][int(curr_wolf.split("_")[-1])] = 1
    # for wolf in env.werewolves_remaining:

def random_single_target_villager(env, agent):
    targets = set(env.world_state["alive"]) - set([agent])
    return int(random.choice(list(targets)).split("_")[-1])

# random_coordinated_wolf(env)
def random_agent_action(env, agent, action=None):
   return env.action_space(agent).sample()

def random_coordinated_single_wolf(env, agent, action=None):
    villagers_remaining = set(env.world_state["villagers"]) & set(env.world_state['alive'])
    return action if action != None else int(random.choice(list(villagers_remaining)).split("_")[-1])

config_training = {
    "model": {
        "recurrent_layers": 1, # 1,2 (2)
        "recurrent_hidden_size": 256, # 64-128-256-512 (4)
        "joint_mlp_size": 128,
        "split_mlp_size": 64,
        "num_votes": 1,
        "approval_states": 15 # this is tied to the number of players
    },
    "training" : {
        "batch_size": 256, # 32-64-128-256-512-1024 (6)
        "epochs": 3, # 4,5,6,7,8,9,10 (7)
        "updates": 301, # 1000 (1)
        "buffer_games_per_update": 200, # 50-100-200 (3)
        "clip_range": 0.1, # 0.1,0.2,0.3 (3)
        "value_loss_coefficient": 0.1, # 0.1, 0.05, 0.01, 0.005, 0.001 (5)
        "max_grad_norm": 0.5, 
        "beta": 0.01, # entropy loss multiplier # 0.1, 0.05, 0.01, 0.005, 0.001
        "learning_rate": 0.00001, # 0.001, 0.0005, 0.0001, 0.00005, 0.00001
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
        "num_agents": 15,
        "num_werewolves": 2,
    }
}

config = {
    "config_game": config_game,
    "config_training": config_training,
}

env = plurality_env(num_agents=config["config_game"]["gameplay"]["num_agents"], 
                    werewolves=config["config_game"]["gameplay"]["num_werewolves"], 
                    num_accusations=config["config_game"]['gameplay']["accusation_phases"], 
                    #rewards=self.config["config_game"]['rewards']
                    )

finished_one = False
for _ in range(50):
    try:
        trainer = PPOTrainer(env,
                             config=config,
                             wolf_policy=random_coordinated_single_wolf,
                             run_id="Plurality_15",
                             device=torch.device("cpu"),
                             mlflow_uri="http://mlflow:5000")
        trainer.train(voting_type="plurality", save_threshold=25.0)
        finished_one = True
    except ValueError as e:
        print("Probably a nan error")
        if ("nan" in str(e)):
            print("It was value errors")
        print("Trying again")
    finally:
        if finished_one == True:
            break