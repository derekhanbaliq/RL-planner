import argparse
import os
import random

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from f110_rlenv import F110Env_Continuous_Planner

def make_env(gamma=0.99):

    def thunk():
        # env = F110Env_Continuous_Planner()
        env = F110Env_Continuous_Planner(T=1)
        
        env.f110.add_render_callback(env.main_renderer.render_waypoints)
        env.f110.add_render_callback(env.opponent_renderer.render_waypoints)
        
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk
    
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env() for i in range(1)]
    )
    # import ipdb; ipdb.set_trace()
    # envs = make_env()()
    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    
    model_path = "/home/oem/Documents/School/ESE_615/RL-planner/pure_pursuit/runs/F1Tenth-Planner__ppo_continuous__1__1682911130/24_model.pt"

    agent = Agent(envs).to(device)
    model = torch.load(model_path)
    agent.load_state_dict(model["model_state_dict"])

    # ALGO Logic: Storage setup

    # TRY NOT TO MODIFY: start the game
    next_obs = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    done = False

    while not done:

        # ALGO LOGIC: action logic
        with torch.no_grad():
            # action here is the "betterPoint"
            action, logprob, _, value = agent.get_action_and_value(next_obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, infos = envs.step(action.cpu().numpy())
            next_obs, done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            envs.envs[0].render(mode='human')