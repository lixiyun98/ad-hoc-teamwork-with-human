import os
import gym
from stable_baselines3 import PPO, DQN

from multiagentworld.envs.overcookedgym.overcooked import OvercookedEnv
from multiagentworld.common.agents import OnPolicyAgent, OffPolicyAgent

env = gym.make("OvercookedMultiEnv-v0", layout_name="random0")
env.add_partner_policy(OnPolicyAgent(PPO("MlpPolicy", env)))
# env.add_partner_policy(OffPolicyAgent(DQN("MlpPolicy", env)))

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
# model.save("ppo_rps")