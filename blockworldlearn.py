from multiagentworld.envs.blockworldgym.blockworld import CONSTRUCTOR_OBS_SPACE
import gym
import time
import multiagentworld.envs.blockworldgym.simpleblockworld as sbw
import multiagentworld.envs.blockworldgym.blockworld as bw
from stable_baselines3 import PPO


from multiagentworld.common.agents import OnPolicyAgent, OffPolicyAgent
from multiagentworld.common.wrappers import TurnBasedFrameStack

print(sbw.PLANNER_OBS_SPACE)
print(sbw.CONSTRUCTOR_OBS_SPACE)
# env = gym.make("multiagentworld:BlockEnv-v0")
env = gym.make("multiagentworld:BlockEnv-v1")
env.add_partner_agent(bw.DefaultConstructorAgent())
# env.add_partner_agent(sbw.SBWEasyPartner())
# env.add_partner_agent(OnPolicyAgent(PPO("MlpPolicy", env.partner_env)))

# model = PPO("MlpPolicy", env, verbose=1)
# wrappedmodel = OnPolicyAgent(model)
# wrappedmodel.learn(total_timesteps=1000000)

env.reset()
for i in range(10):
    env.render()
    env.step(i)
    time.sleep(2)