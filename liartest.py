import gym
import liargym

env = gym.make('LiarsDice-v0')
policy = liargym.DefaultLiarAgent()
env.add_partner_policy(liargym.DefaultLiarAgent())

numgames = 100
rewards = []
for game in range(numgames):
    obs = env.reset()
    done = False
    while not done:
        action = policy.get_action(obs, False)
        obs, reward, done, _ = env.step(action)
    rewards.append(reward)

obs = env.reset()
print("Ego obs:", env.getObs(True))
print("Alt obs:", env.getObs(False))
done = False
while not done:
    action = policy.get_action(obs, False)
    print(action)
    obs, reward, done, _ = env.step(action)
print("ENDING")
print("Ego obs:", env.getObs(True))
print("Alt obs:", env.getObs(False))
env.close()
print(f"numwin: {len([x for x in rewards if x == 1])/numgames}")
print(f"numlose: {len([x for x in rewards if x == -1])/numgames}")
