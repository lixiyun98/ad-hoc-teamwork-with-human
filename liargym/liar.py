import gym
import common
import numpy as np
from gym.envs.registration import register

N = 6 # Num sides per dice
M = 6 # Num dice per player

WIN = (1, -1)
LOSE = (-1, 1)

MAX_MOVES = 2 * M

BLUFF = [N, 2 * M - 1]
DEFAULT = [N, 0]

ACTION_SPACE = gym.spaces.MultiDiscrete([N + 1, 2 * M])
OBS_SPACE = gym.spaces.MultiDiscrete([M + 1] * N + [N + 1, 2 * M] * MAX_MOVES)

def randRoll():
    dice = []
    for i in range(M):
        dice.append(np.random.randint(N))
    return [dice.count(i) for i in range(N)]

class DefaultLiarAgent(common.Agent):

    def get_action(self, obs, recording=True):
        obs = obs.tolist()
        hand = obs[:N]
        maxval = max(hand)
        count = hand.index(maxval)
        if obs[N] != N and obs[N+1] > maxval:
            return np.array(BLUFF)
        return np.array([count, maxval])

    def update(self, reward, done):
        pass

class LiarEnv(common.TurnBasedEnv):

    def __init__(self):
        super(LiarEnv, self).__init__(probegostart=0.5)
        self.history = []
        self.observation_space = OBS_SPACE
        self.action_space = ACTION_SPACE

    def getObs(self, isego):
        prevmove = self.history + DEFAULT * (MAX_MOVES - len(self.history) // 2)
        return np.array((self.egohand if isego else self.althand) + prevmove)

    def sanitize_action(self, action):
        if action[0] == N:
            return BLUFF

        if len(self.history) != 0 and action[1] <= self.history[1]:
            return BLUFF

        return action.tolist()

    def eval_bluff(self):
        if len(self.history) == 0:
            return False

        side = self.history[0]
        trueans = self.egohand[side] + self.althand[side] - 1
        return self.history[1] > trueans

    def player_step(self, action, isego):
        action = self.sanitize_action(action)
        if action == BLUFF:
            didwin = (self.eval_bluff() == isego)
            return self.getObs(not isego), WIN if didwin else LOSE, True, {}
        self.history = action + self.history
        return self.getObs(not isego), (0,0), False, {}


    def ego_step(self, action):
        """
        Return partner's obs, both rewards, is done, and info
        """
        return self.player_step(action, True)

    def alt_step(self, action):
        """
        Return ego's obs, both rewards, is done, and info
        """
        return self.player_step(action, False)

    def multi_reset(self, egofirst):
        self.history = []
        self.egohand = randRoll()
        self.althand = randRoll()

        return self.getObs(egofirst)



register(
    id='LiarsDice-v0',
    entry_point='liargym.liar:LiarEnv'
)
