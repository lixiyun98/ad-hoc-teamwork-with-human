from gym.envs.registration import register

register(
    id='RPS-v0',
    entry_point='pantheonrl.envs.rpsgym.rps:RPSEnv'
)

register(
    id='LiarsDice-v0',
    entry_point='pantheonrl.envs.liargym.liar:LiarEnv'
)

register(
    id='OvercookedMultiEnv-v0',
    entry_point='pantheonrl.envs.overcookedgym.overcooked:'
                'OvercookedMultiEnv'
)

register(
    id='BlockEnv-v0',
    entry_point='pantheonrl.envs.blockworldgym.simpleblockworld:SimpleBlockEnv'
)

register(
    id='BlockEnv-v1',
    entry_point='pantheonrl.envs.blockworldgym.blockworld:BlockEnv'
)

register(
    id='PartnerBlockEnv-v0',
    entry_point='pantheonrl.envs.blockworldgym.simpleblockworld:PartnerEnv'
)