import argparse
import json
from time import sleep
from stable_baselines3 import PPO
import gym
import numpy as np
from pantheonrl.common.agents import OnPolicyAgent, StaticPolicyAgent
import gym
from trainer import (generate_env, gen_fixed, gen_fixed2, gen_fixed3, gen_default, EnvException,
                     ENV_LIST, ADAP_TYPES, LAYOUT_LIST)
from pantheonrl.common.agents import OnPolicyAgent
EGO_LIST = ['PPO', 'ModularAlgorithm', 'BC'] + ADAP_TYPES
PARTNER_LIST = ['PPO', 'DEFAULT', 'BC'] + ADAP_TYPES


def input_check(args):
    # Env checking
    if args.env == 'OvercookedMultiEnv-v0':
        if 'layout_name' not in args.env_config:
            raise EnvException(f"layout_name needed for {args.env}")
        elif args.env_config['layout_name'] not in LAYOUT_LIST:
            raise EnvException(
                f"{args.env_config['layout_name']} is not a valid layout")

    # Construct ego config
    if 'verbose' not in args.ego_config:
        args.ego_config['verbose'] = 1

    if args.ego_load is None:
        raise EnvException("Need to provide file for ego to load")

    if (args.alt_load is None) != (args.alt == 'DEFAULT'):
        raise EnvException("Load policy if and only if alt is not DEFAULT")

def generate_agent(env, policy_type, config, location):
    config['env'] = env
    if policy_type == 'DEFAULT':
        return gen_default(config, env)
    # return gen_fixed(config, policy_type, location)
    return gen_fixed(config, policy_type, location)

def generate_agent2(env, policy_type, config, location1, location2,args,trainflag):
    config['env'] = env
    if policy_type == 'DEFAULT':
        return gen_default(config, env)
    # return gen_fixed(config, policy_type, location)
    if trainflag==False:
        return gen_fixed2(config, policy_type, location1, location2,args.expert)
    else:
        return gen_fixed3(config, policy_type, location1, location2,args,env)

def run_test(ego, env, num_episodes, render=False,flag=False,pego=None):
    rewards = []
    for game in range(num_episodes):
        obs = env.reset()
        done = False
        reward = 0
        if render:
            env.render()
        while not done:
            paction = pego.get_action(obs, None,flag=False, moeflag=True)# do not need action of parnet prediction module
            action = ego.get_action(obs, par_act=paction, record=False, flag=True)
            obs, newreward, done, _ = env.step(action)
            reward += newreward
            if render:
                env.render()
                sleep(1/60)
        rewards.append(reward)
    env.close()
    print(f"Average Reward: {sum(rewards)/num_episodes}")
    print(f"Standard Deviation: {np.std(rewards)}")

def run_train_test(ego, env, num_episodes, render=False,flag=False,pego=None):
    # Before training your ego agent, you first need to add your partner agents
    # to the environment. You can create adaptive partner agents using
    # OnPolicyAgent (for PPO/A2C) or OffPolicyAgent (for DQN/SAC). If you set
    # verbose to true for these agents, you can also see their learning progress
    # partner = OnPolicyAgent(partner)
    # env.add_partner_agent(partner)
    # if flag==True:
    #     env.set_pego(pego)
    # Finally, you can construct an ego agent and train it in the environment
    ego.learn(total_timesteps=500,flag=True,pego=pego)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
            Test ego and partner in an environment.

            Environments:
            -------------
            All MultiAgentEnv environments are supported. Some have additional
            parameters that can be passed into --env-config. Specifically,
            OvercookedMultiEnv-v0 has a required layout_name parameter, so
            one must add:

                --env-config '{"layout_name":"[SELECTED_LAYOUT]"}'

            OvercookedMultiEnv-v0 also has parameters `ego_agent_idx` and
            `baselines`, but these have default initializations. LiarsDice-v0
            has an optional parameter, `probegostart`.

            The environment can be wrapped with a framestack, which transforms
            the observation to stack previous observations as a workaround
            for recurrent networks not being supported. It can also be wrapped
            with a recorder wrapper, which will write the transitions to the
            given file.

            Ego-Agent:
            ----------
            The ego-agent is considered the main agent in the environment.
            From the perspective of the ego agent, the environment functions
            like a regular gym environment.

            Supported ego-agent algorithms include PPO, ModularAlgorithm, ADAP,
            and ADAP_MULT. The default parameters of these algorithms can
            be overriden using --ego-config.

            Alt-Agent:
            -----------
            The alt-agents are the partner agents that are embedded in the
            environment. If multiple are listed, the environment randomly
            samples one of them to be the partner at the start of each episode.

            Supported alt-agent algorithms include PPO, ADAP, ADAP_MULT,
            and DEFAULT. DEFAULT refers to the default hand-made policy
            in the environment (if it exists).

            Default parameters for these algorithms can be overriden using
            --alt-config.

            NOTE:
            All configs are based on the json format, and will be interpreted
            as dictionaries for the kwargs of their initializers.

            Example usage (Overcooked with ADAP agents that share the latent
            space):

            python3 tester.py OvercookedMultiEnv-v0 ADAP ADAP --env-config
            '{"layout_name":"random0"}' -l
            ''')

    parser.add_argument('env',
                        choices=ENV_LIST,
                        help='The environment to train in')

    parser.add_argument('ego',
                        choices=EGO_LIST,
                        help='Algorithm for the ego agent')

    parser.add_argument('alt',
                        choices=PARTNER_LIST,
                        help='Algorithm for the partner agent')

    parser.add_argument('--total-episodes', '-t',
                        type=int,
                        default=100,
                        help='Number of episodes to run')

    # parser.add_argument('--device', '-d',
    #                     default='auto',
    #                     help='Device to run pytorch on')
    parser.add_argument('--seed', '-s',
                        type=int,
                        help='Seed for randomness')

    parser.add_argument('--ego-config',
                        type=json.loads,
                        default={},
                        help='Config for the ego agent')

    parser.add_argument('--alt-config',
                        type=json.loads,
                        default={},
                        help='Config for the partner agent')

    parser.add_argument('--env-config',
                        type=json.loads,
                        default={},
                        help='Config for the environment')

    # Wrappers
    parser.add_argument('--framestack', '-f',
                        type=int,
                        default=1,
                        help='Number of observations to stack')
    parser.add_argument('--expert', 
                        type=int,
                        default=20,
                        help='Number of experts')
    parser.add_argument('--record', '-r',
                        help='Saves joint trajectory into file specified')

    parser.add_argument('--render',
                        action='store_true',
                        help='Render the environment as it is being run')

    parser.add_argument('--ego-load',
                        default="models/OvercookedMultiEnv-v0-simple-MOE-ego-95-20-100",
                        help='File to load the ego agent from')
    
    parser.add_argument('--pego-load',
                        default="models/OvercookedMultiEnv-v0-simple-MOE-alt-95-20-100",
                        help='File to load the pego agent from')

    parser.add_argument('--alt-load',
                        default="models/OvercookedMultiEnv-v0-simple-MOE-alt-95-20-100",
                        help='File to load the partner agent from')

    args = parser.parse_args()

    input_check(args)
    flag = True # test the MOE and PPO together as a myselves
    print(f"Arguments: {args}")
    env, altenv = generate_env(args)
    print(f"Environment: {env}; Partner env: {altenv}")
    # ego = generate_agent(env, args.ego, args.ego_config, args.ego_load)
    # ego = generate_agent(env, args.ego, args.ego_config, args.ego_load)
    trainflag=False
    if(flag): # this flag means MOE-PPO as a whole for test,pego is MOE
        ego,pego = generate_agent2(env, args.ego, args.ego_config, args.ego_load,args.pego_load,args,trainflag=trainflag)
    else:     
        ego = generate_agent(env, args.ego, args.ego_config, args.ego_load)
    print(f'Ego: {ego}')
    args.alt_config['expert'] = args.expert
    alt = generate_agent(altenv, args.alt, args.alt_config, args.alt_load)
    env.add_partner_agent(alt)
    if(flag):
        env.add_ego_agent(ego,True,pego)
    else:
        env.add_ego_agent(ego)

    print(f'Alt: {alt}')

    if(trainflag):
        run_train_test(ego, env, args.total_episodes, args.render,flag,pego=pego)     
        ego = StaticPolicyAgent(ego.model.policy)
        run_test(ego, env, args.total_episodes, args.render, flag, pego)

    if(trainflag==False):
        if(flag and trainflag==False):
            run_test(ego, env, args.total_episodes, args.render, flag, pego)
        else:
            run_test(ego, env, args.total_episodes, args.render)

    if args.record is not None:
        env.get_transitions().write_transition(args.record)
