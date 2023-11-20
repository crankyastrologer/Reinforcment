import gym
import numpy as np
import operator
from IPython.display import clear_output
from time import sleep
import random
import itertools
import tqdm
tqdm.monitor_interval = 0
#%%
def create_random_policy(env):
    policy = {}
    for key in range(0,env.observation_space.n):
        current_end = 0
        p = {}
        for action in range(0,env.action_space.n):
            p[action] = 1/env.action_space.n
            policy[key] = p
    return policy

def create_state_action_dictionary(env,policy):
    Q = {}
    for key in policy.keys():
        Q[key] = {a: 0.0 for a in range(0,env.action_space.n)}
    return Q

def run_game(env,policy,display = True):
    env.reset()
    episode = []
    finished = False

    while not finished:
        s = env.env.s
        if display:
            clear_output(True)
            env.render()
            sleep(1)
        timestep = [s]
        n = random.uniform(0,sum(policy[s].values()))
        top_range = 0
        for prob in policy[s].items():
            top_range +=prob[1]
            if n<top_range:
                action =  prob[0]

                break

        state, reward, finished, info, abc = env.step(action)
        timestep.append(action)
        timestep.append(reward)
        episode.append(timestep)
        if display:
          clear_output(True)
          env.render()
          sleep(1)
    return episode
def test_policy(policy,env):
    wins = 0
    r = 2
    for i in range(r):
        print(f'doing {i}')
        w = run_game(env,policy,display=False)[-1][-1]
        if w == 1:
            wins+=1
    return wins / r
#%%
def monte_carlo_e_soft(env,episodes=100,policy=None,epsilon=0.01):
    if not policy:
        policy = create_random_policy(env)
    Q = create_state_action_dictionary(env,policy)
    returns = {}
    for _ in range(episodes):
        print(_)
        G = 0
        episode = run_game(env = env,policy=policy, display=False)
        for i in reversed(range(0,len(episode))):
            s_t,a_t,r_t = episode[i]
            state_action = (s_t, a_t)
            G += r_t
            if not state_action in [(x[0], x[1]) for x in episode[0:i]]:
                if returns.get(state_action):
                    returns[state_action].append(G)
                else:
                    returns[state_action]=[G]

                Q[s_t][a_t] = sum(returns[state_action])/len(returns[state_action])

                Q_list = list(map(lambda x: x[1], Q[s_t].items()))
                indices = [i for i, x in enumerate(Q_list) if x== max(Q_list)]
                max_Q = random.choice(indices)
                A_star = max_Q
                for a in policy[s_t].items():
                    if a[0]==A_star:
                        policy[s_t][a[0]] = 1 - epsilon + (epsilon / abs(sum(policy[s_t].values())))
                    else:
                        policy[s_t][a[0]] = (epsilon / abs(sum(policy[s_t].values())))

    return policy
#%%
env = gym.make('FrozenLake8x8-v1', render_mode = 'rgb_array')
policy = monte_carlo_e_soft(env)
test_policy(policy,env)