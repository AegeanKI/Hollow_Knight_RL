import sys
import time

import cv2
import torch
import psutil
import numpy as np

from collections import deque
from argparse import ArgumentParser
from torch.backends import cudnn
from colorama import init, Fore, Back, Style
from drqn import DRQN
from pc import Logger
from hkenv import Keys, Action, HKEnv
from utils import Memory

def limit_cpu():
    p = psutil.Process()
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

def interact(env, drqn, i_episode, train=True, indent=0):
    drqn.warmup()
    state, condition = env.warmup()
    # action = drqn.choose_action(state, condition, train=False)
    # warmup completed

    episode_experiences = [deque() for _ in range(8)]
    t = 0
    rewards = 0
    drqn.reset_net()
    state, condition = env.reset()
    episode_start_time = time.time()
    choose_actions = torch.zeros(256)
    while True:
        print(f"step {t + 1}", end='\r')
        action, epsilon = drqn.choose_action(state, condition, train=train)
        if not epsilon:
            choose_actions[action.idx] = choose_actions[action.idx] + 1
        if train:
            action.mutate(env.key_hold, env.observe_interval)
        next_state, next_condition, reward, done, info = env.step(action)
        enemy_remain, affect, win = info
        assert not torch.isnan(done), f"done is nan, {done = }"
        assert not torch.isinf(done), f"done is nan, {done = }"
        assert done == 0 or done == 1, f"done is not 0 and not 1, {done = }"

        if train:
            experience = (state, condition,
                          action.idx, reward, done, affect,
                          next_state, next_condition)
            for dq, val in zip(episode_experiences, experience):
                dq.append(val.unsqueeze(0).cpu() if val.ndim == 0 else val.cpu())

        t += 1
        rewards += reward.item()
        state, condition = next_state, next_condition

        if done:
            print(f"most choose {Action(choose_actions.argmax())} {int(choose_actions.max())} / {int(choose_actions.sum())} times")
            Logger.clear_line()
            env.end(indent=indent)
            Logger.indent(indent=indent)
            if win:
                print(f"Episode {Fore.GREEN}{i_episode} win{Fore.RESET}", end='')
            else:
                print(f"Episode {Fore.RED}{i_episode} lose{Fore.RESET}", end='')
            print(f" after {t} timesteps", end='')
            print(f", total rewards {Fore.CYAN}{rewards:.3f}{Fore.RESET}", end='')
            print(f", cur_enemy_remain = {Fore.CYAN}{enemy_remain}{Fore.RESET}", end='')
            print(f", spend time = {time.time() - episode_start_time:.3f}", end='')
            print(f", ops = {t / (time.time() - episode_start_time):.3f}\n")
            break

        # if t == 20:
        #     break

        del action
        del reward, done, affect, enemy_remain
        del next_state, next_condition

    del state, condition
    del episode_start_time
    del t, rewards

    return [torch.stack(list(deques)) for deques in episode_experiences] if train else None

def evaluate(env, drqn, n_eval=1):
    Logger.indent(4)
    print("evaluating:")
    for i_evaluate in range(n_eval):
        interact(env, drqn, i_evaluate, train=False, indent=4)

def learn(env, drqn, memory, n_frames, n_episodes, n_episodes_save):
    start_learning = False
    for i_train in range(n_episodes):
        torch.cuda.empty_cache()

        episode_experiences = interact(env, drqn, i_train, train=True)
        memory.extend(episode_experiences)

        if not memory.full:
            print(f"not enough memory, {len(drqn.memory)} / {memory_capacity}")
        elif not start_learning:
            start_learning = True
            print(f"enough memory, start learning")

        if memory.full:
            samples = memory.prioritize_sample(n_frames, episode_learn_times)
            drqn.learn(samples)

        if (i_train % n_episodes_save) == n_episodes_save - 1:
            drqn.save(f"drqn_training")
            memory.save(f"drqn_training")

            if memory.full:
                evaluate(env, drqn, n_eval=1)

    print(f"training completed")
    drqn.save(f"drqn_completed")
    memory.save(f"drqn_completed")

def read_args():
    parser = ArgumentParser()

    parser.add_argument('--evaluate', action='store_true')
    # parser.add_argument('--load')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    limit_cpu()
    init(autoreset=True)

    args = read_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{device = }")
    cudnn.enable = True
    cudnn.benchmark = True

    n_frames = 50
    w = 384
    h = 216
    memory_capacity = 3000
    target_replace_iter = memory_capacity // 2
    net_dir = "nets"
    memory_dir = "memories"
    lr = 0.1
    epsilon = 0.2
    gamma = 0.9
    n_episodes = 2000
    n_episodes_save = 20
    episode_learn_times = memory_capacity // 4
    observe_interval = 0.1 

    drqn = DRQN(state_size=(3, h, w),
                condition_size=len(Keys),
                out_classes=len(Action.ALL_POSSIBLE),
                lr=lr,
                total_steps=n_episodes*episode_learn_times,
                epsilon=epsilon,
                gamma=gamma,
                target_replace_iter=target_replace_iter,
                net_dir=net_dir,
                device=device)

                   # state, condition, action.idx, reward, done, affect, next_state, next_condition
    memory_sizes = ((3, h, w), (len(Keys), ), (1, ), (1, ), (1, ), (1, ), (3, h, w), (len(Keys), ))
    memory = Memory(memory_capacity, memory_sizes, memory_dir)

    env = HKEnv(observe_size=(w, h),
                observe_interval=observe_interval,
                device=device)

    drqn.load("drqn_training")
    memory.load("drqn_training")
    # env.test()

    if args.evaluate:
        evaluate(env, drqn, n_eval=2)
    else:
        learn(env, drqn, memory, n_frames, n_episodes, n_episodes_save)

    env.close()
