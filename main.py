import sys
import time

import cv2
import torch
import psutil
import numpy as np

from collections import deque
from argparse import ArgumentParser
from torch.backends import cudnn
from drqn import DRQN
from pc import Logger
from hkenv import Keys, Action, HKEnv
from collections import deque
from colorama import init, Fore, Back, Style

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
    while True:
        print(f"step {t + 1}", end='\r')
        action, epsilon = drqn.choose_action(state, condition, train=train)
        Logger.indent(indent)
        if epsilon:
            print(f"random {Fore.CYAN}{action}{Fore.RESET}")
        else:
            print(f"choose {action}")
        if train:
            action.mutate(env.key_hold, env.observe_interval)
            if not epsilon and action.mutated:
                Logger.indent(indent)
                print(f"-----> {Fore.YELLOW}{action}{Fore.RESET}")
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

def learn(env, drqn, n_episodes, n_episodes_save):
    start_learning = False
    for i_train in range(n_episodes):
        torch.cuda.empty_cache()

        episode_experiences = interact(env, drqn, i_train, train=True)
        drqn.store_episode_experiences(episode_experiences)

        if not drqn.can_learn:
            print(f"not enough memory, {len(drqn.memory)} / {memory_capacity}")
        elif not start_learning:
            start_learning = True
            print(f"enough memory, start learning")

        if drqn.can_learn:
            drqn.learn(times=episode_learn_times)

        if (i_train % n_episodes_save) == n_episodes_save - 1:
            drqn.save(f"drqn_training")

            if drqn.can_learn:
                evaluate(env, drqn, n_eval=1)

    print(f"training completed")
    drqn.save(f"drqn_completed")

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
    lr = 1
    epsilon = 0.2
    gamma = 0.9
    n_episodes = 2000
    n_episodes_save = 20
    episode_learn_times = memory_capacity // 4
    observe_interval = 0.1 

    drqn = DRQN(state_size=(3, h, w),
                condition_size=len(Keys),
                n_frames=n_frames,
                out_classes=len(Action.ALL_POSSIBLE),
                lr=lr,
                total_steps=n_episodes*episode_learn_times,
                epsilon=epsilon,
                gamma=gamma,
                memory_capacity=memory_capacity,
                target_replace_iter=target_replace_iter,
                net_dir=net_dir,
                memory_dir=memory_dir,
                device=device)

    env = HKEnv(observe_size=(w, h),
                observe_interval=observe_interval,
                device=device)

    # drqn.save("drqn_training")
    drqn.load("drqn_training")
    # env.test()

    if args.evaluate:
        evaluate(env, drqn, n_eval=2)
    else:
        learn(env, drqn, n_episodes, n_episodes_save)

    env.close()
