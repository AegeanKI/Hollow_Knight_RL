import os
import sys
import time

import cv2
import torch
import psutil
import numpy as np

from collections import deque
from argparse import ArgumentParser
from chainer import serializers
from torch.backends import cudnn
from dqn import DQN
from pc import Logger
from hkenv import Keys, Action, HKEnv
from collections import namedtuple, deque

def limit_cpu():
    p = psutil.Process()
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

def interact(env, dqn, i_episode, train=True, indent=0):
    dqn.warmup()
    state, condition = env.for_warmup()
    action = dqn.choose_action(state, condition, evaluate=True)
    # warmup completed

    (episode_state, episode_condition,
     episode_action_idx, episode_reward, episode_done,
     episode_next_state, episode_next_condition) = [deque() for _ in range(7)]
    t = 0
    rewards = 0
    state, condition = env.reset()
    episode_start_time = time.time()
    while True:
        print(f"step {t + 1}", end='\r')
        action = dqn.choose_action(state, condition)
        if train:
            action.mutate(env.key_hold, env.observe_interval)
        next_state, next_condition, reward, done, enemy_remain = env.step(action)

        if train:
            episode_state.append(state.cpu())
            episode_condition.append(condition.cpu())
            episode_action_idx.append(action.idx.unsqueeze(0).cpu())
            episode_reward.append(reward.unsqueeze(0).cpu())
            episode_done.append(done.unsqueeze(0).cpu())
            episode_next_state.append(next_state.cpu())
            episode_next_condition.append(next_condition.cpu())


        rewards += reward.item()

        if done:
            env.close()
            Logger.indent(indent=indent)
            print(f"Episode {i_episode} finished after {t + 1} timesteps", end='')
            print(f", total rewards {rewards:.3f}, cur_enemy_remain = {enemy_remain}", end='')
            print(f", spend time = {time.time() - episode_start_time:.3f}", end='')
            print(f", ops = {(t + 1) / (time.time() - episode_start_time):.3f}")
            break

        state, condition = next_state, next_condition
        t += 1

        del action
        del reward, done, enemy_remain
        del next_state, next_condition

    del state, condition
    del episode_start_time
    del t, rewards

    if not train:
        return None
    return (torch.stack(list(episode_state)), torch.stack(list(episode_condition)),
            torch.stack(list(episode_action_idx)), torch.stack(list(episode_reward)), torch.stack(list(episode_done)),
            torch.stack(list(episode_next_state)), torch.stack(list(episode_next_condition)))

def read_args():
    parser = ArgumentParser()

    parser.add_argument('--evaluate', action='store_true')
    # parser.add_argument('--load')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    limit_cpu()

    args = read_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{device = }")
    cudnn.benchmark = True

    n_frames = 12
    w = 384
    h = 216
    memory_capacity = 3000
    target_replace_iter = memory_capacity // 2
    net_dir = "nets"
    memory_dir = "memories"
    lr = 0.1
    epsilon = 0.2
    gamma = 0.99
    n_episodes = 2000
    n_episodes_save = 20
    episode_learn_times = memory_capacity // 4
    observe_interval = 0.1 

    dqn = DQN(state_size=(3, h, w),
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

    # dqn.save("dqn_training")
    # dqn.load("dqn_training")
    # env.test()

    if args.evaluate:
        for i_evaluate in range(3):
            interact(env, dqn, i_evaluate, train=False, indent=4)
    else:
        start_learning = False
        for i_train in range(n_episodes):
            torch.cuda.empty_cache()

            episode_experiences = interact(env, dqn, i_train, train=True)
            dqn.store_episode_experiences(episode_experiences)

            if dqn.can_learn and not start_learning:
                start_learning = True
                print(f"enough memory, start learning")
            elif not dqn.can_learn:
                print(f"not enough memory, {len(dqn.memory)} / {memory_capacity}")

            if dqn.can_learn:
                dqn.learn(times=episode_learn_times)

            if (i_train % n_episodes_save) == n_episodes_save - 1:
                dqn.save(f"dqn_training")

                if dqn.can_learn:
                    Logger.indent(4)
                    print("evaluating:")
                    for i_evaluate in range(3):
                        _ = interact(env, dqn, i_evaluate, train=False, indent=4)

        print(f"training completed")
        dqn.save(f"dqn_completed")

    env.close()
