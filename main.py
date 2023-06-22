import os
import sys
import time

import cv2
import torch
import psutil
import numpy as np

from argparse import ArgumentParser
from chainer import serializers
from torch.backends import cudnn
from dqn import DQN
from pc import Logger
from hkenv import Keys, Operation, HKEnv
from collections import namedtuple, deque

def limit_cpu():
    p = psutil.Process()
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

def interact(env, dqn, observe_interval, i_episode, train=True, indent=0):
    dqn.warmup()
    observe_list, action_list, info_list = env.for_warmup(n_frames, observe_interval)
    state, condition, status = dqn.convert_to_init((observe_list, action_list, info_list))
    action_idx = dqn.choose_action_idx(state, condition, status, evaluate=True)
    # warmup completed

    experiences = []
    t = 0
    rewards = 0
    observe_list, action_list, info_list = env.reset(n_frames, observe_interval)
    state, condition, status = dqn.convert_to_init((observe_list, action_list, info_list))
    episode_start_time = time.time()
    while True:
        action_idx = dqn.choose_action_idx(state, condition, status)
        action = Operation.get_by_idx(action_idx)
        action_idx, action = Operation.replace_conflict(action_idx, action)
        observe, info, reward, done, enemy_remain = env.step(action)

        next_state, next_condition, next_status = dqn.convert_to_next((state, condition, status),
                                                                      (observe, action, info))

        if train:
            experiences.append(dqn.convert_to_experience(state, condition, status,
                                                         action_idx, action, reward,
                                                         next_state, next_condition, next_status))

        rewards += reward.item()

        if done:
            env.close()
            Logger.indent(indent=indent)
            print(f"Episode {i_episode} finished after {t + 1} timesteps", end='')
            print(f", total rewards {rewards:.4f}, cur_enemy_remain = {enemy_remain}", end='')
            print(f", ops = {(t + 1) / (time.time() - episode_start_time):.4f}")
            break

        state, condition, status = next_state, next_condition, next_status
        t += 1
        time.sleep(0.02 + observe_interval - 0.07)

        del action_idx, action
        del observe, info, reward, done, enemy_remain
        del next_state, next_condition, next_status

    del observe_list, action_list, info_list
    del state, condition, status
    del episode_start_time
    del t, rewards

    return experiences

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
    h = 384
    w = 216
    info_size = 1 # got hit
    memory_capacity = 2000
    target_replace_iter = memory_capacity // 2
    net_dir = "nets"
    memory_dir = "memories"
    lr = 0.1
    epsilon = 0.15
    gamma = 0.9
    n_episodes = 3000
    n_episodes_save = 20
    batch_learn = memory_capacity // 4
    observe_interval = 0.09 # min is 0.07

    dqn = DQN(state_size=(n_frames, w, h),
              condition_size=(n_frames, len(Keys)),
              status_size=(n_frames, info_size),
              in_channels=n_frames,
              out_classes=len(Operation.POSSIBLE_ACTION),
              lr=lr,
              total_steps=n_episodes*batch_learn,
              epsilon=epsilon,
              gamma=gamma,
              memory_capacity=memory_capacity,
              target_replace_iter=target_replace_iter,
              net_dir=net_dir,
              memory_dir=memory_dir,
              device=device)

    env = HKEnv(observe_size=(h, w),
                info_size=info_size,
                device=device)

    # dqn.save("dqn_training")
    # dqn.load("dqn_training")
    # env.test()
    # sys.exit()

    if args.evaluate:
        for i_evaluate in range(3):
            _ = interact(env, dqn, observe_interval, i_evaluate, train=False, indent=4)
    else:
        start_learning = False
        for i_train in range(n_episodes):
            torch.cuda.empty_cache()
            # print(f"{len(dqn.memory) = }")

            experiences = interact(env, dqn, observe_interval, i_train, train=True)
            dqn.store_experiences(experiences)

            if dqn.can_learn:
                if not start_learning:
                    start_learning = True
                    print(f"start learning")
                dqn.learn(batch_learn)

            if (i_train % n_episodes_save) == n_episodes_save - 1:
                dqn.save(f"dqn_training")

                if dqn.can_learn:
                    for i_evaluate in range(3):
                        _ = interact(env, dqn, observe_interval, i_evaluate, train=False, indent=4)

        print(f"training completed")
        dqn.save(f"dqn_completed")

    env.close()
