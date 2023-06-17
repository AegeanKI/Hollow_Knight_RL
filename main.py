import os
import sys
import time

import cv2
import torch
import psutil
import numpy as np

from chainer import serializers
from torch.backends import cudnn
from dqn import DQN
from pc import Logger
from hkenv import HKEnv
from collections import namedtuple, deque

def limit_cpu():
    p = psutil.Process()
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{device = }")
    # device = 'cpu'
    cudnn.benchmark = True

    batch_size = 1
    lr = 0.1
    epsilon = 0.1
    gamma = 0.9
    # target_replace_iter = 10000
    target_replace_iter = 500
    # memory_capacity = 100000
    memory_capacity = 5000
    n_episodes_save = 20
    # n_episodes_save = 1
    n_episodes = 1000
    h = 210
    w = 120
    n_actions = HKEnv.Hotkey.ALL_POSSIBLE_NUM
    n_frames = 16
    condition_size = (n_frames, HKEnv.Hotkey.KEYS_NUM)
    state_size = (n_frames, w, h)
    net_dir = "nets"
    memory_dir = "memories"

    obs_interval = 0.09 # min is 0.07


    dqn = DQN(state_size, n_frames, n_actions, condition_size, batch_size, lr, epsilon,
              net_dir, memory_dir, gamma, target_replace_iter, memory_capacity, device)

    # print(f"default {len(dqn.memory) = }")
    # dqn.load("dqn_training")
    # print(f"dqn_training {len(dqn.memory) = }")
    # print(f"{len(dqn.memory) = }")
    # dqn.save("dqn_training")

    # usage_memory = psutil.Process(os.getpid()).memory_info().rss
    # print(f"\nafter load")
    # print(f"usage_memory = {usage_memory / 1024 ** 2} MiB")
    # print(f"{usage_memory = }")
    # print(f"{psutil.Process().nice() = }")

    env = HKEnv((h, w), device=device)
    # env.test()

    # sys.exit()

    min_enemy_remain = 1e9
    for i_episode in range(n_episodes):
        torch.cuda.empty_cache()

        print(f"{len(dqn.memory) = }")

        print("warm up episode", end='\r')
        dqn.warmup()
        state, condition = env.for_warmup(n_frames, obs_interval)
        action, new_cond = dqn.choose_action(state, condition, evaluate=True)
        print("warm up episode completed")

        history = []
        t = 0
        rewards = 0
        state, condition = env.reset(n_frames, obs_interval)
        episode_start_time = time.time()
        # sys.exit()
        # print(f"in loop")
        while True:
            action, next_cond = dqn.choose_action(state, condition)
            next_obs, reward, done, info = env.step(action)

            next_state = torch.cat((state, next_obs.unsqueeze(0)))[1:]
            next_condition = torch.cat((condition, next_cond.unsqueeze(0)))[1:]

            history.append([state, condition, action, reward, next_state])

            rewards += reward.item()
            min_enemy_remain = min(min_enemy_remain, info)

            if done:
                print(f"Episode {i_episode} finished after {t + 1} timesteps", end='')
                print(f", total rewards {rewards:.4f}, cur_enemy_remain = {info}, {min_enemy_remain = }", end='')
                print(f", ops = {(t + 1) / (time.time() - episode_start_time):.4f}")
                break

            state, condition = next_state, next_condition
            t += 1
            time.sleep(0.02 + obs_interval - 0.07)

            del action, next_cond
            del next_obs, reward, done, info
            del next_state, next_condition

            # if t == 100:
            #     break

        dqn.store_history(history)

        if dqn.can_learn:
            dqn.learn(times=min(2000, len(dqn.memory)))

        if (i_episode % n_episodes_save) == n_episodes_save - 1:
            dqn.save(f"dqn_training")

            for i_eval in range(3):
                print("evaluating", end='\r')
                t = 0
                rewards = 0
                state, condition = env.reset(n_frames, obs_interval)
                episode_start_time = time.time()
                while True:
                    action, next_cond = dqn.choose_action(state, condition, evaluate=True)
                    next_obs, reward, done, info = env.step(action)

                    next_state = torch.cat((state, next_obs.unsqueeze(0)))[1:]
                    next_condition = torch.cat((condition, next_cond.unsqueeze(0)))[1:]

                    rewards += reward.item()
                    min_enemy_remain = min(min_enemy_remain, info)

                    if done:
                        Logger.indent(indent=4)
                        print(f"Evaluate {i_eval} finished after {t + 1} timesteps", end='')
                        print(f", total rewards {rewards:.4f}", end='')
                        print(f", ops = {(t + 1) / (time.time() - episode_start_time):.4f}")
                        break

                    state, condition = next_state, next_condition
                    t += 1
                    time.sleep(0.02 + obs_interval - 0.07)

                    del action, next_cond
                    del next_obs, reward, done, info
                    del next_state, next_condition

        del rewards
        del state, condition
        del t
        del history

        # break

    print(f"training completed")
    dqn.save(f"dqn_completed")

    env.close()
