import os
import sys
import time

import cv2
import torch
import psutil
import hkenv
import numpy as np

from chainer import serializers
from torch.backends import cudnn
from dqn import DQN
from pc import Logger
from collections import namedtuple, deque

def limit_cpu():
    p = psutil.Process()
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

if __name__ == "__main__":
    # assign_job(create_job())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{device = }")
    # device = 'cpu'
    cudnn.benchmark = True

    batch_size = 1
    lr = 0.01
    epsilon = 0.1
    gamma = 0.9
    target_replace_iter = 10000
    # target_replace_iter = 50
    memory_capacity = 100000
    # memory_capacity = 20
    n_episodes = 1000
    h = 210
    w = 120
    n_actions = hkenv.Hotkey.ALL_POSSIBLE_NUM
    n_frames = 12
    condition_size = (n_frames, hkenv.Hotkey.KEYS_NUM)
    state_size = (n_frames, w, h)
    save_dir = "saves"

    obs_interval = 0.09 # min is 0.07

    n_episodes_save = 20
    # n_episodes_save = 1

    dqn = DQN(state_size, n_frames, n_actions, condition_size, batch_size, lr, epsilon,
              gamma, target_replace_iter, memory_capacity, device)

    usage_memory = psutil.Process(os.getpid()).memory_info().rss
    print(f"usage_memory = {usage_memory / 1024 ** 2} MiB")
    print(f"{usage_memory = }")
    print(f"{psutil.Process().nice() = }")

    dqn.load(f"{save_dir}/dqn_training")
    # print(f"{len(dqn.memory) = }")
    # dqn.save(f"{save_dir}/dqn_training")
    # print(f"{dqn.eval_net.device = }")
    # print(f"{dqn.target_net.device = }")

    usage_memory = psutil.Process(os.getpid()).memory_info().rss
    print(f"\nafter load")
    print(f"usage_memory = {usage_memory / 1024 ** 2} MiB")
    print(f"{usage_memory = }")
    print(f"{psutil.Process().nice() = }")

    env = hkenv.HKEnv((h, w), device=device)
    # env.test()
    # sys.exit()

    start_learning = False

    min_enemy_remain = 1e9
    for i_episode in range(n_episodes):
        torch.cuda.empty_cache()

        print("warm up episode", end='\r')
        dqn.warmup()
        state, condition = env.for_warmup(n_frames, obs_interval)
        action, new_cond = dqn.choose_action(state, condition, evaluate=True)
        print("warm up episode completed")

        if not start_learning:
            print(f"{len(dqn.memory) = }")

        t = 0
        rewards = 0
        episode_start_time = time.time()
        state, condition = env.reset(n_frames, obs_interval)
        # sys.exit()
        while True:
            action, next_cond = dqn.choose_action(state, condition)
            next_obs, reward, done, info = env.step(action)

            next_state = torch.cat((state, next_obs.unsqueeze(0)))[1:]
            next_condition = torch.cat((condition, next_cond.unsqueeze(0)))[1:]

            dqn.store_transition(state, condition, action, reward, next_state)

            rewards += reward.item()

            min_enemy_remain = min(min_enemy_remain, info)

            if dqn.can_learn:
                if not start_learning:
                    start_learning = True
                    print(f"start_learning")
                dqn.learn() # 0.01
                time.sleep(obs_interval - 0.07)
            else:
                time.sleep(0.01 + obs_interval - 0.07)

            state = next_state
            condition = next_condition

            if done:
                print(f"Episode {i_episode} finished after {t + 1} timesteps, total rewards {rewards:.4f}, {min_enemy_remain = }, ops = {(t + 1) / (time.time() - episode_start_time):.4f}")

                if (i_episode % n_episodes_save) == n_episodes_save - 1:
                    dqn.save(f"{save_dir}/dqn_training")

                    if dqn.can_learn:
                        print("evaluating", end='\r')
                        for i_eval in range(3):
                            rewards = 0
                            episode_start_time = time.time()
                            state, condition = env.reset(n_frames, obs_interval)
                            while True:
                                # env.render()
                                action, next_cond = dqn.choose_action(state, condition, evaluate=True)
                                next_obs, reward, done, info = env.step(action)

                                next_state = torch.cat((state, next_obs.unsqueeze(0)))[1:]
                                next_condition = torch.cat((condition, next_cond.unsqueeze(0)))[1:]

                                rewards += reward.item()

                                time.sleep(0.02 + obs_interval - 0.07)

                                state = next_state
                                condition = next_condition

                                if done:
                                    Logger.indent(indent=4)
                                    print(f"evaluate {i_eval} finished, total rewards {rewards:.4f}, ops = {(t + 1) / (time.time() - episode_start_time):.4f}")
                                    break
                break # if done:


            del action, next_cond
            del next_obs, reward, done, info
            del next_state, next_condition

            t += 1
            # if t == 100:
            #     break

        del rewards
        del state, condition
        del t

        # break

    print(f"training completed")
    dqn.save(f"{save_dir}/dqn")

    env.close()
