import sys

import hkenv
import time
import cv2
import torch
import numpy as np

from torch.backends import cudnn
from chainer import serializers
from dqn import DQN
from collections import namedtuple, deque

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    cudnn.benchmark = True

    batch_size = 1
    # lr = 0.01
    lr = 0.0001
    epsilon = 0.1
    gamma = 0.9
    target_replace_iter = 10000
    # target_replace_iter = 5
    memory_capacity = 100000
    # memory_capacity = 2
    n_episodes = 500
    h = 160
    w = 160
    n_actions = hkenv.Hotkey.ALL_POSSIBLE_NUM
    n_frames = 3 # must be 3 now (resnet need 3 channels)
    condition_size = (n_frames, hkenv.Hotkey.KEYS_NUM)
    state_size = (n_frames, h, w)

    obs_interval = 0.09 # min is 0.07

    n_episodes_save = 10

    dqn = DQN(state_size, n_actions, condition_size, batch_size, lr, epsilon,
              gamma, target_replace_iter, memory_capacity, device)

    dqn.load_state_dict(torch.load("dqn_training.pt"))

    env = hkenv.HKEnv((h, w), device=device)

    # env.test()
    # sys.exit()
    start_learning = False

    for i_episode in range(n_episodes):
        t = 0
        rewards = 0
        obs0, obs1, obs2, cond0, cond1, cond2 = env.reset(obs_interval)
        state = torch.stack((obs0, obs1, obs2))
        condition = torch.cat((cond0, cond1, cond2))
        # sys.exit()
        while True:
            # env.render()
            action, cond3 = dqn.choose_action(state, condition)
            obs3, reward, done, info = env.step(action)
            next_state = torch.stack((obs1, obs2, obs3))
            next_condition = torch.cat((cond1, cond2, cond3))

            dqn.store_transition(state, condition, action, reward, next_state)

            rewards += reward.item()

            # if dqn.memory_counter > memory_capacity:
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
            obs1, obs2 = obs2, obs3
            cond1, cond2 = cond2, cond3

            if done:
                print(f"Episode {i_episode} finished after {t + 1} timesteps, total rewards {rewards}")

                if (i_episode % n_episodes_save) == n_episodes_save - 1:
                    torch.save(dqn.state_dict(), "dqn_training.pt")

                if dqn.can_learn:
                    print("evaluating")
                    rewards = 0
                    obs0, obs1, obs2, cond0, cond1, cond2 = env.reset(obs_interval)
                    state = torch.stack((obs0, obs1, obs2))
                    condition = torch.cat((cond0, cond1, cond2))
                    while True:
                        # env.render()
                        action, cond3 = dqn.choose_action(state, condition)
                        obs3, reward, done, info = env.step(action)
                        next_state = torch.stack((obs1, obs2, obs3))
                        next_condition = torch.cat((cond1, cond2, cond3))

                        rewards += reward.item()

                        time.sleep(0.02 + obs_interval - 0.07)

                        state = next_state
                        condition = next_condition
                        obs1, obs2 = obs2, obs3
                        cond1, cond2 = cond2, cond3

                        if done:
                            print(f"evaluate finished after Episode {i_episode}, total rewards {rewards}")
                break

            t += 1

            if t == 10:
                break
        break

    torch.save(dqn.state_dict(), "dqn.pt")

    env.close()
