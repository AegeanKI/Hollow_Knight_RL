import sys
import hkenv
import time
import cv2
import torch
import numpy as np

from chainer import serializers
from dqn import DQN
from collections import namedtuple, deque

if __name__ == "__main__":

    # model = ResNet18(out_features=len(hkenv.Hotkey.KEYS))
    batch_size = 1
    lr = 0.01
    epsilon = 0.1
    gamma = 0.9
    target_replace_iter = 100
    memory_capacity = 10000
    n_episodes = 1000
    h = 160
    w = 160
    state_size = (3, h, w)
    n_actions = hkenv.Hotkey.NUM

    n_episodes_save = 10

    dqn = DQN(state_size, n_actions, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity)

    # dqn.load_state_dict(torch.load("dqn.pt"))

    env = hkenv.HKEnv((h, w))

    # env.test()
    # sys.exit()
    start_learning = False

    for i_episode in range(n_episodes):
        t = 0
        rewards = 0
        obs0, obs1, obs2 = env.reset()
        state = np.stack((obs0, obs1, obs2))
        while True:
            # env.render()
            time.sleep(0.1)
            action = dqn.choose_action(state)
            obs3, reward, done, info = env.step(action)
            next_state = np.stack((obs1, obs2, obs3))

            dqn.store_transition(state, action, reward, next_state)

            rewards += reward

            if dqn.memory_counter > memory_capacity:
                if not start_learning:
                    start_learning = True
                    print(f"start_learning")
                dqn.learn()

            state = next_state
            obs1, obs2 = obs2, obs3

            if done:
                print(f"Episode {i_episode} finished after {t + 1} timesteps, total rewards {rewards}")

                if (i_episode % n_episodes_save) == n_episodes_save - 1:
                    torch.save(dqn.state_dict(), "dqn_training.pt")
                break

            t += 1

    torch.save(dqn.state_dict(), "dqn.pt")

    env.close()
