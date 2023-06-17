import time

import torch
import torch.nn as nn
import numpy as np
from collections import deque

from pc import FileAdmin
from hkenv import HKEnv
from utils import Memory, Logger
from network import resnet18

class DQN(nn.Module):
    def __init__(self, state_size, n_frames, n_actions, condition_size, batch_size, lr, epsilon,
                 net_dir, memory_dir, gamma, target_replace_iter, memory_capacity, device):
        super().__init__()
        self.condition_size = condition_size
        self.state_size = state_size
        self.n_condition = torch.prod(torch.Tensor(condition_size).to(torch.int64))
        self.n_states = torch.prod(torch.Tensor(state_size).to(torch.int64))

        self.net_dir = net_dir
        self.eval_net = resnet18(n_frames, n_actions, self.n_condition).to(device)
        self.target_net = resnet18(n_frames, n_actions, self.n_condition).to(device)

        self.memory_capacity = memory_capacity
        self.memory_dir = memory_dir
        self.memory = Memory(memory_dir, memory_capacity)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0

        self.n_actions = n_actions
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.device = device

        # self.warmup()

    def warmup(self):
        # print("warm up", end='\r')
        fake_state = torch.rand(self.state_size).to(self.device)
        fake_condition = torch.rand(self.condition_size).to(self.device)

        act = self.choose_action(fake_state, fake_condition, True)

        q_eval = self.eval_net(fake_state.unsqueeze(0), fake_condition)
        q_next = self.target_net(fake_state.unsqueeze(0), fake_condition).detach()
        loss = self.loss_func(q_eval, q_eval)
        loss.backward()
        self.optimizer.step()

        self.optimizer.zero_grad()

        # print("warm up completed")
        del fake_state, fake_condition
        del q_eval, q_next, loss

    def choose_action(self, state, condition, evaluate=False):
        # epsilon-greedy
        if torch.rand(1) < self.epsilon and not evaluate:
            action = torch.randint(self.n_actions, (1,))[0]
        else:
            actions_value = self.eval_net(state.unsqueeze(0), condition)
            action = torch.argmax(actions_value)
        action = torch.Tensor(action).to(torch.int64).to(self.device)
        cond = HKEnv.idx_to_hotkey(action).to(self.device)
        return action, cond

    def store_history(self, history):
        # flatten_history = deque()
        flatten_history = []
        for state, condition, action, reward, next_state in history:
            trans = torch.hstack((state.flatten(), condition.flatten(), action, reward, next_state.flatten()))
            flatten_history.append(trans.cpu())
        self.memory.extend(flatten_history)

        del state, condition, action, reward, next_state, trans
        del flatten_history


    # def store_transition(self, state, condition, action, reward, next_state):
    #     # print(f"{state.shape = }")
    #     # print(f"{condition.shape = }")
    #     # print(f"{action = }")
    #     # print(f"{reward = }")
    #     # print(f"{next_state.shape = }")
    #     trans = torch.hstack((state.flatten(), condition.flatten(), action, reward, next_state.flatten()))
    #     self.memory.append(trans.cpu())
    #     del trans

    def random_sample_transitions(self, times=1):
        # idx = torch.randint(self.memory_capacity, (1,))[0]
        batch_idx = torch.randint(len(self.memory), (times,))

        transitions = deque()
        memory_datas = self.memory.batch_getitem(batch_idx)
        for trans in memory_datas:
            state = trans[:self.n_states].reshape(self.state_size)
            condition = trans[self.n_states:self.n_states + self.n_condition].to(torch.int64)
            condition = condition.reshape(self.condition_size)
            action = trans[self.n_states + self.n_condition].to(torch.int64)
            reward = trans[self.n_states + self.n_condition + 1]
            next_state = trans[-self.n_states:].reshape(self.state_size)
            transitions.append((state, condition, action, reward, next_state))

            del trans
        # return state, condition, action, reward, next_state
        return transitions

    @property
    def can_learn(self):
        # return len(self.memory) >= self.memory_capacity / 5
        return len(self.memory) == self.memory_capacity

    def learn(self, times=1):
        print(f"sampling", end='\r')
        cur = time.time()
        transitions = self.random_sample_transitions(times=times)
        print(f"sampling completed, time = {time.time() - cur}")

        print(f"learning", end='\r')
        cur = time.time()
        for idx, (state, condition, action, reward, next_state) in enumerate(transitions):
            print(f"learning {idx + 1} / {len(transitions)}", end='\r')
            state = state.to(self.device)
            condition = condition.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)
            next_state = next_state.to(self.device)

            next_cond = HKEnv.idx_to_hotkey(action).to(self.device)
            next_condition = torch.cat((condition, next_cond.unsqueeze(0)))[1:]

            q_eval = self.eval_net(state.unsqueeze(0), condition)[0, action]
            q_next = self.target_net(next_state.unsqueeze(0), next_condition).detach()
            q_target = reward + self.gamma * q_next.max()
            loss = self.loss_func(q_eval, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.learn_step_counter += 1
            if self.learn_step_counter % self.target_replace_iter == 0:
                print(f"copy target from eval")
                self.target_net.load_state_dict(self.eval_net.state_dict())

            del state, condition, action, reward, next_state
            del next_cond, next_condition
            del q_eval, q_next, q_target, loss
        Logger.clear_line()
        print(f"learning completed, time = {time.time() - cur}")

    def save(self, name):
        print(f"saving net", end='\r')
        FileAdmin.safe_save_net(self.eval_net, f"{self.net_dir}/{name}.pt")
        print(f"saving net completed")

        print(f"saving {name} memory")
        self.memory.save(new_prefix=name)
        print(f"saving {name} memory completed")

    def load(self, name):
        print(f"loading net", end='\r')
        self.eval_net = FileAdmin.safe_load_net(self.eval_net, f"{self.net_dir}/{name}.pt")
        self.target_net = FileAdmin.safe_load_net(self.target_net, f"{self.net_dir}/{name}.pt")
        print(f"loading net completed")

        print(f"loading {name} memory")
        self.memory.load(new_prefix=name)
        print(f"loading {name} memory completed")

        self.warmup()
