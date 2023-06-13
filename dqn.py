import time

import torch
import torch.nn as nn
import numpy as np

from network import ResNet18
from hkenv import Hotkey
from utils import Memory

class DQN(nn.Module):
    def __init__(self, state_size, n_actions, condition_size, batch_size, lr, epsilon,
                 gamma, target_replace_iter, memory_capacity, device):
        super().__init__()
        self.condition_size = condition_size
        self.state_size = state_size
        self.n_condition = torch.prod(torch.Tensor(condition_size).to(torch.int64))
        self.n_states = torch.prod(torch.Tensor(state_size).to(torch.int64))

        self.eval_net = ResNet18(n_actions, self.n_condition).to(device)
        self.target_net = ResNet18(n_actions, self.n_condition).to(device)

        self.memory = Memory(memory_capacity)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0

        self.n_actions = n_actions
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity
        self.device = device

        self.warmup()

    def warmup(self):
        for _ in range(3):
            fake_input = torch.zeros(self.state_size).unsqueeze(0).to(self.device)
            fake_condition = torch.zeros(self.n_condition).to(self.device)

            q_eval = self.eval_net(fake_input, fake_condition)
            q_next = self.target_net(fake_input, fake_condition).detach()
            loss = self.loss_func(q_eval, q_eval)
            loss.backward()
            # self.optimizer.step()

            self.optimizer.zero_grad()
        for _ in range(self.memory_capacity * 2):
            self.memory.store(torch.zeros(self.n_states * 2 + self.n_condition + 2))

    def choose_action(self, state, condition, use_epsilon=True):
        # epsilon-greedy
        if torch.rand(1) < self.epsilon:
            action = torch.randint(self.n_actions, (1,))[0]
        else:
            actions_value = self.eval_net(state.unsqueeze(0), condition)
            action = torch.argmax(actions_value)
        action = torch.Tensor(action).to(torch.int64).to(self.device)
        cond = Hotkey.idx_to_hotkey(action).to(self.device)
        return action, cond


    def store_transition(self, state, condition, action, reward, next_state):
        trans = torch.hstack((state.flatten(), condition, action, reward, next_state.flatten()))
        self.memory.store(trans)

    def random_sample_transition(self):
        idx = torch.randint(self.memory_capacity, (1,))[0]
        trans = self.memory.get(idx)

        state = trans[:self.n_states].reshape(self.state_size).to(self.device)
        condition = trans[self.n_states:self.n_states + self.n_condition].to(torch.int64).to(self.device)
        action = trans[self.n_states + self.n_condition].to(torch.int64).to(self.device)
        reward = trans[self.n_states + self.n_condition + 1].to(self.device)
        next_state = trans[-self.n_states:].reshape(self.state_size).to(self.device)

        return state, condition, action, reward, next_state

    @property
    def can_learn(self):
        return self.memory.is_full

    def learn(self):
        state, condition, action, reward, next_state = self.random_sample_transition()

        next_condition = torch.cat((condition, Hotkey.idx_to_hotkey(action).to(self.device)))
        next_condition = next_condition[-self.n_condition:]

        q_eval = self.eval_net(state.unsqueeze(0), condition)[action]
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
