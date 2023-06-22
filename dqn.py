import os
import time

import torch
import torch.nn as nn
import numpy as np

from pc import FileAdmin
# from hkenv import Operation
from utils import Memory
from network import resnet18

class DQN(nn.Module):

    def __init__(self, state_size, condition_size, status_size, in_channels,
                 out_classes, lr, total_steps, epsilon, gamma, memory_capacity, 
                 target_replace_iter, net_dir, memory_dir, device):
        super().__init__()

        self.state_size = state_size
        self.condition_size = condition_size
        self.status_size = status_size

        flatten_condition_size = torch.prod(torch.Tensor(condition_size).to(torch.int64))
        flatten_state_size = torch.prod(torch.Tensor(state_size).to(torch.int64))
        flatten_status_size = torch.prod(torch.Tensor(status_size).to(torch.int64))

        self.out_classes = out_classes
        self.eval_net = resnet18(in_channels=in_channels,
                                 out_classes=out_classes,
                                 in_condition_size=flatten_condition_size,
                                 in_status_size=flatten_status_size).to(device)
        self.target_net = resnet18(in_channels=in_channels,
                                   out_classes=out_classes,
                                   in_condition_size=flatten_condition_size,
                                   in_status_size=flatten_status_size).to(device)

        self.memory = Memory(memory_capacity)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,
                                                           base_lr=lr/ 10,
                                                           max_lr=lr,
                                                           gamma=0.8,
                                                           step_size_up=total_steps/(2*10),
                                                           step_size_down=total_steps/(2*10),
                                                           cycle_momentum=False)

        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0
        self.target_replace_iter = target_replace_iter
        self.net_dir = net_dir
        self.memory_dir = memory_dir

        if not os.path.exists(self.net_dir):
            os.makedirs(self.net_dir)
        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir)

        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device

    def warmup(self):
        fake_state = torch.rand(self.state_size).to(self.device)
        fake_condition = torch.rand(self.condition_size).to(self.device)
        fake_status = torch.rand(self.status_size).to(self.device)

        action_idx = self.choose_action_idx(fake_state, fake_condition, fake_status, True)

        q_eval = self.eval_net(fake_state.unsqueeze(0), fake_condition, fake_status)
        q_next = self.target_net(fake_state.unsqueeze(0), fake_condition, fake_status).detach()
        loss = self.loss_func(q_eval, q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del fake_state, fake_condition, fake_status
        del q_eval, q_next, loss

    def choose_action_idx(self, state, condition, status, evaluate=False):
        if torch.rand(1) < self.epsilon and not evaluate:
            actions_value = torch.rand(self.out_classes)
        else:
            actions_value = self.eval_net(state.unsqueeze(0), condition, status)
        action_idx = torch.argmax(actions_value).to(self.device)
        return action_idx

    def convert_to_init(self, elements_list):
        res = []
        for elements in elements_list:
            res.append(torch.stack(elements).to(self.device))

        if len(elements_list) == 1:
            return res[0]
        return res

    def convert_to_next(self, cur_elements_list, new_element_list):
        res = []
        for cur_elements, new_element in zip(cur_elements_list, new_element_list):
            cur_elements = cur_elements.to(self.device)
            new_element = new_element.to(self.device)
            res.append(torch.cat((cur_elements, new_element.unsqueeze(0)))[1:])
        return res

    def convert_to_experience(self, *experience):
        res = []
        for e in experience:
            res.append(e.cpu())
        return res

    def store_experiences(self, experiences):
        self.memory.extend(experiences)

    def random_sample_experience(self):
        idx = torch.randint(len(self.memory), (1,))[0]
        memory_data = self.memory[idx]

        state = memory_data[0].to(self.device)
        condition = memory_data[1].to(torch.int64).to(self.device)
        status = memory_data[2].to(self.device)
        action_idx = memory_data[3].to(torch.int64).to(self.device)
        action = memory_data[4].to(torch.int64).to(self.device)
        reward = memory_data[5].to(self.device)
        next_state = memory_data[6].to(self.device)
        next_condition = memory_data[7].to(torch.int64).to(self.device)
        next_status = memory_data[8].to(self.device)

        del idx, memory_data
        return (state, condition, status,
                action_idx, action, reward,
                next_state, next_condition, next_status)

    @property
    def can_learn(self):
        # return len(self.memory) >= self.memory.maxlen / 4
        return len(self.memory) == self.memory.maxlen

    def learn(self, batch_learn=1):
        for i_learn in range(batch_learn):
            (state, condition, status,
             action_idx, action, reward,
             next_state, next_condition, next_status) = self.random_sample_experience()

            q_eval = self.eval_net(state.unsqueeze(0), condition, status)[0, action_idx]
            q_next = self.target_net(next_state.unsqueeze(0), next_condition, next_status).detach()
            q_target = reward + self.gamma * q_next.max()
            loss = self.loss_func(q_eval, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.learn_step_counter += 1
            if self.learn_step_counter % self.target_replace_iter == 0:
                print(f"copy target from eval")
                self.target_net.load_state_dict(self.eval_net.state_dict())

            del state, condition, status
            del action_idx, action, reward
            del next_state, next_condition, next_status
            del q_eval, q_next, q_target, loss

    def save(self, name):
        print(f"saving net", end='\r')
        FileAdmin.safe_save_net(self.eval_net, f"{self.net_dir}/{name}", quiet=True)
        print(f"saving net completed")

        print(f"saving memory", end='\r')
        self.memory.save(f"{self.memory_dir}/{name}")
        print(f"saving memory completed")

    def load(self, name):
        print(f"loading net", end='\r')
        self.eval_net = FileAdmin.safe_load_net(self.eval_net, f"{self.net_dir}/{name}", quiet=True)
        self.target_net = FileAdmin.safe_load_net(self.target_net, f"{self.net_dir}/{name}", quiet=True)
        print(f"loading net completed")

        print(f"loading memory", end='\r')
        self.memory.load(f"{self.memory_dir}/{name}")
        print(f"loading memory completed")

        self.warmup()
