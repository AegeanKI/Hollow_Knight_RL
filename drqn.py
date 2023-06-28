import os
import time

import torch
import torch.nn as nn

from pc import FileAdmin, Logger
from hkenv import Action
from utils import Memory
from network import ResNetLSTM

class DRQN(nn.Module):
    def __init__(self, state_size, condition_size, n_frames,
                 out_classes, lr, total_steps, epsilon, gamma, memory_capacity, 
                 target_replace_iter, net_dir, memory_dir, device):
        super().__init__()

        self.state_size = state_size
        self.condition_size = condition_size
        self.n_frames = n_frames

        self.out_classes = out_classes
        self.eval_net = ResNetLSTM(out_classes=out_classes,
                                   in_condition_size=self.condition_size).to(device)
        self.target_net = ResNetLSTM(out_classes=out_classes,
                                     in_condition_size=self.condition_size).to(device)

        self.memory = Memory(memory_capacity, (state_size, (condition_size,),
                                               (1,), (1,), (1,), state_size, (condition_size,)))

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,
                                                           base_lr=lr/10,
                                                           max_lr=lr,
                                                           gamma=1.0,
                                                           step_size_up=total_steps/(2*20),
                                                           step_size_down=total_steps/(2*20),
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

        # self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device

    def warmup(self):
        fake_state = torch.rand(self.state_size).to(self.device)
        fake_condition = torch.rand(self.condition_size).to(self.device)

        action = self.choose_action(fake_state, fake_condition, train=False)

        q_eval = self.eval_net(fake_state.unsqueeze(0), fake_condition.unsqueeze(0)) # n_frames
        q_next = self.target_net(fake_state.unsqueeze(0), fake_condition.unsqueeze(0)).detach()

        loss = self.loss_func(q_eval, q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del fake_state, fake_condition
        del q_eval, q_next, loss

    def choose_action(self, state, condition, train=False):
        if torch.rand(1) < self.epsilon and train:
            actions_value = torch.rand(self.out_classes)
        else:
            actions_value = self.eval_net(state.unsqueeze(0), condition.unsqueeze(0))
        action_idx = torch.argmax(actions_value)
        return Action(action_idx).to(self.device)

    def store_episode_experiences(self, experiences):
        self.memory.extend(experiences)

    @property
    def can_learn(self):
        # return len(self.memory) == self.memory.maxlen
        return len(self.memory) > 30

    def learn(self, times=1):
        for i_learn in range(times):
            n_frames_experiences = self.memory.random_sample(self.n_frames)
            n_frames_experiences = [data.to(self.device) for data in n_frames_experiences]

            (n_frames_state, n_frames_condition,
             n_frames_action_idx, n_frames_reward, n_frames_done,
             n_frames_next_state, n_frames_next_condition) = n_frames_experiences

            q_eval = self.eval_net(n_frames_state, n_frames_condition)
            q_eval = torch.gather(q_eval, dim=1, index=n_frames_action_idx.long())
            q_next = self.target_net(n_frames_next_state, n_frames_next_condition).detach()
            q_target = n_frames_reward + self.gamma * q_next.max(dim=1)[0].view(-1, 1) * (1 - n_frames_done)
            loss = self.loss_func(q_eval, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.learn_step_counter += 1
            if self.learn_step_counter % self.target_replace_iter == 0:
                print(f"copy target from eval")
                self.target_net.load_state_dict(self.eval_net.state_dict())

            del n_frames_state, n_frames_condition
            del n_frames_action_idx, n_frames_reward, n_frames_done
            del n_frames_next_state, n_frames_next_condition
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
