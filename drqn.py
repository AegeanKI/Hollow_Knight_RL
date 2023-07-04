import os
import time

import torch
import torch.nn as nn

from pc import FileAdmin, Logger
from hkenv import Action
from utils import Memory
from network import ResNetLSTM

class DRQN(nn.Module):
    def __init__(self, state_size, condition_size,
                 out_classes, lr, total_steps, epsilon, gamma,
                 target_replace_iter, net_dir, device):
        super().__init__()

        self.state_size = state_size
        self.condition_size = condition_size

        self.out_classes = out_classes
        self.eval_net = ResNetLSTM(out_classes=out_classes,
                                   in_condition_size=self.condition_size).to(device)
        self.target_net = ResNetLSTM(out_classes=out_classes,
                                     in_condition_size=self.condition_size).to(device)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,
                                                           base_lr=lr/100,
                                                           max_lr=lr,
                                                           gamma=0.9,
                                                           step_size_up=total_steps/(2*40),
                                                           step_size_down=total_steps/(2*40),
                                                           cycle_momentum=False)

        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0
        self.target_replace_iter = target_replace_iter
        self.net_dir = net_dir

        if not os.path.exists(self.net_dir):
            os.makedirs(self.net_dir)

        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device

    def reset_net(self):
        self.eval_net.init_lstm_hidden()
        self.target_net.init_lstm_hidden()

    def warmup(self):
        self.reset_net()

        fake_state = torch.rand(self.state_size).to(self.device)
        fake_condition = torch.rand(self.condition_size).to(self.device)

        actions_value = self.eval_net(fake_state.unsqueeze(0), fake_condition.unsqueeze(0))
        print(f"warmup {actions_value.max().item() = }, {actions_value.min().item() = }")

        q_eval = self.eval_net(fake_state.unsqueeze(0), fake_condition.unsqueeze(0))
        q_next = self.target_net(fake_state.unsqueeze(0), fake_condition.unsqueeze(0)).detach()

        loss = self.loss_func(q_eval, q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del fake_state, fake_condition
        del actions_value
        del q_eval, q_next, loss

    def choose_action(self, state, condition, train=False):
        epsilon = False
        if torch.rand(1) < self.epsilon and train:
            actions_value = torch.rand(self.out_classes)
            epsilon = True
        else:
            actions_value = self.eval_net(state.unsqueeze(0), condition.unsqueeze(0))
        action_idx = torch.argmax(actions_value)
        return Action(action_idx).to(self.device), epsilon

    def learn(self, samples):
        for i_learn, sample in enumerate(samples):
            print(f"learning {i_learn}", end='\r')
            experience = [s.to(self.device) for s in sample]

            self.reset_net()

            (state, condition,
             action_idx, reward, done, affect,
             next_state, next_condition) = experience

            q_eval = self.eval_net(state, condition)
            q_eval = torch.gather(q_eval, dim=1, index=action_idx.long())

            next_action_value = self.eval_net(next_state, next_condition).detach()
            next_action_idx = next_action_value.argmax(dim=1)[0].view(-1, 1)
            q_next = self.target_net(next_state, next_condition).detach()
            q_next = torch.gather(q_next, dim=1, index=next_action_idx.long())

            q_target = reward + self.gamma * q_next * (1 - done)

            loss = self.loss_func(q_eval, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.learn_step_counter += 1
            if self.learn_step_counter % self.target_replace_iter == 0:
                print(f"copy target from eval")
                self.target_net.load_state_dict(self.eval_net.state_dict())

            del state, condition
            del action_idx, reward, done
            del next_state, next_condition
            del q_eval, q_next, q_target, loss
        Logger.clear_line()
        print(f"learning completed")

    def save(self, name):
        print(f"saving net", end='\r')
        FileAdmin.safe_save_net(self.eval_net, f"{self.net_dir}/{name}", quiet=True)
        print(f"saving net completed")

    def load(self, name):
        print(f"loading net", end='\r')
        self.eval_net = FileAdmin.safe_load_net(self.eval_net, f"{self.net_dir}/{name}", quiet=True)
        self.target_net = FileAdmin.safe_load_net(self.target_net, f"{self.net_dir}/{name}", quiet=True)
        print(f"loading net completed")
