import time
import torch
import torch.nn as nn
import numpy as np
from network import ResNet18

class DQN(nn.Module):
    # def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity):
    def __init__(self, state_size, n_actions, batch_size, lr, epsilon,
                 gamma, target_replace_iter, memory_capacity, device):
        super().__init__()
        # self.eval_net, self.target_net = Net(n_states, n_actions, n_hidden), Net(n_states, n_actions, n_hidden)
        self.eval_net, self.target_net = ResNet18(n_actions).to(device), ResNet18(n_actions).to(device)

        self.state_size = state_size
        self.n_states = torch.prod(torch.Tensor(state_size)).to(torch.int64)
        self.memory = torch.zeros((memory_capacity, self.n_states.item() * 2 + 2)).to(device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0
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
        fake_input = torch.zeros(self.state_size).unsqueeze(0).to(self.device)
        q_eval = self.eval_net(fake_input)
        q_next = self.target_net(fake_input).detach()
        q_target = q_eval
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def choose_action(self, state):
        x = state.float().unsqueeze(0)
        # epsilon-greedy
        if torch.rand(1) < self.epsilon:
            action = torch.randint(self.n_actions, (1,))
        else:
            pass_net_time = time.time()
            actions_value = self.eval_net(x)
            action = torch.argmax(actions_value)
        return torch.Tensor([action]).to(torch.int64).to(self.device)


    def store_transition(self, state, action, reward, next_state):
        transition = torch.hstack((state.flatten(), action, reward, next_state.flatten()))

        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):
        start = time.time()
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)

        b_memory = self.memory[sample_index, :]
        b_state = b_memory[:, :self.n_states]
        b_action = b_memory[:, self.n_states:self.n_states+1].to(torch.int64)
        b_reward = b_memory[:, self.n_states+1]
        b_next_state = b_memory[:, -self.n_states:]

        b_state = b_state.reshape(self.state_size).unsqueeze(0)
        # b_action = b_action[0]
        # b_reward = b_reward[0]
        b_next_state = b_next_state.reshape(self.state_size).unsqueeze(0)

        q_eval = torch.gather(self.eval_net(b_state), 1, b_action)
        q_next = self.target_net(b_next_state).detach()
        q_target = b_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            print(f"copy target from eval")
            self.target_net.load_state_dict(self.eval_net.state_dict())
