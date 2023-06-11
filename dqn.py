import torch
import torch.nn as nn
import numpy as np
from network import ResNet18

class DQN(nn.Module):
    # def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity):
    def __init__(self, state_size, n_actions, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity):
        super().__init__()
        # self.eval_net, self.target_net = Net(n_states, n_actions, n_hidden), Net(n_states, n_actions, n_hidden)
        self.eval_net, self.target_net = ResNet18(n_actions), ResNet18(n_actions)

        self.state_size = state_size
        self.n_states = np.prod(state_size)
        self.memory = np.zeros((memory_capacity, self.n_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0
        self.learn_step_counter = 0

        self.n_actions = n_actions
        # self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity

    def choose_action(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state), 0)
        # epsilon-greedy
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
            # action = np.random.randint(2, size=self.n_actions)
        else:
            actions_value = self.eval_net(x)
            # action = actions_value > 0.5
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        return action


    def store_transition(self, state, action, reward, next_state):
        # print(f"{state.shape = }")
        # print(f"{action = }")
        # print(f"{reward = }")
        # print(f"{next_state.shape = }")
        transition = np.hstack((state.flatten(), [action, reward], next_state.flatten()))

        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_state = torch.FloatTensor(b_memory[:, :self.n_states])
        b_action = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int))
        b_reward = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2])
        b_next_state = torch.FloatTensor(b_memory[:, -self.n_states:])

        b_state = b_state.reshape(self.state_size).unsqueeze(0)
        # b_action = b_action[0]
        # b_reward = b_reward[0]
        b_next_state = b_next_state.reshape(self.state_size).unsqueeze(0)

        q_eval = self.eval_net(b_state).gather(1, b_action)
        q_next = self.target_net(b_next_state).detach()
        q_target = b_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
