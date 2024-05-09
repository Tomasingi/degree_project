import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
from collections import namedtuple, deque

from model import MiniAllocationData

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()

        self.n_hidden = 32

        self.fc1 = nn.Linear(in_dim, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.final = nn.Linear(self.n_hidden, out_dim)

        # self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x = self.linear(x)

        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        x = self.final(x)

        return x

class Agent:
    def __init__(self):
        self.D = MiniAllocationData()
        # self.n_doctors = len(self.D.doctors)
        self.n_doctors = 2
        self.n_slots = 2
        self.n_days = 5

        self.reset_state()

        # self.order_weights = torch.tensor([4, 5, 5.5, 6.5, 9], dtype=torch.float)
        self.order_weights = torch.arange(self.n_slots)

        self.gamma = 0.9

        self.trainable = True

    def reset_state(self):
        self.S = {
            'day': 0,
            'worked_days': torch.ones(self.n_doctors, dtype=torch.float),
            'points': torch.ones(self.n_doctors, dtype=torch.float),
            'hours': torch.zeros(self.n_doctors, dtype=torch.float),
            'schedule': torch.zeros((self.n_days, self.n_slots)), # days x slots x doctors
            'temp_scheduled': torch.zeros(self.n_doctors, dtype=torch.float),
            # 'working': torch.zeros(self.n_doctors, dtype=torch.float),
            'working': torch.ones(self.n_doctors, dtype=torch.float),
            'cumulative_reward': 0,
        }
        self.S['working'][np.random.choice(self.n_doctors, self.n_slots, replace=False)] = 1

    def features(self, state):
        a1 = torch.logical_and(state['working'], torch.logical_not(state['temp_scheduled']))
        a2 = state['points']
        return a2[None, :]
        # return torch.cat((a1, a2))[None, :]

    def next_state(self, state, doctor):
        new_state = copy.deepcopy(state)
        rank = torch.sum(new_state['temp_scheduled'] > 0)

        new_state['schedule'][self.S['day']%self.n_days,rank] = doctor

        new_state['worked_days'][doctor] += 1
        new_state['points'][doctor] += rank + 1
        new_state['hours'][doctor] += self.order_weights[rank]

        new_state['temp_scheduled'][doctor] = rank + 1

        new_state['cumulative_reward'] *= self.gamma
        new_state['cumulative_reward'] += self.reward(state, new_state)

        if torch.sum(new_state['temp_scheduled'] > 0) == self.n_slots:
            new_state['temp_scheduled'] = torch.zeros(self.n_doctors, dtype=torch.float)
            # new_state['working'] = torch.zeros(self.n_doctors, dtype=torch.float)
            # new_state['working'][np.random.choice(self.n_doctors, self.n_slots, replace=False)] = 1
            new_state['working'] = torch.ones(self.n_doctors, dtype=torch.float)

            new_state['day'] = self.S['day'] + 1


        return new_state

    def sample_action(self, epsilon=0):
        pass

    def perform_action(self, action):
        new_state = self.next_state(self.S, action)
        self.S = new_state

    def reward(self, s1, s2):
        # v1 = -(torch.max(s1['hours'] / s1['worked_days']) - torch.min(s1['hours'] / s1['worked_days']))
        # v2 = -(torch.max(s2['hours'] / s2['worked_days']) - torch.min(s2['hours'] / s2['worked_days']))
        v1 = -torch.std(s1['hours'] / s1['worked_days'])
        v2 = -torch.std(s2['hours'] / s2['worked_days'])
        # v1 = -torch.std(s1['points'])
        # v2 = -torch.std(s2['points'])
        res = 10 * (v2 - v1)
        cap = 10
        # if res > cap:
        #     res = cap
        # if res < -cap:
        #     res = -cap
        return res

    def learn(self):
        pass

    def schedule_day(self, epsilon=0, learn=True):
        if not learn:
            epsilon = 0
        for _ in range(self.n_slots):
            action = self.sample_action(epsilon)
            self.perform_action(action)
            if learn:
                self.learn()

    def schedule_week(self, learn=True):
        for day in range(self.n_days):
            self.schedule_day(learn=learn)

    def schedule_epoch(self, num_episodes=100, learn=True):
        history = list()
        for _ in range(num_episodes):
            for day in range(self.n_days):
                self.schedule_day(learn=learn)
                history.append(-torch.std(self.S['hours'] / self.S['worked_days']))
                # history.append(self.S['cumulative_reward'])

        return history

    def print_schedule(self):
        schedule = self.S['schedule']
        n_rows = self.n_slots
        n_cols = self.n_days

        hline = '+---' * n_cols + '+\n'
        hempty = '|   ' * n_cols + '|\n'
        center = f' |\n{hempty}{hline}{hempty}| '.join(" | ".join([str(int(j+0.5)) for j in schedule[:,i]]) for i in range(n_rows))
        print(f'{hline}{hempty}| {center} |\n{hempty}{hline}')

class RandomAgent(Agent):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def sample_action(self, epsilon=0):
        return np.random.choice(torch.where(torch.logical_and(self.S['working'], torch.logical_not(self.S['temp_scheduled'])))[0])

class SimpleHeuristicAgent(Agent):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def sample_action(self, epsilon=0):
        return torch.argmax(self.S['points'] * torch.logical_and(self.S['working'], torch.logical_not(self.S['temp_scheduled'])))

class AverageHeuristicAgent(Agent):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def sample_action(self, epsilon=0):
        action_scores = self.S['points'] / self.S['worked_days']
        mask = torch.logical_and(self.S['working'], torch.logical_not(self.S['temp_scheduled']))
        temp = torch.argmax(action_scores[mask])
        index = torch.arange(len(action_scores))[mask][temp]
        return index

class DeepQAgent(Agent):
    def __init__(self, epsilon_init, epsilon_final, tau=0.005):
        super().__init__()

        self.q = Net(self.n_doctors, self.n_doctors) # points + temp order
        self.optimizer = optim.AdamW(self.q.parameters(), lr=0.0001, amsgrad=True)
        self.q_hat = Net(self.n_doctors, self.n_doctors)

        self.alpha = 0.1
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.tau = tau

        self.memory = ReplayMemory(10000)
        self.batch_size = 32

        self.actions_taken = 0

    def sample_action(self, epsilon):
        p = np.random.rand()

        if p < epsilon:
            return np.random.choice(torch.where(torch.logical_and(self.S['working'], torch.logical_not(self.S['temp_scheduled'])))[0])
        else:
            with torch.no_grad():
                action_scores = self.q(self.features(self.S)).detach()
                mask = torch.logical_and(self.S['working'], torch.logical_not(self.S['temp_scheduled']))
                mask = torch.arange(action_scores.shape[1])[torch.logical_and(self.S['working'], torch.logical_not(self.S['temp_scheduled']))]
                temp = torch.argmax(action_scores[:,mask])
                index = torch.arange(action_scores.shape[1])[mask][temp]
                return index
                # return action_scores.max(1).indices.view(1, 1)

    def perform_action(self, action):
        new_state = self.next_state(self.S, action)
        self.memory.push(self.features(self.S), torch.tensor([[action]]), torch.tensor([[self.reward(self.S, new_state)]]), self.features(new_state))
        self.S = new_state
        self.actions_taken += 1

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        state_action_values = self.q(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, dtype=torch.float)
        with torch.no_grad():
            next_state_values = self.q_hat(next_state_batch)

        max_values = next_state_values.max(1).values.unsqueeze(1)
        min_values = next_state_values.min(1).values.unsqueeze(1)
        diffs = 1 + max_values - min_values
        unavailable_mask = 1 - next_state_batch[:,:self.n_doctors]
        next_state_values -= diffs * unavailable_mask

        next_state_values = next_state_values.max(1).values.unsqueeze(1)

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.q.parameters(), 100)
        self.optimizer.step()

        q_hat_state_dict = self.q_hat.state_dict()
        q_state_dict = self.q.state_dict()
        for key in q_state_dict:
            q_hat_state_dict[key] = q_state_dict[key]*self.tau + q_hat_state_dict[key]*(1-self.tau)
        self.q_hat.load_state_dict(q_hat_state_dict)

    def schedule_epoch(self, num_episodes=100, learn=True):
        history = list()
        for i in range(num_episodes):
            for day in range(self.n_days):
                epsilon = self.epsilon_init + (self.epsilon_final - self.epsilon_init) * (self.n_days * i + day) / (self.n_days * num_episodes)
                self.schedule_day(epsilon, learn=learn)
                history.append(-torch.std(self.S['hours'] / self.S['worked_days']))
                # history.append(self.S['cumulative_reward'])

        return history

def main():
    pass

if __name__ == '__main__':
    main()