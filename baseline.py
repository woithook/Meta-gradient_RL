import gym
import numpy as np
import random
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from models import ActorCritic

import matplotlib.pyplot as plt

ITERATION = 500
SAMPLE_NUMS = 100
LR = 0.01


def run(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    gym_name = "CartPole-v1"
    task = gym.make(gym_name)
    task.seed(random_seed)

    discrete = isinstance(task.action_space, gym.spaces.Discrete)
    STATE_DIM = task.observation_space.shape[0]
    ACTION_DIM = task.action_space.n if discrete else task.action_space.shape[0]

    agent = ActorCritic(STATE_DIM, ACTION_DIM)
    optim = torch.optim.Adam(agent.parameters(), lr=LR)

    iterations = []
    test_results = []

    gamma = 0.99

    init_state = task.reset()

    for i in range(ITERATION):
        states, actions, returns, current_state = roll_out(agent, task, SAMPLE_NUMS, init_state, gamma)
        train(agent, optim, states, actions, returns, ACTION_DIM)

        # testing
        if (i + 1) % 10 == 0:
            result = test(gym_name, agent)
            print("iteration:", i + 1, "test result:", result / 10.0)
            iterations.append(i + 1)
            test_results.append(result / 10)
            if test_results[-1] > task.spec.reward_threshold:
                break

    return test_results


def roll_out(agent, task, sample_nums, init_state, gamma):
    is_done = False
    states = []
    actions = []
    rewards = []
    final_r = 0
    state = init_state

    for i in range(sample_nums):
        states.append(state)
        with torch.no_grad():
            logits, _ = agent(torch.Tensor(state))
            act_probs = F.softmax(logits, dim=-1)
            m = Categorical(act_probs)
            act = m.sample()
            actions.append(act)

        next_state, reward, done, _ = task.step(act.numpy())
        rewards.append(reward)
        state = next_state
        if done:
            is_done = True
            task.reset()
            break
    if not is_done:
        _, final_r = agent(torch.Tensor(state))

    def calculate_returns(rewards, final_r=0, gamma=0.99):
        returns = []
        R = final_r
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        return returns

    return states, actions, calculate_returns(rewards, final_r, gamma), state


def train(agent, optim, states, actions, returns, action_dim):
    agent.zero_grad()
    optim.zero_grad()
    states = torch.Tensor(np.array(states))
    actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1)
    returns = torch.Tensor(returns)

    logits, v = agent(states)
    logits = logits.view(-1, action_dim)
    v = v.view(-1)
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    log_probs_act = log_probs.gather(1, actions).view(-1)

    q = returns.detach()
    # q = (q - q.mean()) / (q.std() + 1e-8)
    a = q - v
    # a = (a - a.mean()) / (a.std() + 1e-8)

    criterion = torch.nn.MSELoss()
    loss_policy = - (a.detach() * log_probs_act).sum()
    loss_critic = criterion(v, q)
    loss_entropy = (log_probs * probs).sum()

    loss = loss_policy + .5 * loss_critic + .05 * loss_entropy
    loss.backward()

    torch.nn.utils.clip_grad_norm_(agent.parameters(), 20)
    optim.step()


def test(gym_name, agent):
    result = 0
    test_task = gym.make(gym_name)
    for test_epi in range(10):
        state = test_task.reset()
        for test_step in range(500):
            with torch.no_grad():
                logits, _ = agent(torch.Tensor(state))
                act_probs = F.softmax(logits, dim=-1)
                m = Categorical(act_probs)
                act = m.sample()
            next_state, reward, done, _ = test_task.step(act.numpy())
            result += reward
            state = next_state
            if done:
                break
    return result


if __name__ == '__main__':
    for random_seed in range(30):
        test_results = run(random_seed)
