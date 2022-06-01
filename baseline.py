"""
A2C baseline
Training with V-trace return
"""
import gym
import numpy as np
import random
import time
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from models import ActorCritic

import matplotlib.pyplot as plt

ITERATION_NUMS = 500
SAMPLE_NUMS = 100
LR = 0.01
LAMBDA = 0.99
CLIP_GRAD_NORM = 40


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
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.0, total_iters=ITERATION_NUMS)

    gamma = 0.99

    iterations = []
    test_results = []

    init_state = task.reset()
    for i in range(ITERATION_NUMS):
        states, actions, returns, current_state = roll_out(agent, task, SAMPLE_NUMS, init_state, gamma)
        init_state = current_state
        train(agent, optim, scheduler, states, actions, returns, ACTION_DIM)

        # testing
        if (i + 1) % 10 == 0:
            result = test(gym_name, agent)
            print("iteration:", i + 1, "test result:", result / 10.0)
            iterations.append(i + 1)
            test_results.append(result / 10)

    return test_results


def roll_out(agent, task, sample_nums, init_state, gamma):
    states = []
    actions = []
    rewards = []
    vs = []
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
        _, v_t_1 = agent(torch.Tensor(next_state))

        rewards.append(reward)
        vs.append(v_t_1.detach().numpy())
        state = next_state
        if done:
            state = task.reset()  # fatal bug happened
            break

    return states, actions, calculate_returns(rewards, vs, gamma), state


def calculate_returns(rewards, vs, gamma):
    returns = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        R = rewards[t] + gamma * (1 - LAMBDA) * vs[t] + gamma * LAMBDA * R
        returns[t] = R
    return returns


def train(agent, optim, scheduler, states, actions, returns, action_dim):
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
    q = (q - q.mean()) / (q.std(unbiased=False) + 1e-12)
    a = q - v
    a = (a - a.mean()) / (a.std(unbiased=False) + 1e-12)

    loss_policy = - (a.detach() * log_probs_act).mean()
    # loss_critic = a.pow(2.).sum()
    loss_critic = F.mse_loss(q, v, reduction='mean')
    loss_entropy = - (log_probs * probs).mean()

    loss = loss_policy + .5 * loss_critic - .001 * loss_entropy
    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP_GRAD_NORM)
    optim.step()
    scheduler.step()


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
    date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    total_test_results = []
    for random_seed in range(30):
        test_results = run(random_seed)
        total_test_results.append(test_results)

    dir = 'learning_results' + date + '.npy'
    np.save(dir, total_test_results)
