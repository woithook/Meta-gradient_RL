"""
To deal with non-stationarity in the value function and policy which caused by the rapid change of gamma,
we utilise an idea similar to universal value function approximation (UVFA).
This means that gamma are combined with observation as inputs of the neural network (i.e., gamma as additional inputs).
"""
import gym
import numpy as np
import random
from copy import deepcopy
import argparse
import time
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from models import ActorCritic

import matplotlib.pyplot as plt

# parser = argparse.ArgumentParser(description='Meta-grad learning of gamma under CartPole-v1')
# parser.add_argument('--iteration_num', type=int, default=500, help='number of total iteration')
# parser.add_argument('--sample_num', type=int, default=100, help='length of single trajectory')
# parser.add_argument('--lr', type=float, default=0.01, help='learning rate of reinforcement learning')
# parser.add_argument('--clip_grad_norm', type=float, default=0.5, help='grad norm limitation for agent policy update')
# parser.add_argument('--mu', type=float, default=0.0, help='hyper-parameter decays the meta-gradient trace and '
#                                                           'focuses on recent updates')
# parser.add_argument('--beta', type=float, default=0.0001, help='learning rate of meta-learning')
# parser.add_argument('--gamma_init', type=float, default=0.99, help='init value of gamma')
# parser.add_argument('--gamma_fix', type=float, default=1.0, help='fixed gamma in meta-objective function')


ITERATION_NUMS = 500
SAMPLE_NUMS = 100
LR = 0.01
LAMBDA = torch.tensor(0.99)
CLIP_GRAD_NORM = 40
MU = 0.
BETA = 0.001
GAMMA_INIT = 0.99
GAMMA_FIX = 0.995


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

    agent = ActorCritic(STATE_DIM + 1, ACTION_DIM)
    optim = torch.optim.Adam(agent.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.0, total_iters=ITERATION_NUMS)

    iterations = []
    test_results = []

    # Define gamma as a tensor for meta-gradient calculating
    gamma = torch.tensor(GAMMA_INIT, requires_grad=True)
    z_dash = torch.tensor(0)
    gamma_buffer = []

    init_state = task.reset()
    for i in range(ITERATION_NUMS):
        states, actions, returns, current_state = roll_out(agent, task, SAMPLE_NUMS, init_state, gamma)
        init_state = current_state
        z = train(agent, optim, scheduler, states, actions, returns, gamma, ACTION_DIM)
        z_dash = MU * z_dash + z

        # Cross-validating trajectory
        states_dash, actions_dash, returns_dash, current_state = roll_out(agent, task, SAMPLE_NUMS, init_state,
                                                                          GAMMA_FIX)
        init_state = current_state
        gamma_buffer.append(deepcopy(gamma.detach().numpy()))
        gamma = meta_grad(agent, states_dash, actions_dash, returns_dash, gamma, z_dash, ACTION_DIM)

        # testing
        if (i + 1) % 10 == 0:
            result = test(gym_name, agent, gamma)
            print("iteration:", i + 1, "test result:", result / 10.0, "gamma:", gamma.data)
            iterations.append(i + 1)
            test_results.append(result / 10)
            # Break when the reward in an epoch is higher than the reward threshold
            # if test_results[-1] > task.spec.reward_threshold:
            #     break

    return test_results


def roll_out(agent, task, sample_nums, init_state, gamma):
    states = []
    actions = []
    rewards = []
    vs = []
    state = init_state
    gamma_input = gamma.detach().numpy() if torch.is_tensor(gamma) else gamma

    for i in range(sample_nums):
        state = np.append(state, gamma_input)
        states.append(state)
        with torch.no_grad():
            logits, _ = agent(torch.Tensor(state))
            act_probs = F.softmax(logits, dim=-1)
            m = Categorical(act_probs)
            act = m.sample()
            actions.append(act)

        next_state, reward, done, _ = task.step(act.numpy())
        _, v_t_1 = agent(torch.Tensor(np.append(next_state, gamma_input)))

        rewards.append(reward)
        vs.append(v_t_1.detach().numpy())
        state = next_state
        if done:
            state = task.reset()  # fatal bug happened
            break

    return states, actions, calculate_returns_with_grad(rewards, vs, gamma), state


def calculate_returns_with_grad(rewards, vs, gamma):
    returns = torch.zeros_like(torch.tensor(rewards))
    R = torch.reshape(torch.tensor(0), (1, 1))
    for t in reversed(range(0, len(rewards))):
        R = torch.tensor(rewards[t]) + gamma * (1 - LAMBDA) * torch.tensor(vs[t]) + gamma * LAMBDA * R
        returns[t] = R
    return returns


def train(agent, optim, scheduler, states, actions, returns, gamma, action_dim):
    def compute_z(agent_, log_probs_act_, v_, returns_, gamma_, lr, b):
        lr = torch.tensor(lr)
        b = torch.tensor(b)

        # compute trace z (Equation (13))
        pi_ = log_probs_act_.mean()
        v_ = v_.mean()
        returns_ = returns_.mean()

        # theta1: d(log_probs_act) / d_theta
        # theta2: d_v / d_theta
        theta1 = torch.autograd.grad(pi_, agent_.parameters(), retain_graph=True, allow_unused=True)
        theta2 = torch.autograd.grad(v_, agent_.parameters(), allow_unused=True)
        dg_dgamma = torch.autograd.grad(returns_, gamma_)

        theta1 = [torch.zeros_like(params.data) if item is None else item
                  for (item, params) in zip(theta1, agent_.parameters())]
        theta2 = [torch.zeros_like(params.data) if item is None else item
                  for (item, params) in zip(theta2, agent_.parameters())]
        theta1 = [item.view(-1) for item in theta1]
        theta2 = [item.view(-1) for item in theta2]
        theta1 = torch.cat(theta1)
        theta2 = torch.cat(theta2)

        dg_dgamma = dg_dgamma[0]
        z = lr * dg_dgamma * (theta1 + b * theta2)

        return z

    states = torch.Tensor(np.array(states))
    actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1)

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
    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP_GRAD_NORM)

    z = compute_z(agent, log_probs_act, v, returns, gamma, LR, .5)

    optim.step()
    scheduler.step()

    return z.clone().detach().requires_grad_(True)


def meta_grad(agent, states_dash, actions_dash, returns_dash, gamma, z_dash, action_dim):
    agent.zero_grad()
    states = torch.Tensor(np.array(states_dash))
    actions = torch.tensor(actions_dash, dtype=torch.int64).view(-1, 1)

    logits, v = agent(states)
    logits = logits.view(-1, action_dim)
    v = v.view(-1)
    log_probs = F.log_softmax(logits, dim=1)
    log_probs_act = log_probs.gather(1, actions).view(-1)

    # Compute Equation(14)
    q = returns_dash
    adv = q - v
    pi = adv.detach() * log_probs_act
    # J1: d(log_probs_act_dash) / d(theta_dash)
    J1 = torch.autograd.grad(pi.mean(), agent.parameters(), allow_unused=True)
    J1 = [torch.zeros_like(params.data) if item is None else item
          for (item, params) in zip(J1, agent.parameters())]
    J1 = [item.view(-1) for item in J1]
    J1 = torch.cat(J1)

    n_delta = - BETA * torch.dot(J1, z_dash)
    gamma.data += n_delta

    # Limit gamma into [0, 1]
    if gamma.data > torch.tensor(1.):
        gamma.data = torch.tensor(1.)
    elif gamma.data < torch.tensor(0.):
        gamma.data = torch.tensor(0.)

    agent.zero_grad()

    return gamma.clone().detach().requires_grad_(True)


def test(gym_name, agent, gamma):
    gamma_input = gamma.detach().numpy() if torch.is_tensor(gamma) else gamma

    result = 0
    test_task = gym.make(gym_name)
    for test_epi in range(10):
        state = test_task.reset()
        for test_step in range(500):
            with torch.no_grad():
                state = np.append(state, gamma_input)
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
