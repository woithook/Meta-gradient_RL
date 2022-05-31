import gym
import numpy as np
import random
from copy import deepcopy
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from models import ActorCritic

import matplotlib.pyplot as plt

ITERATION = 500
SAMPLE_NUMS = 100
LR = 0.006
CLIP_GRAD_NORM = 0.5
MU = 0.
BETA = 0.0001
GAMMA_INIT = 0.99
GAMMA_FIX = 1.


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

    # Define gamma as a tensor for meta-gradient calculating
    gamma = torch.tensor(GAMMA_INIT, requires_grad=True)
    z_dash = torch.tensor(0)
    gamma_buffer = []

    init_state = task.reset()

    for i in range(ITERATION):
        states, actions, returns, current_state = roll_out(agent, task, SAMPLE_NUMS, init_state, gamma)
        init_state = current_state
        z = train(agent, optim, states, actions, returns, gamma, ACTION_DIM)
        z_dash = MU * z_dash + z

        # Cross-validating trajectory
        states_dash, actions_dash, returns_dash, current_state = roll_out(agent, task, SAMPLE_NUMS, init_state,
                                                                          GAMMA_FIX)
        init_state = current_state
        gamma_buffer.append(deepcopy(gamma.detach().numpy()))
        gamma = meta_grad(agent, states_dash, actions_dash, returns_dash, gamma, z_dash, ACTION_DIM)

        # testing
        if (i + 1) % 10 == 0:
            result = test(gym_name, agent)
            print("iteration:", i + 1, "test result:", result / 10.0, "gamma:", gamma.data)
            iterations.append(i + 1)
            test_results.append(result / 10)
            # Break when the reward in an epoch is higher than the reward threshold
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
            state = task.reset()
            break
    if not is_done:
        _, final_r = agent(torch.Tensor(state))
        final_r = final_r.detach().numpy()

    def calculate_returns_with_grad(rewards, final_r, gamma):
        returns = torch.zeros_like(torch.tensor(rewards))
        R = torch.reshape(torch.tensor(final_r), (1, 1))
        for t in reversed(range(0, len(rewards))):
            R = R * gamma + torch.tensor(rewards[t])
            returns[t] = R
        return returns

    return states, actions, calculate_returns_with_grad(rewards, final_r, gamma), state


def train(agent, optim, states, actions, returns, gamma, action_dim):
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

    agent.zero_grad()
    optim.zero_grad()
    states = torch.Tensor(np.array(states))
    actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1)

    logits, v = agent(states)
    logits = logits.view(-1, action_dim)
    v = v.view(-1)
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    log_probs_act = log_probs.gather(1, actions).view(-1)

    q = returns.detach()
    # TODO: trouble shooting of nan emerging while training with normalized q(or a).
    # q = (q - q.mean()) / (q.std() + 1e-8)
    adv = q - v
    # a = (a - a.mean()) / (a.std() + 1e-8)

    criterion = torch.nn.MSELoss()
    loss_policy = - (adv.detach() * log_probs_act).sum()
    loss_critic = criterion(v, q)
    # loss_critic = a.pow(2.).sum()
    loss_entropy = (log_probs * probs).sum()
    loss = loss_policy + .5 * loss_critic + .05 * loss_entropy
    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP_GRAD_NORM)

    z = compute_z(agent, log_probs_act, v, returns, gamma, LR, .5)

    optim.step()
    # Clean grad after computing z
    agent.zero_grad()
    optim.zero_grad()

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
