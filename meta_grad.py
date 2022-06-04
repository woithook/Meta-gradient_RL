import gym
import numpy as np
import random
from copy import deepcopy
import time
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from models import ActorCritic

import matplotlib.pyplot as plt

ITERATION_NUMS = 1000  # 500
SAMPLE_NUMS = 50  # 100
LR = 0.01
LAMBDA = torch.tensor(0.98)
CLIP_GRAD_NORM = 40  # 40
MU = 0.
BETA = 0.001
GAMMA_INIT = 0.99  # 0.99
GAMMA_FIX = 0.995  # 0.995


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

    iterations = []
    test_results = []

    # Define gamma as a tensor for meta-gradient calculating
    gamma = torch.tensor(GAMMA_INIT, requires_grad=True)
    z_dash = torch.tensor(0)
    gamma_buffer = []

    init_state = task.reset()
    for i in range(ITERATION_NUMS):
        states, actions, advs, v_targets, current_state = roll_out(agent, task, SAMPLE_NUMS, init_state, gamma)
        init_state = current_state
        z = train(agent, optim, scheduler, states, actions, advs, v_targets, gamma, ACTION_DIM)
        z_dash = MU * z_dash + z

        # Cross-validating trajectory
        states_dash, actions_dash, advs_dash, v_targets_dash, current_state = \
            roll_out(agent, task, SAMPLE_NUMS, init_state, GAMMA_FIX)
        init_state = current_state
        gamma_buffer.append(deepcopy(gamma.detach().numpy()))
        gamma = meta_grad(agent, states_dash, actions_dash, advs_dash, gamma, z_dash, ACTION_DIM)

        # testing
        if (i + 1) % (ITERATION_NUMS // 10) == 0:
            result = test(gym_name, agent)
            print("iteration:", i + 1, "test result:", result / 10.0, "gamma:", gamma.data)
            iterations.append(i + 1)
            test_results.append(result / 10)

    return test_results


def roll_out(agent, task, sample_nums, init_state, gamma):
    states = []
    actions = []
    advs = []
    v_targets = []

    rewards = []
    v_t_s = []
    v_t1_s = []
    dones = []

    state = init_state

    for i in range(sample_nums):
        states.append(state)
        act, v_t = choose_action(agent, state)
        actions.append(act)

        next_state, reward, done, _ = task.step(act.numpy())
        with torch.no_grad():
            _, v_t1 = agent(torch.Tensor(next_state))

        rewards.append(reward)
        v_t = v_t.detach().numpy()
        v_t1 = v_t1.detach().numpy()
        v_t_s.append(v_t)
        v_t1_s.append(v_t1)
        dones.append(1 if done is False else 0)

        state = next_state
        if done:
            state = task.reset()
            adv, v_target = gae_calculater_grad(rewards, v_t_s, v_t1_s, dones, gamma, LAMBDA)

            advs.append(adv)
            v_targets.append(v_target)
            rewards = []
            v_t_s = []
            v_t1_s = []
            dones = []
    adv, v_target = gae_calculater_grad(rewards, v_t_s, v_t1_s, dones, gamma, LAMBDA)
    advs.append(adv)
    advs = torch.cat(advs).float()
    v_targets.append(v_target)
    v_targets = torch.cat(v_targets).float()

    return states, actions, advs, v_targets, state


def gae_calculater_grad(rewards, v_t_s, v_t1_s, dones, gamma, lambda_):
    """
    Calculate advantages and target v-values
    """
    batch_size = len(rewards)
    R = torch.reshape(torch.tensor(0), (1, 1))  # increment term
    advs = torch.zeros(batch_size)
    for t in reversed(range(0, batch_size)):
        delta = torch.tensor(rewards[t]) - torch.tensor(v_t_s[t]) + \
                (gamma * torch.tensor(v_t1_s[t]) * torch.tensor(dones[t]))
        R = delta + (gamma * lambda_ * R * torch.tensor(dones[t]))
        advs[t] = R
    value_target = advs + torch.tensor(np.squeeze(v_t_s))  # target v is calculated from adv.

    return advs, value_target


def train(agent, optim, scheduler, states, actions, advs, v_targets, gamma, action_dim):
    def compute_z(agent_, log_probs_act_, v_, v_targets_, gamma_, lr, b):
        lr = torch.tensor(lr)
        b = torch.tensor(b)

        # compute trace z (Equation (13))
        pi_ = log_probs_act_.mean()
        v_ = v_.mean()
        v_targets_ = v_targets_.mean()

        # theta1: d(log_probs_act) / d_theta
        # theta2: d_v / d_theta
        theta1 = torch.autograd.grad(pi_, agent_.parameters(), retain_graph=True, allow_unused=True)
        theta2 = torch.autograd.grad(v_, agent_.parameters(), allow_unused=True)
        dg_dgamma = torch.autograd.grad(v_targets_, gamma_)

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

    advs = torch.Tensor(advs).detach()

    logits, v = agent(states)
    logits = logits.view(-1, action_dim)
    v = v.view(-1)
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    log_probs_act = log_probs.gather(1, actions).view(-1)

    loss_policy = - (advs * log_probs_act).mean()
    loss_critic = F.mse_loss(v_targets.detach(), v, reduction='mean')
    loss_entropy = - (log_probs * probs).mean()

    loss = loss_policy + .25 * loss_critic - .001 * loss_entropy
    optim.zero_grad()
    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP_GRAD_NORM)

    z = compute_z(agent, log_probs_act, v, v_targets, gamma, LR, .5)

    optim.step()
    scheduler.step()


    return z.clone().detach().requires_grad_(True)


def meta_grad(agent, states_dash, actions_dash, advs_dash, gamma, z_dash, action_dim):
    agent.zero_grad()
    states = torch.Tensor(np.array(states_dash))
    actions = torch.tensor(actions_dash, dtype=torch.int64).view(-1, 1)

    logits, _ = agent(states)
    logits = logits.view(-1, action_dim)
    log_probs = F.log_softmax(logits, dim=1)
    log_probs_act = log_probs.gather(1, actions).view(-1)

    # Compute Equation(14)
    advs_dash = torch.Tensor(advs_dash)
    pi = advs_dash.detach() * log_probs_act
    # J1: d(log_probs_act_dash) / d(theta_dash)
    J1 = torch.autograd.grad(pi.mean(), agent.parameters(), allow_unused=True)
    J1 = [torch.zeros_like(params.data) if item is None else item
          for (item, params) in zip(J1, agent.parameters())]
    J1 = [item.view(-1) for item in J1]
    J1 = torch.cat(J1)

    n_delta = - BETA * torch.dot(J1, z_dash)
    if torch.abs(n_delta) < 0.01:
        gamma.data += n_delta
    else:
        gamma.data += torch.sign(n_delta) * torch.tensor(0.01)

    # Limit gamma into [0, 1], bug might happen after this mechanism take effect
    if gamma.data > torch.tensor(0.999):
        gamma.data = torch.tensor(0.999)
    elif gamma.data < torch.tensor(0.001):
        gamma.data = torch.tensor(0.001)

    agent.zero_grad()

    return gamma.clone().detach().requires_grad_(True)


def test(gym_name, agent):
    result = 0
    test_task = gym.make(gym_name)
    for test_epi in range(10):
        state = test_task.reset()
        for test_step in range(500):
            act, _ = choose_action(agent, state)
            next_state, reward, done, _ = test_task.step(act.numpy())
            result += reward
            state = next_state
            if done:
                break
    return result


@torch.no_grad()
def choose_action(agent, state):
    logits, v = agent(torch.Tensor(state))
    act_probs = F.softmax(logits, dim=-1)
    m = Categorical(act_probs)
    act = m.sample()

    return act, v


if __name__ == '__main__':
    date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    total_test_results = []
    for random_seed in range(30):
        test_results = run(random_seed)
        total_test_results.append(test_results)

    dir = 'learning_results' + date + '.npy'
    np.save(dir, total_test_results)
