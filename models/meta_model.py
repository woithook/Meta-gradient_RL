"""
Custom network models that allow grad pass through network parameter.
source: https://github.com/danieltan07/learning-to-reweight-examples

How to use:
1. Build network models (refer to ActorCritic)
2. Initiate optimizer for the models (e.g.: optim.Adam(model.params())), noting that using .params() rather than
    .parameter()
3. Calculate loss function and obtain grad by torch.autograd.grad()
4. Initialize a dummy network meta-model for meta-learning and update its parameters by
    meta-model.update_params(lr, source_params=grads)
5. Do meta-learning and optimizer.step()
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import itertools


def to_var(x, requires_grad=True):
    # if torch.cuda.is_available():
    #     x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class ActorCritic(MetaModule):
    def __init__(self, l_obs, n_act, hidden_size=32):
        super(ActorCritic, self).__init__()
        self.l_obs = l_obs
        self.n_act = n_act
        self.hidden_size = hidden_size

        self.actor = nn.Sequential(
            MetaLinear(self.l_obs, self.hidden_size),
            nn.ReLU(inplace=True),
            MetaLinear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            MetaLinear(self.hidden_size, self.n_act)
        )
        self.critic = nn.Sequential(
            MetaLinear(self.l_obs, self.hidden_size),
            nn.ReLU(inplace=True),
            MetaLinear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            MetaLinear(self.hidden_size, 1)
        )

    def forward(self, inputs):
        pi = self.actor(inputs)
        vi = self.critic(inputs)

        return pi, vi
