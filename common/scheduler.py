"""
A wrapper class for scheduled optimizer (Adam optimizer with warmup and lr decay).

Source: https://github.com/jadore801120/attention-is-all-you-need-pytorch

"""

import numpy as np

class CosineWarmup():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, n_warmup_steps, max_steps):
        self._optimizer = optimizer
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
        self.max_steps = max_steps

    def get_lr(self):
        return self._optimizer.param_groups[0]['lr']


    def _get_lr_factor(self):
        lr_factor = 0.5 * (1 + np.cos(np.pi * self.n_steps / self.max_steps))
        if self.n_steps <= self.n_warmup_steps:
            lr_factor *= self.n_steps * 1.0 / self.n_warmup_steps
        return lr_factor


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.initial_lr * self._get_lr_factor()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        optim_state = self._optimizer.state_dict()
        sched_state = {"n_steps": self.n_steps, "max_steps": self.max_steps, "n_warmup_steps": self.n_warmup_steps, "initial_lr": self.initial_lr}

        return {"optimizer": optim_state, "lr_sched": sched_state}
    
    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict['optimizer'])
        self.n_steps = state_dict['lr_sched']['n_steps']
        self.max_steps = state_dict['lr_sched']['max_steps']
        self.n_warmup_steps = state_dict['lr_sched']['n_warmup_steps']
        self.initial_lr = state_dict['lr_sched']['initial_lr']