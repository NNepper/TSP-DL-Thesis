"""
A wrapper class for scheduled optimizer (Adam optimizer with warmup and lr decay).

Source: https://github.com/jadore801120/attention-is-all-you-need-pytorch

"""

import numpy as np

class NOAM():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps, lr_mul=1.0):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def get_lr(self):
        return self._optimizer.param_groups[0]['lr']


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        optim_state = self._optimizer.state_dict()
        sched_state = {"n_steps": self.n_steps}

        return {"optimizer": optim_state, "lr_sched": sched_state}
    
    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict['optimizer'])
        self.n_steps = state_dict['lr_sched']['n_steps']
        self._update_learning_rate()