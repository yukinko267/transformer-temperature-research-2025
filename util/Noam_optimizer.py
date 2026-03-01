
# Warm up

class NoamOpt:
    """Optim wrapper that implements Noam LR schedule (Transformer)"""

    def __init__(self, d_model, warmup_step, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup_step
        self.d_model = d_model

    def step(self):
        """Update parameters and learning rate"""
        self._step += 1
        lr = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        """Compute learning rate at current step"""
        if step is None:
            step = self._step
        return (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup ** -1.5)
        )
