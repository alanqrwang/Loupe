import torch

class STIdentity(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, x):
        return torch.bernoulli(x)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class STSigmoid(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, x, epoch, tot_epochs):
        output = torch.bernoulli(x)
        ctx.epoch = epoch
        ctx.tot_epochs = tot_epochs
        ctx.save_for_backward(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        steps = 10
        output, = ctx.saved_variables
        l = torch.linspace(1, 100, steps=steps)
        factor = l[int(ctx.epoch / ctx.tot_epochs) * steps]
        print('factor: ', factor)
        return grad_output * factor * (1. - output) * output, None, None