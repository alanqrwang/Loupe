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

# Dynamically changing sample slope across training
class STSigmoidAnneal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, epoch, tot_epochs, sample_slope):
        output = torch.bernoulli(x)
        ctx.epoch = epoch
        ctx.tot_epochs = tot_epochs
        ctx.sample_slope = sample_slope
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        steps = 100
        forward_input, = ctx.saved_variables
        l = torch.linspace(1, ctx.sample_slope, steps=steps)
        factor = l[int(ctx.epoch / ctx.tot_epochs * steps)]
        print('factor: ', factor.item())
        return grad_output * factor * (1. - STSigmoid.sigmoid(forward_input, factor)) * STSigmoid.sigmoid(forward_input, factor), None, None, None

    @staticmethod
    def sigmoid(x, slope):
        return 1 / (1 + torch.exp(-slope*x))

# Fixed sample slope across training
class STSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sample_slope):
        output = torch.bernoulli(x)
        ctx.sample_slope = sample_slope
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        forward_input, = ctx.saved_variables
        factor = ctx.sample_slope
        return grad_output * factor * (1. - STSigmoid.sigmoid(forward_input, factor)) * STSigmoid.sigmoid(forward_input, factor), None, None, None

    @staticmethod
    def sigmoid(x, slope):
        return 1 / (1 + torch.exp(-slope*x))
