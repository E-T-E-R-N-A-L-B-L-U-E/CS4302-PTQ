import torch
import torch.nn as nn

def uniform_quantize(k):
    class qfn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2 ** k - 1)
                out = torch.round(input * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input

    return qfn().apply


class WeightQuantizeFn(nn.Module):
    def __init__(self, w_bit):
        super(WeightQuantizeFn, self).__init__()
        assert w_bit <= 16 or w_bit == 32
        self.w_bit = w_bit
        self.uniform_q = uniform_quantize(k=w_bit)

    def forward(self, x):
        if self.w_bit == 32:
            weight_q = x
        elif self.w_bit == 1:
            E = torch.mean(torch.abs(x)).detach()
            weight_q = self.uniform_q(x / E) * E
        else:
            weight = torch.tanh(x)
            max_w = torch.max(torch.abs(weight)).detach()
            weight = weight / 2 / max_w + 0.5
            weight_q = max_w * (2 * self.uniform_q(weight) - 1)
        return weight_q


class ActivationQuantizeFn(nn.Module):
    def __init__(self, a_bit):
        super(ActivationQuantizeFn, self).__init__()
        assert a_bit <= 16 or a_bit == 32
        self.a_bit = a_bit
        self.uniform_q = uniform_quantize(k=a_bit)

    def forward(self, x):
        if self.a_bit == 32:
            activation_q = x
        else:
            activation_q = self.uniform_q(torch.clamp(x, 0, 1))
        return activation_q


def quantize_weights(weights, num_bits):
    quantize_fn = WeightQuantizeFn(num_bits)
    quantized_weights = quantize_fn(weights)
    return quantized_weights


def quantize_activations(activations, num_bits):
    quantize_fn = ActivationQuantizeFn(num_bits)
    quantized_activations = quantize_fn(activations)
    return quantized_activations