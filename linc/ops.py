import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableBiasVector(nn.Module):
    def __init__(self, in_features):
        super(LearnableBiasVector, self).__init__()
        self.bias = nn.Parameter(torch.randn(1, in_features) * 0.01, requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class NeuralNetworkUnit(nn.Module):
    def __init__(self, in_features, k, T):
        super(NeuralNetworkUnit, self).__init__()
        self.register_buffer("T", torch.ones(1) * T)
        self.alpha = nn.Parameter(torch.randn(1, in_features) * 0.001)
        self.k = k
        self.bias = LearnableBiasVector(in_features)
        self.hidden_dim = in_features // 2

    @torch.no_grad()
    def get_mask(self):
        w = F.softmax(self.alpha / self.T, dim=1).flatten()
        zeros_mask = torch.zeros_like(w)
        index = torch.argsort(w)
        index_m = index[len(index) - self.k :]
        zeros_mask[index_m] = w[index_m]
        return zeros_mask.unsqueeze(0)

    def forward(self, x):
        # implement the Eq. 1~2
        replace = x * F.softmax(self.alpha / self.T, dim=1).expand_as(x)
        out = x * self.get_mask().expand_as(x)
        z_hat = self.bias((out - replace).detach() + replace)
        return z_hat


class EnsembleNeuralNetworkUnits(nn.Module):
    """$E$ Neural Network Units for one class."""

    def __init__(self, in_features, k, T, E):
        super(EnsembleNeuralNetworkUnits, self).__init__()
        self.nnus = nn.ModuleList(
            [
                nn.Sequential(
                    NeuralNetworkUnit(in_features, k, T),
                    nn.Linear(in_features, in_features // 2),
                    nn.BatchNorm1d(in_features // 2),
                    nn.PReLU(),
                    nn.Linear(in_features // 2, 1, True),
                )
                for _ in range(E)
            ]
        )

    def forward(self, x):
        scores = []
        for nnu in self.nnus:
            scores.append(nnu(x))
        scores = torch.concat(scores, dim=-1)
        if self.training:
            # a category score is randomly sampled from the E candidates
            index = torch.multinomial(F.softmax(scores, dim=-1), 1)
        else:
            # directly choose the max score when testing
            index = torch.argmax(scores, dim=-1).view(-1, 1)
        return torch.gather(scores, 1, index)


if __name__ == "__main__":
    in_features = 10
    k = 4
    T = 1.0
    E = 8
    nnu = NeuralNetworkUnit(in_features, k, T)
    ensemble_nnus = EnsembleNeuralNetworkUnits(in_features, k, T, E)

    inputs = torch.randn(32, in_features)
    outputs = nnu(inputs)
    assert outputs.shape == (32, 1)

    ensemble_outputs = ensemble_nnus(inputs)
    assert ensemble_outputs.shape == (32, 1)
