"""Readout heads: map molecule embeddings [batch, in_dim] to predictions [batch, n_targets].

To add a head, write a class taking (in_dim, n_targets, **options), register it in
READOUTS and add conf/readout/<name>.yaml. Use it with e.g. readout=<name>."""
from torch import nn


class LinearReadout(nn.Module):
    def __init__(self, in_dim, n_targets):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_targets)

    def forward(self, x):
        return self.linear(x)


class MLPReadout(nn.Module):
    """num_layers Linear layers (including the output layer) with ReLU and dropout."""

    def __init__(self, in_dim, # input dimension af molecule embedding
                 n_targets, # output dimension - altså antal targets vi skal forudsige
                 hidden_dim=64,
                 num_layers=2,
                 dropout=0.6):
        super().__init__()

        #laver en liste med dimensioner for hver layer, hvor hidden_dim gentages num_layers-1 gange.
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) #fx input=10, hidden_dim=20, num_layers=3 => dims = [10, 20, 20]
        layers = []
        
        # for loop der går igennem par af dimensioner (a, b) fra dims listen, og tilføjer aktiveringsfunktioner og dropout til layers listen:
        for (a, b) in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b),
                       nn.ReLU(),
                       nn.Dropout(dropout)]

        #samler hidden layers og output:
        self.mlp = nn.Sequential(*layers, #de hidden layers
                                 nn.Linear(dims[-1], n_targets)) #output layer uden aktivering og dropout

    def forward(self, x):
        return self.mlp(x) #kører input x igennem mlp'en og returnerer outputtet


READOUTS = {'linear': LinearReadout, 'mlp': MLPReadout}


def build_readout(cfg, in_dim, n_targets):
    options = {k: v for k, v in cfg.items() if k != 'type'}
    return READOUTS[cfg.type](in_dim, n_targets, **options)
