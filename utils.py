
import torch
import torch.nn as nn

class NNSingleOut(nn.Module):
    def __init__(self, inChannel):
        super().__init__()
        self.inLayer = nn.Sequential(
            nn.Linear(inChannel, 64),
            nn.Tanh()
        )
        self.linLayer1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.outLayer = nn.Sequential(
            nn.Linear(64, 1),
        )
    def forward(self, x):
        x = self.inLayer(x.float())
        x = self.linLayer1(x)
        return self.outLayer(x)

class NormalDistribParam(nn.Module):
    def __init__(self, inChannel):
        super().__init__()
        self.inLayer = nn.Sequential(
            nn.Linear(inChannel, 64),
            nn.Tanh()
        )
        self.linLayer1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.muLayer = nn.Sequential(
            nn.Linear(64, 1),
        )
        self.sigmaLayer = nn.Sequential(
            nn.Linear(64, 1),
        )
    def forward(self, x):
        x = self.inLayer(x.float())
        x = self.linLayer1(x)
        return self.muLayer(x),torch.log(1+torch.exp(self.sigmaLayer(x))) 