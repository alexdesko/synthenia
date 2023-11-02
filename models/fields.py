import torch
import torch.nn as nn


class SmallDensity(nn.Module):
    def __init__(self, input_dim, feature_dim, t_features_dim = 0):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.t_features_dim = t_features_dim

        # Takes as input only the positions
        self.block = nn.Sequential(
            nn.Linear(self.input_dim + self.t_features_dim, self.feature_dim),
            nn.Softplus(beta=100),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Softplus(beta=100),
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)


class MediumDensity(nn.Module):
    def __init__(self, input_dim, feature_dim, t_features_dim = 0):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.t_features_dim = t_features_dim

        # Takes as input only the positions
        self.block1 = nn.Sequential(
            nn.Linear(self.input_dim + self.t_features_dim, self.feature_dim),
            nn.Softplus(beta=100),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Softplus(beta=100),
        )
        # Skip connection
        self.block2 = nn.Sequential(
            nn.Linear(self.input_dim + self.t_features_dim + self.feature_dim, self.feature_dim),
            nn.Softplus(beta=100),
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.block1(x)
        output = torch.cat([output, x], dim = -1)
        return self.block2(output)
    
class LargeDensity(nn.Module):
    def __init__(self, input_dim, feature_dim, t_features_dim = 0):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.t_features_dim = t_features_dim

        # Takes as input only the positions
        self.block1 = nn.Sequential(
            nn.Linear(self.input_dim + self.t_features_dim, self.feature_dim),
            nn.Softplus(beta=100),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Softplus(beta=100),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Softplus(beta=100),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Softplus(beta=100)
        )
        # Skip connection
        self.block2 = nn.Sequential(
            nn.Linear(self.input_dim + self.feature_dim + self.t_features_dim, self.feature_dim),
            nn.Softplus(beta=100),
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.block1(x)
        output = torch.cat([output, x], dim = -1)
        return self.block2(output)
