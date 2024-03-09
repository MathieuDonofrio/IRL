import torch
import torch.nn as nn

class MLPRewardBlock(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 activation_layer = nn.ReLU, 
                 norm_layer = nn.BatchNorm1d, 
                 residual=True):
        super().__init__()

        if residual and input_dim != output_dim:
            raise ValueError("Residual connection is only supported when input_dim == output_dim")

        self.skip = residual
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = norm_layer(output_dim)
        self.activation = activation_layer()

    def forward(self, x):
        residual = x
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.activation(x)
        if self.skip:
            x = x + residual
        return x
    
class MLRReward(nn.Sequential):
    def __init__(self, 
                 input_dim, 
                 hidden_sizes, 
                 activation_layer=nn.ReLU, 
                 norm_layer=nn.BatchNorm1d, 
                 residual=True,
                 clamp_magnitude=10.0):
        
        blocks = [MLPRewardBlock(input_dim, hidden_sizes[0], activation_layer, norm_layer, False)]
        for i in range(1, len(hidden_sizes)):
            blocks.append(MLPRewardBlock(hidden_sizes[i-1], hidden_sizes[i], activation_layer, norm_layer, residual))
        blocks.append(nn.Linear(hidden_sizes[-1], 1))

        super().__init__(*blocks)

        self.clamp_magnitude = clamp_magnitude

    def forward(self, x):
        x = super().forward(x)
        x = torch.clamp(x, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return x      


