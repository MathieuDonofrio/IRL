import math
import torch
import torch.nn as nn

class FrequencyEncoding(nn.Module):
    def __init__(self, latent_size, device=None):
        super().__init__()

        idx = torch.arange(latent_size, dtype=torch.float32)
        self.div_term = torch.exp(idx * (math.log(10000.0) / latent_size)).to(device)

    def forward(self, x):
        x = torch.unsqueeze(x, -1)
        return torch.cat((torch.sin(x * self.div_term), torch.cos(x * self.div_term)), dim=-1)

        
class TransformerReward(nn.Module):
    def __init__(self,
                 num_tokens,
                 latent_size = 32, 
                 num_layers=4, 
                 num_heads=4, 
                 hidden_dim=256, 
                 clamp_magnitude=10.0,
                 dropout=0.1,
                 device=None):
        super(TransformerReward, self).__init__()

        self.clamp_magnitude = clamp_magnitude

        self.freq_encoder = FrequencyEncoding(latent_size, device=device)
        self.embedding = nn.Embedding(num_tokens, latent_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=dropout,
            activation=nn.GELU(approximate="tanh"),
            norm_first=True,
            device=device)
        
        encoder_norm = nn.LayerNorm(latent_size, device=device)
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=encoder_norm)

        self.linear = nn.Linear(latent_size, 1)

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, tokens):
        x = self.embedding(tokens) + self.freq_encoder(x)
        x = self.transformer_encoder(x)
        x = self.linear(x)
        x = torch.clamp(x, -self.clamp_magnitude, self.clamp_magnitude)
        return x
        
        

        
      
        


        