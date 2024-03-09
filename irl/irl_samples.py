import numpy as np
import torch
import torch.nn as nn
import gymnasium
import tianshou as ts

from models.mlp_reward import MLRReward

if __name__ == "__main__":
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Numpy version: {np.__version__}")
    print(f"Gymnasium version: {gymnasium.__version__}")
    print(f"Tianshou version: {ts.__version__}")

    print(f"GPU available: {torch.cuda.is_available()}")
