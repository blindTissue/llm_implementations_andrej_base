import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicAttention(nn.Module):
    def __init__(self, hidden_dim, positional_embedding, dropout=.1):
        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.possitional_embedding_type = positional_embedding['type']
    
    def forward(self, x):
        if self.positional_embedding_type == ''