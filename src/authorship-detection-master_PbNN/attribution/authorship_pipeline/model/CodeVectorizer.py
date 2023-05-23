import torch
from torch import nn
from torch.nn import functional as F
from model.randwidth_ops import RWLinear

# Implementation of code2vec's vectorization part in PyTorch.
# Since it is a PyTorch Module, it can be reused as a part of another pipeline.

class CodeVectorizer(nn.Module):

    def __init__(self, n_tokens, n_paths, dim):
        super(CodeVectorizer, self).__init__()
        self.tokens_embed = nn.Embedding(n_tokens, dim)
        self.paths_embed = nn.Embedding(n_paths, dim)
        self.dropout = nn.Dropout(p=0.2)
        self.transform = nn.Sequential(nn.Linear(3 * dim, dim,False), nn.Tanh())    
        self.attention = nn.Linear(dim, 1) 


    def forward(self, contexts,new_dim=128):
        starts, paths, ends = contexts
        starts = self.tokens_embed(starts)
        paths = self.paths_embed(paths)
        ends = self.tokens_embed(ends)
        concatenated_contexts = torch.cat((starts, paths, ends), dim=2)
        concatenated_contexts = self.dropout(concatenated_contexts)
        transformed_contexts = self.transform(concatenated_contexts)
        context_attentions = F.softmax(self.attention(transformed_contexts), dim=1)
        aggregated_context = torch.sum(torch.mul(transformed_contexts, context_attentions), dim=1)
        return aggregated_context