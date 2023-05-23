from torch import nn
import torch

from model.CodeVectorizer import CodeVectorizer
from model.randwidth_ops import RWLinear,make_divisible


class ProjectClassifier(nn.Module):

    def __init__(self, n_tokens, n_paths, dim, n_classes):
        super(ProjectClassifier, self).__init__()
        self.vectorization = CodeVectorizer(n_tokens, n_paths, dim)        
        self.transform = nn.Sequential(RWLinear(dim, dim,us=[False, True]), nn.Tanh())
        self.classifier = RWLinear(dim, n_classes, us=[True, False])
        
    def forward(self, contexts):
        new_dim = int((self.classifier.width_mult*256)+0.5)
        vectorized_contexts_ = self.vectorization(contexts)
        vectorized_contexts=vectorized_contexts_[:,0:new_dim]
        predictions = self.classifier(vectorized_contexts)
        return predictions

    def get_matrix(self,context):
        return self.vectorization(context)
    def predictions(self,vectorization_con):
        return self.classifier(vectorization_con)
