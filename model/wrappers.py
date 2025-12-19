import torch.utils.data
import torch
from transformers import PreTrainedModel
import torch.nn as nn
from abc import ABC, abstractmethod

class Wrapper_pretrain(nn.Module, ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode(self, x):
        pass
    
    @abstractmethod
    def denoising(self, x, emb = None, condition = None):
        pass
    
    @abstractmethod
    def text_embedding(self, x):
        pass
    
    @abstractmethod
    def forward(self, x, context):
        pass

class Wrapper_DELTA(nn.Module, ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def forward(self):
        pass