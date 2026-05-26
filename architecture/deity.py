from architecture.based_model import BaseArchitecture
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import timm

class DeiTArchitecture(BaseArchitecture):
    SUPPORTED_MODELS = [
        'deit_tiny_patch16_224',
        'deit_small_patch16_224',
        'deit_base_patch16_224',
        'deit_tiny_distilled_patch16_224',
        'deit_small_distilled_patch16_224',
        'deit_base_distilled_patch16_224'
    ]

    def __init__(self, model_name='deit_small_patch16_224', num_classes=7, pretrained=True, device="cuda"):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"model '{model_name}' is not supported. choose from: {self.SUPPORTED_MODELS}")

        # featurizer
        featurizer = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        embed_dim = featurizer.num_features
        
        # classifier head
        classifier_head = nn.Linear(embed_dim, num_classes)
        
        # initialize the parent class with both components
        super().__init__(featurizer=featurizer, classifier_head=classifier_head, device=device)
        
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.num_classes = num_classes
