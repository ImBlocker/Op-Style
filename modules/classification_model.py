import torch
import torch.nn as nn



class ClassificationModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super(ClassificationModel, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits