import torch
import torch.nn as nn

class Augment(nn.Module):
    def __init__(self):
        super(Augment, self).__init__()
        self.dropout = nn.Dropout(p=0.1).cuda()

    def forward(self, image):
        image = self.dropout(image)
        return image
