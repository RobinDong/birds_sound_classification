import torch
import torch.nn as nn


class Augment(nn.Module):
    def __init__(self, training=True):
        super(Augment, self).__init__()
        self._training = training
        # self.dropout = nn.Dropout(p=0.2).cuda()
        # self._dropout = dropout

    def forward(self, image):
        image = (image - image.mean()) / image.std()
        if self._training:
            gaussian_noise = torch.randn(image.shape, device='cuda')
            image += gaussian_noise * (torch.rand(1, device='cuda') * 0.8 - 0.4)
        # if self._dropout:
        #    image = self.dropout(image)
        return image
