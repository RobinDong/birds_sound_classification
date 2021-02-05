import torch.nn as nn


class Augment(nn.Module):
    def __init__(self, dropout=True):
        super(Augment, self).__init__()
        self.dropout = nn.Dropout(p=0.3).cuda()
        self._dropout = True

    def forward(self, image):
        image = (image - image.mean()) / image.std()
        if self._dropout:
            image = self.dropout(image)
        return image
