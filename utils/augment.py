import torch
import torch.nn as nn

HEIGHT = 128
WIDTH = 312


def generate_grid(h, w):
    x = torch.arange(0, h, dtype=torch.int)
    y = torch.arange(0, w, dtype=torch.int)
    grid = torch.stack([x.repeat(w), y.repeat(h, 1).t().contiguous().view(-1)], 1)
    return grid


class Augment(nn.Module):
    def __init__(self, training=True):
        super(Augment, self).__init__()
        self._training = training
        self._dropout = nn.Dropout(p=0.1).cuda()

    def _image_part(self, split, coordinates, index):
        coord = coordinates[index]
        hpart = WIDTH // split
        wpart = HEIGHT // split
        return coord[0] * wpart, (coord[0] + 1) * wpart, coord[1] * hpart, (coord[1] + 1) * hpart

    def puzzle(self, image):
        raw_image = image.clone()
        split = 2
        coordinates = generate_grid(split, split)
        for index, offset in enumerate(torch.randperm(split*split)):
            nx1, nx2, ny1, ny2 = self._image_part(split, coordinates, index)
            ox1, ox2, oy1, oy2 = self._image_part(split, coordinates, offset)
            image[:, :, nx1:nx2, ny1:ny2] = raw_image[:, :, ox1:ox2, oy1:oy2]
        return image

    def forward(self, image):
        image = (image - image.mean()) / image.std()
        # Don't need high pitch
        '''if self._training:
            gaussian_noise = torch.randn(image.shape, device='cuda')
            image += gaussian_noise * (torch.rand(1, device='cuda') * 0.002 - 0.001)'''
        #if self._training:
        #    image = self._dropout(image)
        #image = self.puzzle(image)
        return image
