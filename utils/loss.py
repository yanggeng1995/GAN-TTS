from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn

class HingeLoss(nn.Module):
    def __init__(self, margin=1.0, size_average=True, sign=1.0):
        super(HingeLoss, self).__init__()

        self.sign = sign
        self.margin = margin
        self.size_average = size_average
 
    def forward(self, input, target):
        assert input.dim() == target.dim()
        output = self.margin - torch.mul(target, input)

        if 'cuda' in input.data.type():
            mask = torch.cuda.FloatTensor(input.size()).zero_()
        else:
            mask = torch.FloatTensor(input.size()).zero_()
        mask[torch.gt(output, 0.0)] = 1.0

        output = torch.mul(output, mask)
        # size average
        if self.size_average:
            output = torch.mul(output, 1.0 / input.nelement())
        output = output.sum()
        # apply sign
        output = torch.mul(output, self.sign)
        return output
