import torch
import torch.nn as nn
from .modules import Generator, Multiple_Random_Window_Discriminators

class GanTTs(nn.Module):
    def __init__(self,
                 lc_channels,
                 z_channels):
        super(GanTTs, self).__init__()
        
        self.lc_channels = lc_channels
        self.z_channels = z_channels
        
        self.generator = Generator(lc_channels, z_channels)
        self.discriminator = Multiple_Random_Window_Discriminators(lc_channels)
        
    def forward(self, inputs=None, conditions=None, z=None, generator=True):
        if generator:
            return self.forward_generator(conditions, z)
        else:
            return self.forward_discriminator(inputs, conditions)
    
    def forward_generator(self, conditions, z):
        outputs = self.generator(conditions, z)
        
        return outputs
    
    def forward_discriminator(self, inputs, conditions):
        outputs = self.discriminator(inputs, conditions)
        
        return outputs
