import torch.nn as nn
from .modules import Conv1d
import numpy as np

class Multiple_Random_Window_Discriminators(nn.Module):
    def __init__(self,
                 lc_channels,
                 window_size=(2, 4, 8, 16, 30),
                 upsample_factor=120):

        super(Multiple_Random_Window_Discriminators, self).__init__()

        self.lc_channels = lc_channels
        self.window_size = window_size
        self.upsample_factor = upsample_factor

        self.udiscriminators = nn.ModuleList([
            UnConditionalDBlocks(in_channels=1, factors=(5, 3), out_channels=(128, 256)),
            UnConditionalDBlocks(in_channels=2, factors=(5, 3), out_channels=(128, 256)),
            UnConditionalDBlocks(in_channels=4, factors=(5, 3), out_channels=(128, 256)),
            UnConditionalDBlocks(in_channels=8, factors=(5, 3), out_channels=(128, 256)),
            UnConditionalDBlocks(in_channels=15, factors=(2, 2), out_channels=(128, 256)),
        ])

        self.discriminators = nn.ModuleList([
            ConditionalDBlocks(in_channels=1, lc_channels=lc_channels,
                               factors=(5, 3, 2, 2, 2), out_channels=(128, 128, 256, 256)),
            ConditionalDBlocks(in_channels=2, lc_channels=lc_channels,
                               factors=(5, 3, 2, 2), out_channels=(128, 256, 256)),
            ConditionalDBlocks(in_channels=4, lc_channels=lc_channels,
                               factors=(5, 3, 2), out_channels=(128, 256)),
            ConditionalDBlocks(in_channels=8, lc_channels=lc_channels,
                               factors=(5, 3), out_channels=(256,)),
            ConditionalDBlocks(in_channels=15, lc_channels=lc_channels,
                               factors=(2, 2, 2), out_channels=(128, 256)),
        ])

    def forward(self, real_samples, fake_samples, conditions):

        real_outputs, fake_outputs = [], []
        #unconditional discriminator
        for (size, layer) in zip(self.window_size, self.udiscriminators):
            size = size * self.upsample_factor
            index = np.random.randint(real_samples.size()[-1] - size)

            real_output = layer(real_samples[:, :, index : index + size])
            real_outputs.append(real_output)

            fake_output = layer(fake_samples[:, :, index : index + size])
            fake_outputs.append(fake_output)

        #conditional discriminator
        for (size, layer) in zip(self.window_size, self.discriminators):
            lc_index = np.random.randint(conditions.size()[-1] - size)
            sample_index = lc_index * self.upsample_factor
            real_x = real_samples[:, :, sample_index : (lc_index + size) * self.upsample_factor]
            fake_x = fake_samples[:, :, sample_index : (lc_index + size) * self.upsample_factor]
            lc = conditions[:, :, lc_index : lc_index + size]

            real_output = layer(real_x, lc)
            real_outputs.append(real_output)
            fake_output = layer(fake_x, lc)
            fake_outputs.append(fake_output)

        return real_outputs, fake_outputs

class CondDBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 lc_channels,
                 downsample_factor):
        super(CondDBlock, self).__init__()

        self.in_channels = in_channels
        self.lc_channels = lc_channels
        self.downsample_factor = downsample_factor

        self.start = nn.Sequential(
            nn.AvgPool1d(downsample_factor, stride=downsample_factor),
            nn.ReLU(),
            Conv1d(in_channels, in_channels * 2, kernel_size=3)
        )
        self.lc_conv1d = Conv1d(lc_channels, in_channels * 2, 1)
        self.end = nn.Sequential(
            nn.ReLU(),
            Conv1d(in_channels * 2, in_channels * 2, kernel_size=3, dilation=2)
        )
        self.residual = nn.Sequential(
            Conv1d(in_channels, in_channels * 2, kernel_size=1),
            nn.AvgPool1d(downsample_factor, stride=downsample_factor)
        )

    def forward(self, inputs, conditions):
        outputs = self.start(inputs) + self.lc_conv1d(conditions)
        outputs = self.end(outputs)
        residual_outputs = self.residual(inputs)
        outputs = outputs + residual_outputs

        return outputs

class DBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 downsample_factor):
        super(DBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_factor = downsample_factor

        self.layers = nn.Sequential(
            nn.AvgPool1d(downsample_factor, stride=downsample_factor),
            nn.ReLU(),
            Conv1d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(),
            Conv1d(out_channels, out_channels, kernel_size=3, dilation=2)
        )
        self.residual = nn.Sequential(
            Conv1d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool1d(downsample_factor, stride=downsample_factor)
        )

    def forward(self, inputs):
        outputs = self.layers(inputs) + self.residual(inputs)

        return outputs

class ConditionalDBlocks(nn.Module):
    def __init__(self,
                 in_channels,
                 lc_channels,
                 factors=(2, 2, 2),
                 out_channels=(128, 256)):
        super(ConditionalDBlocks, self).__init__()

        assert len(factors) == len(out_channels) + 1

        self.in_channels = in_channels
        self.lc_channels = lc_channels
        self.factors = factors
        self.out_channels = out_channels

        self.layers = nn.ModuleList()
        self.layers.append(DBlock(in_channels, 64, 1))
        in_channels = 64
        for (i, channel) in enumerate(out_channels):
            self.layers.append(DBlock(in_channels, channel, factors[i]))
            in_channels = channel

        self.cond_layer = CondDBlock(in_channels, lc_channels, factors[-1])

        self.post_process = nn.ModuleList([
            DBlock(in_channels * 2, in_channels * 2, 1),
            DBlock(in_channels * 2, in_channels * 2, 1)
        ])

    def forward(self, inputs, conditions):
        batch_size = inputs.size()[0]
        outputs = inputs.view(batch_size, self.in_channels, -1)
        for layer in self.layers:
            outputs = layer(outputs)
        outputs = self.cond_layer(outputs, conditions)
        for layer in self.post_process:
            outputs = layer(outputs)

        return outputs

class UnConditionalDBlocks(nn.Module):
    def __init__(self,
                 in_channels,
                 factors=(5, 3),
                 out_channels=(128, 256)):
        super(UnConditionalDBlocks, self).__init__()

        self.in_channels = in_channels
        self.factors = factors
        self.out_channels = out_channels

        self.layers = nn.ModuleList()
        self.layers.append(DBlock(in_channels, 64, 1))
        in_channels = 64
        for (i, factor) in enumerate(factors):
            self.layers.append(DBlock(in_channels, out_channels[i], factor))
            in_channels = out_channels[i]
        self.layers.append(DBlock(in_channels, in_channels, 1))
        self.layers.append(DBlock(in_channels, in_channels, 1))

    def forward(self, inputs):
        batch_size = inputs.size()[0]
        outputs = inputs.view(batch_size, self.in_channels, -1)
        for layer in self.layers:
            outputs = layer(outputs)

        return outputs

'''
model = Multiple_Random_Window_Discriminators(567)
#model = Generator(567, 128)

x = torch.randn(2, 1, 36000)
z = torch.randn(2, 128)
lc = torch.randn(2, 567, 300)
output = model(x, lc)
print(output.shape)
print(output)
'''
