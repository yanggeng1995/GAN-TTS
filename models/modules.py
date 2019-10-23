import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np

class Generator(nn.Module):
    def __init__(self,
                 in_channels=567,
                 z_channels=128):
        super(Generator, self).__init__()
        
        self.in_channels = in_channels
        self.z_channels = z_channels
        
        self.preprocess =spectral_norm(nn.Conv1d(in_channels, 768,
                            kernel_size=3, padding=(3 - 1) // 2))
        self.gblocks = nn.ModuleList ([
            GBlock(768, 768, z_channels, 1),
            GBlock(768, 768, z_channels, 1),
            GBlock(768, 384, z_channels, 2),
            GBlock(384, 384, z_channels, 2),
            GBlock(384, 384, z_channels, 2),
            GBlock(384, 192, z_channels, 3),
            GBlock(192, 96, z_channels, 5)
        ])
        self.postprocess = nn.Sequential(
            spectral_norm(nn.Conv1d(96, 1, kernel_size=3, padding=(3 - 1) // 2)),
            nn.Tanh()
        )
        
    def forward(self, inputs, z):
        inputs = self.preprocess(inputs)
        outputs = inputs
        for (i, layer) in enumerate(self.gblocks):
            outputs = layer(outputs, z)
        outputs = self.postprocess(outputs)
        
        return outputs

class GBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 z_channels,
                 upsample_factor):
        super(GBlock, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.upsample_factor = upsample_factor
        
        self.condition_batchnorm1 = ConditionalBatchNorm1d(in_channels)
        self.linear1 = spectral_norm(nn.Linear(z_channels, in_channels))
        self.stack_first = nn.Sequential(
            nn.ReLU(inplace=False),
            UpsampleNet(in_channels, in_channels, upsample_factor),
            spectral_norm(nn.Conv1d(in_channels, hidden_channels,
                        kernel_size=3, padding=(3 - 1) // 2))
        )
        
        self.condition_batchnorm2 = ConditionalBatchNorm1d(hidden_channels)
        self.linear2 = spectral_norm(nn.Linear(z_channels, hidden_channels))
        self.second_stack = nn.Sequential(
            nn.ReLU(inplace=False),
            spectral_norm(nn.Conv1d(hidden_channels, hidden_channels,
                        kernel_size=3, dilation=2, padding=2 * (3 - 1) // 2))
        )
        
        self.residual1 = nn.Sequential(
           UpsampleNet(in_channels, in_channels, upsample_factor), 
           spectral_norm(nn.Conv1d(in_channels, hidden_channels, kernel_size=1))
        )
        
        self.condition_batchnorm3 = ConditionalBatchNorm1d(hidden_channels)
        self.linear3 = spectral_norm(nn.Linear(z_channels, hidden_channels))
        self.third_stack = nn.Sequential(
            nn.ReLU(inplace=False),
            spectral_norm(nn.Conv1d(hidden_channels, hidden_channels,
                        kernel_size=3, dilation=4, padding=4 * (3 - 1) // 2))
        )
        
        self.condition_batchnorm4 = ConditionalBatchNorm1d(hidden_channels)
        self.linear4 = spectral_norm(nn.Linear(z_channels, hidden_channels))
        self.fourth_stack = nn.Sequential(
            nn.ReLU(inplace=False),
            spectral_norm(nn.Conv1d(hidden_channels, hidden_channels,
                        kernel_size=3, dilation=8, padding=8 * (3 - 1) // 2))
        )
        
    def forward(self, condition, z):
        inputs = condition
        outputs = self.condition_batchnorm1(inputs, self.linear1(z))
        outputs = self.stack_first(outputs)
        outputs = self.condition_batchnorm2(outputs, self.linear2(z))
        
        residual_outputs = self.residual1(inputs)
        
        residual_outputs = outputs + residual_outputs
        
        outputs = self.condition_batchnorm3(residual_outputs, self.linear3(z))
        outputs = self.third_stack(outputs)
        outputs = self.condition_batchnorm4(outputs, self.linear4(z))
        outputs = self.fourth_stack(outputs)
        
        outputs = outputs + residual_outputs
        
        return outputs

class ConditionalBatchNorm1d(nn.BatchNorm1d):
    
    """Conditional Batch Normalization"""

    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=False,
                 track_running_stats=True):
        
        super(ConditionalBatchNorm1d, self).__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )
        self.scale = spectral_norm(nn.Linear(num_features, num_features))
        self.shift = spectral_norm(nn.Linear(num_features, num_features))

    def forward(self, input, condition):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)
        
        size = output.size()
        
        scale = self.scale(condition).unsqueeze(-1).expand(size)
        shift = self.shift(condition).unsqueeze(-1).expand(size)
        output =  scale * output + shift
        
        return output
    
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
        real_features, fake_features = [], []
        #unconditional discriminator
        for (size, layer) in zip(self.window_size, self.udiscriminators):
            size = size * self.upsample_factor
            index = np.random.randint(real_samples.size()[-1] - size)

            real_output, real_feature = layer(real_samples[:, :, index : index + size])
            real_outputs.append(real_output)
            real_features.extend(real_feature)

            fake_output, fake_feature = layer(fake_samples[:, :, index : index + size])
            fake_outputs.append(fake_output)
            fake_features.extend(fake_feature)

        #conditional discriminator
        for (size, layer) in zip(self.window_size, self.discriminators): 
            lc_index = np.random.randint(conditions.size()[-1] - size)
            sample_index = lc_index * self.upsample_factor
            real_x = real_samples[:, :, sample_index : (lc_index + size) * self.upsample_factor]
            fake_x = fake_samples[:, :, sample_index : (lc_index + size) * self.upsample_factor]
            lc = conditions[:, :, lc_index : lc_index + size]

            real_output, real_feature = layer(real_x, lc)
            real_outputs.append(real_output)
            real_features.extend(real_feature)
            fake_output, fake_feature = layer(fake_x, lc)
            fake_outputs.append(fake_output)
            fake_features.extend(fake_feature)
             
        return real_outputs, fake_outputs, real_features, fake_features 
 
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
            spectral_norm(nn.Conv1d(in_channels, in_channels * 2,
                                    kernel_size=3, padding=1))
        )
        self.lc_conv1d = spectral_norm(nn.Conv1d(lc_channels, in_channels * 2, 1))
        self.end = nn.Sequential(
            nn.ReLU(),
            spectral_norm(nn.Conv1d(in_channels * 2, in_channels * 2, kernel_size=3,
                                    dilation=2, padding=2 * (3 - 1) // 2))            
        )
        self.residual = nn.Sequential(
            spectral_norm(nn.Conv1d(in_channels, in_channels * 2, kernel_size=1)),
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
            spectral_norm(nn.Conv1d(in_channels, out_channels,
                                    kernel_size=3, padding=1)),
            nn.ReLU(),
            spectral_norm(nn.Conv1d(out_channels, out_channels, kernel_size=3,
                                    dilation=2, padding=2 * (3 - 1) // 2))
        )
        self.residual = nn.Sequential(
            spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size=1)),
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
            DBlock(in_channels * 2, in_channels * 2, 1),
            DBlock(in_channels * 2, 1, 1)
        ])
        
    def forward(self, inputs, conditions):
        batch_size = inputs.size()[0]
        outputs = inputs.view(batch_size, self.in_channels, -1)
        lists = []
        for layer in self.layers:
            outputs = layer(outputs)
            lists.append(outputs)
        outputs = self.cond_layer(outputs, conditions)
        lists.append(outputs)
        for layer in self.post_process:
            outputs = layer(outputs)
            lists.append(outputs)

        return lists[-1], lists[:-1]
    
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
        self.layers.append(DBlock(in_channels, 1, 1))
        
    def forward(self, inputs):
        batch_size = inputs.size()[0]
        outputs = inputs.view(batch_size, self.in_channels, -1)
        lists = []
        for layer in self.layers:
            outputs = layer(outputs)
            lists.append(outputs)

        return lists[-1], lists[:-1]

class UpsampleNet(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 upsample_factor):

        super(UpsampleNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.upsample_factor = upsample_factor

        self.layer = spectral_norm(nn.ConvTranspose1d(input_size, output_size,
                 upsample_factor * 2, upsample_factor, padding=upsample_factor // 2))

    def forward(self, inputs):
        outputs = self.layer(inputs)
        outputs = outputs[:, :, : inputs.size(-1) * self.upsample_factor]

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
