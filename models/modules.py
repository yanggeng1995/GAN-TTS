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
                 lc_channels):
        super(Multiple_Random_Window_Discriminators, self).__init__()
        
        self.lc_channels = lc_channels

        self.udiscriminator1 = UnConditionalDBlocks(in_channels=1, factors=(5, 3),
                                            out_channels=(128, 256))
        self.udiscriminator2 = UnConditionalDBlocks(in_channels=2, factors=(5, 3), 
                                            out_channels=(128, 256))
        self.udiscriminator4 = UnConditionalDBlocks(in_channels=4, factors=(5, 3), 
                                            out_channels=(128, 256))
        self.udiscriminator8 = UnConditionalDBlocks(in_channels=8, factors=(5, 3), 
                                            out_channels=(128, 256))
        self.udiscriminator15 = UnConditionalDBlocks(in_channels=15, factors=(2, 2),
                                            out_channels=(128, 256))
        
        self.discriminator1 = ConditionalDBlocks(in_channels=1, lc_channels=lc_channels,
                            factors=(5, 3, 2, 2, 2), out_channels=(128, 128, 256, 256))
        self.discriminator2 = ConditionalDBlocks(in_channels=2, lc_channels=lc_channels,
                            factors=(5, 3, 2, 2), out_channels=(128, 256, 256))
        self.discriminator4 = ConditionalDBlocks(in_channels=4, lc_channels=lc_channels,
                            factors=(5, 3, 2), out_channels=(128, 256))
        self.discriminator8 = ConditionalDBlocks(in_channels=8, lc_channels=lc_channels,
                            factors=(5, 3), out_channels=(256,))
        self.discriminator15 = ConditionalDBlocks(in_channels=15, lc_channels=lc_channels,
                            factors=(2, 2, 2), out_channels=(128, 256))
        
    def forward(self, inputs, conditions):
        
        outputs = []
        #unconditional discriminator
        index = np.random.randint(inputs.size()[-1] - 240)
        output = self.udiscriminator1(inputs[:, :, index : index + 240])
        outputs.append(output)
        index = np.random.randint(inputs.size()[-1] - 480)
        output = self.udiscriminator2(inputs[:, :, index : index + 480])
        outputs.append(output)
        index = np.random.randint(inputs.size()[-1] - 960)
        output = self.udiscriminator4(inputs[:, :, index : index + 960])
        outputs.append(output)
        index = np.random.randint(inputs.size()[-1] - 1920)
        output = self.udiscriminator8(inputs[:, :, index : index + 1920])
        outputs.append(output)
        index = np.random.randint(inputs.size()[-1] - 3600)
        output = self.udiscriminator15(inputs[:, :, index : index + 3600])
        outputs.append(output)
        
        #conditional discriminator
        lc_index = np.random.randint(conditions.size()[-1] - 2)
        x = inputs[:, :, lc_index * 120 : (lc_index + 2) * 120]
        lc = conditions[:, :, lc_index : lc_index + 2]
        output = self.discriminator1(x, lc)
        outputs.append(output)
        lc_index = np.random.randint(conditions.size()[-1] - 4)
        x = inputs[:, :, lc_index * 120 : (lc_index + 4) * 120]
        lc = conditions[:, :, lc_index : lc_index + 4]
        output = self.discriminator2(x, lc)
        outputs.append(output)
        lc_index = np.random.randint(conditions.size()[-1] - 8)
        x = inputs[:, :, lc_index * 120 : (lc_index + 8) * 120]
        lc = conditions[:, :, lc_index : lc_index + 8]
        output = self.discriminator4(x, lc)
        outputs.append(output)
        lc_index = np.random.randint(conditions.size()[-1] - 16)
        x = inputs[:, :, lc_index * 120 : (lc_index + 16) * 120]
        lc = conditions[:, :, lc_index : lc_index + 16]
        output = self.discriminator8(x, lc)
        outputs.append(output)
        lc_index = np.random.randint(conditions.size()[-1] - 30)
        x = inputs[:, :, lc_index * 120 : (lc_index + 30) * 120]
        lc = conditions[:, :, lc_index : lc_index + 30]
        output = self.discriminator15(x, lc)
        outputs.append(output)
        
        outputs = sum(outputs) / len(outputs)
        
        return outputs
 
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
        
        lists = []
        lists.append(DBlock(in_channels, 64, 1))
        in_channels = 64
        for (i, channel) in enumerate(out_channels):
            lists.append(DBlock(in_channels, channel, factors[i]))
            in_channels = channel
        self.layers = nn.Sequential(*lists)
        
        self.cond_layer = CondDBlock(in_channels, lc_channels, factors[-1])
        
        self.end = nn.Sequential(
            DBlock(in_channels * 2, in_channels * 2, 1),
            DBlock(in_channels * 2, in_channels * 2, 1),
            DBlock(in_channels * 2, 1, 1),
        )
        
    def forward(self, inputs, conditions):
        batch_size = inputs.size()[0]
        inputs = inputs.view(batch_size, self.in_channels, -1)
        outputs = self.layers(inputs)
        outputs = self.cond_layer(outputs, conditions)
        outputs = self.end(outputs)
        
        outputs = outputs.mean()
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
        
        lists = []
        lists.append(DBlock(in_channels, 64, 1))
        in_channels = 64
        for (i, factor) in enumerate(factors):
            lists.append(DBlock(in_channels, out_channels[i], factor))
            in_channels = out_channels[i]
        lists.append(DBlock(in_channels, in_channels, 1))
        lists.append(DBlock(in_channels, in_channels, 1))
        lists.append(DBlock(in_channels, 1, 1))
        
        self.layers = nn.Sequential(*lists)
        
    def forward(self, inputs):
        batch_size = inputs.size()[0]
        inputs = inputs.view(batch_size, self.in_channels, -1)
        outputs = self.layers(inputs)
        outputs = outputs.mean()
        
        return outputs

class UpsampleNet(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 upsample_factor,
                 upsample_method="duplicate"):

        super(UpsampleNet, self).__init__()
        self.upsample_method = upsample_method
        self.upsample_factor = upsample_factor
        
        if upsample_method == 'duplicate':
            self.upsample_factor = int(np.prod(upsample_factor))
        elif upsample_method == 'transposed_conv2d':
            assert isinstance(upsample_factor, tuple)
            kernel_size = 3
            self.upsamples = nn.ModuleList()
            for u in upsample_factor:
                padding = (kernel_size - 1) // 2
                conv = spectral_norm(nn.ConvTranspose2d(1, 1, (kernel_size, 2 * u),
                                          padding=(padding, u // 2),
                                          dilation=1, stride=(1, u)))
                self.upsamples.append(conv)

    def forward(self, inputs):
        if self.upsample_method == 'duplicate':
            output = F.interpolate(inputs, scale_factor=self.upsample_factor,
                                   mode='nearest')
        elif self.upsample_method == 'transposed_conv2d':
            output = input.unsqueeze(1)
            for layer in self.upsamples:
                output = layer(output)
            output = output.squeeze(1)
            output = output[:, :, : input.size(-1) * np.prod(self.upsample_factor)]

        return output


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
