import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self,
                 in_channels=567,
                 z_channels=128):
        super(Encoder, self).__init__()
        
        self.in_channels = in_channels
        self.z_channels = z_channels
        
        self.conv1d_start = nn.Conv1d(in_channels, 768, kernel_size=3,
                                      padding=(3 - 1) // 2)
        self.gblocks = nn.ModuleList ([
            GBlock(768, 768, z_channels, 1),
            GBlock(768, 768, z_channels, 1),
            GBlock(768, 384, z_channels, 2),
            GBlock(384, 384, z_channels, 2),
            GBlock(384, 384, z_channels, 2),
            GBlock(384, 192, z_channels, 3),
            GBlock(192, 96, z_channels, 5)
        ])
        self.conv1d_end = nn.Conv1d(96, 1, kernel_size=3, padding=(3 - 1) // 2)
        
    def forward(self, inputs, z):
        outputs = self.conv1d_start(inputs)
        for layer in self.gblocks:
            outputs = layer(outputs, z)
        outputs = self.conv1d_end(outputs)
        
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
        self.linear1 = nn.Linear(z_channels, in_channels)
        self.stack_first = nn.Sequential(
            nn.ReLU(inplace=False),
            UpsampleNet(in_channels, in_channels, upsample_factor),
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3,
                      padding=(3 - 1) // 2)
        )
        
        self.condition_batchnorm2 = ConditionalBatchNorm1d(hidden_channels)
        self.linear2 = nn.Linear(z_channels, hidden_channels)
        self.second_stack = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3,
                      dilation=2, padding=2 * (3 - 1) // 2)
        )
        
        self.residual1 = nn.Sequential(
           UpsampleNet(in_channels, in_channels, upsample_factor), 
           nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        )
        
        self.condition_batchnorm3 = ConditionalBatchNorm1d(hidden_channels)
        self.linear3 = nn.Linear(z_channels, hidden_channels)
        self.third_stack = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3,
                      dilation=4, padding=4 * (3 - 1) // 2)
        )
        
        self.condition_batchnorm4 = ConditionalBatchNorm1d(hidden_channels)
        self.linear4 = nn.Linear(z_channels, hidden_channels)
        self.fourth_stack = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3,
                      dilation=8, padding=8 * (3 - 1) // 2)
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
        
    
class UpsampleNet(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 upsample_factor,
                 use_lstm=False,
                 lstm_layer=2,
                 upsample_method="duplicate"):

        super(UpsampleNet, self).__init__()
        self.upsample_method = upsample_method
        self.upsample_factor = upsample_factor
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm_layer = nn.LSTM(input_size, output_size, num_layers=lstm_layer, batch_first=True)
        if upsample_method == 'duplicate':
            self.upsample_factor = int(np.prod(upsample_factor))
        elif upsample_method == 'transposed_conv2d':
            assert isinstance(upsample_factor, tuple)
            kernel_size = 3
            self.upsamples = nn.ModuleList()
            for u in upsample_factor:
                padding = (kernel_size - 1) // 2
                conv = nn.ConvTranspose2d(1, 1, (kernel_size, 2 * u),
                                          padding=(padding, u // 2),
                                          dilation=1, stride=(1, u))
                self.upsamples.append(conv)

    def forward(self, inputs):
        if self.use_lstm:
           inputs, _ = self.lstm_layer(inputs.transpose(1, 2))
           inputs = inputs.transpose(1, 2)
        if self.upsample_method == 'duplicate':
            output = F.interpolate(inputs, scale_factor=self.upsample_factor, mode='nearest')
        elif self.upsample_method == 'transposed_conv2d':
            output = input.unsqueeze(1)
            for layer in self.upsamples:
                output = layer(output)
            output = output.squeeze(1)
            output = output[:, :, : input.size(-1) * np.prod(self.upsample_factor)]

        return output
    
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
        self.scale = nn.Linear(num_features, num_features)
        self.shift = nn.Linear(num_features, num_features)

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

model = Encoder(567, 128)

condition = torch.randn(2, 567, 10)
z = torch.randn(2, 128)

output = model(condition, z)
print(output.shape)