import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalBatchNorm1d(nn.BatchNorm1d):
    
    """Conditional Batch Normalization"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        
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

m = ConditionalBatchNorm1d(80, affine=False)
input = torch.randn(20, 80, 100)
weight = torch.randn(20, 80)

output = m(input, weight)
print(output.shape)

