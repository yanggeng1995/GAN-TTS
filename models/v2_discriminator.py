import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                nn.utils.spectral_norm(nn.Conv1d(1, 16, kernel_size=15)),
                nn.LeakyReLU(0.2, True),
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(16, 64, kernel_size=41,
                                     stride=4, padding=20, groups=4)),
                nn.LeakyReLU(0.2, True),
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(64, 256, kernel_size=41,
                                     stride=4, padding=20, groups=16)),
                nn.LeakyReLU(0.2, True),
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(256, 1024, kernel_size=41,
                                     stride=4, padding=20, groups=64)),
                nn.LeakyReLU(0.2, True),
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(1024, 1024, kernel_size=41,
                                     stride=4, padding=20, groups=256)),
                nn.LeakyReLU(0.2, True),
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(1024, 1024, kernel_size=5,
                                     stride=1, padding=2)),
                nn.LeakyReLU(0.2, True),
            ),
            nn.utils.spectral_norm(nn.Conv1d(1024, 1, kernel_size=3,
                                 stride=1, padding=1)),
        ])

    def forward(self, x):
        for layer in self.discriminator:
            x = layer(x)

        return x

if __name__ == '__main__':

    model = Discriminator()
    x = torch.randn(3, 1, 24000)

    score = model(x)
    print(score.shape)
