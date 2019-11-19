import torch
import torch.nn as nn
import torch.nn.functional as F

def stft(x, fft_size, hop_size, win_size, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x: Input signal tensor (B, T).

    Returns:
        Tensor: Magnitude spectrogram (B, T, fft_size // 2 + 1).

    """
    x_stft = torch.stft(x, fft_size, hop_size, win_size, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    outputs = torch.clamp(real ** 2 + imag ** 2, min=1e-7).transpose(2, 1)
    outputs = torch.sqrt(outputs)

    return outputs 

class SpectralConvergence(nn.Module):
    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergence, self).__init__()
        
    def forward(self, predicts_mag, targets_mag):
        x = torch.norm(targets_mag - predicts_mag, p='fro')
        y = torch.norm(targets_mag, p='fro')
        
        return x / y 

class LogSTFTMagnitude(nn.Module):
    def __init__(self):
        super(LogSTFTMagnitude, self).__init__()
        
    def forward(self, predicts_mag, targets_mag):
        log_predicts_mag = torch.log(predicts_mag)
        log_targets_mag = torch.log(targets_mag)
        outputs = F.l1_loss(log_predicts_mag, log_targets_mag)

        return outputs

class STFTLoss(nn.Module):
    def __init__(self,
                 fft_size=1024,
                 hop_size=120,
                 win_size=600):
        super(STFTLoss, self).__init__()
        
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.window = torch.hann_window(win_size)
        self.sc_loss = SpectralConvergence()
        self.mag = LogSTFTMagnitude()
        
    
    def forward(self, predicts, targets):
        """
        Args:
            x: predicted signal (B, T).
            y: truth signal (B, T).

        Returns:
            Tensor: STFT loss values.
        """
        predicts_mag = stft(predicts, self.fft_size, self.hop_size, self.win_size, self.window)
        targets_mag = stft(targets, self.fft_size, self.hop_size, self.win_size, self.window)
        
        sc_loss = self.sc_loss(predicts_mag, targets_mag)
        mag_loss = self.mag(predicts_mag, targets_mag)
        
        return sc_loss, mag_loss

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 win_sizes=[600, 1200, 240],
                 hop_sizes=[120, 240, 50]):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.loss_layers = torch.nn.ModuleList()
        for (fft_size, win_size, hop_size) in zip(fft_sizes, win_sizes, hop_sizes):
            self.loss_layers.append(STFTLoss(fft_size, hop_size, win_size))
            
    def forward(self, fake_signals, true_signals):
        sc_losses, mag_losses = [], []
        for layer in self.loss_layers:
            sc_loss, mag_loss = layer(fake_signals, true_signals)
            sc_losses.append(sc_loss)
            mag_losses.append(mag_loss)
        
        sc_loss = sum(sc_losses) / len(sc_losses)
        mag_loss = sum(mag_losses) / len(mag_losses)

        return sc_loss, mag_loss

if __name__ == "__main__":
    model = MultiResolutionSTFTLoss()
    x = torch.randn(2, 16000)
    y = torch.randn(2, 16000)
    
    loss = model(x, y)
    print(loss)
