from tensorboardX import SummaryWriter
from .plot import plot_waveform_to_numpy

class Writer(SummaryWriter):
    def __init__(self, logdir, sample_rate=16000):
        super(Writer, self).__init__(logdir)

        self.sample_rate = sample_rate
        self.logdir = logdir

    def logging_loss(self, losses, step):
        for key in losses:
            self.add_scalar('{}'.format(key), losses[key], step)

    def logging_audio(self, target, prediction, step):
        self.add_audio('raw_audio_predicted', prediction, step, self.sample_rate)
        self.add_image('waveform_predicted', plot_waveform_to_numpy(prediction), step)
        self.add_audio('raw_audio_target', target, step, self.sample_rate)
        self.add_image('waveform_target', plot_waveform_to_numpy(target), step)

    def logging_histogram(self, model, step):
        for tag, value in model.named_parameters():
            self.add_histogram(tag.replace('.', '/'), value.cpu().detach().numpy(), step)
