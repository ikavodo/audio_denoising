import torch
import os
import librosa
from util import NUM_CHANNELS, PI, WAV

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, samplesDir, labelsDir, samp_num=None):
        self.separator1 = '_'
        self.suffix = WAV
        self.separator2 = '__'
        self.fs = None
        self.sampDir = samplesDir
        self.labelsDir = labelsDir
        self.samples = []
        self.labels = []
        self.len = samp_num
        self.n_fft = 1024
        self.frames_num = 128
        self.frames = None

        # Loop through the augmented directories
        for folder in os.listdir(self.sampDir):
            folder_path = os.path.join(self.sampDir, folder)

            # Only process directories
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(self.suffix):  # Ensure it's a .wav file
                        sample_path = os.path.join(folder_path, file)

                        # Get the original file name (without _index suffix)
                        label_name = file.split('_')[0]
                        label_path = os.path.join(self.labelsDir, label_name + self.suffix)

                        self.samples.append(sample_path)
                        self.labels.append(label_path)

        # Take a random subset of the samples if specified
        if self.len is not None:
            inds = torch.randint(len(self.samples), (self.len,))
            self.samples = [self.samples[ind] for ind in inds]
            self.labels = [self.labels[ind] for ind in inds]

    def __getitem__(self, index):
        sample, fs = librosa.load(self.samples[index])
        if self.fs is None or self.fs != fs:
            self.fs = fs
        label, _ = librosa.load(self.labels[index])
        return self._prepareSample(sample), self._prepareLabel(label)

    def __len__(self):
        return self.len if self.len is not None else \
            len(self.samples)

    def _prepareSample(self, data):
        """
        Convert to 12-channel STFT input
        """
        tens = torch.Tensor(data)
        stftData = torch.view_as_real(torch.stft(tens, n_fft=self.n_fft,
                                                 window=torch.hamming_window(self.n_fft), return_complex=True))
        # Choose random frames from sample
        self.frames = torch.randint(stftData.size(dim=1), (self.frames_num,))
        stftData = stftData[:, self.frames, :]
        freqVect = torch.arange(0, self.n_fft // 2 + 1) * (self.fs / self.n_fft)

        newTens = torch.zeros([NUM_CHANNELS, stftData.shape[1], stftData.shape[0]])
        newTens[:2, :, :] = stftData.permute(*torch.arange(stftData.ndim - 1, -1, -1))
        posEmbeds = torch.FloatTensor([[torch.cos(2 ** i * PI * f / self.fs) for i in range(10)]
                                       for f in freqVect])
        fullSize = posEmbeds.T.unsqueeze(1).repeat(1, stftData.shape[1], 1)
        newTens[2:, :, :] = fullSize
        return newTens[:, :, :-1]

    def _prepareLabel(self, data):
        """
        Convert to 12-channel STFT input
        """
        toMono = torch.sum(data, dim=1) if data.ndim > 1 else data
        tens = torch.Tensor(toMono)
        stftData = torch.view_as_real(torch.stft(tens, n_fft=self.n_fft,
                                                 window=torch.hamming_window(self.n_fft), return_complex=True))
        stftData = stftData.permute(*torch.arange(stftData.ndim - 1, -1, -1))
        return stftData[:, self.frames, :-1]
