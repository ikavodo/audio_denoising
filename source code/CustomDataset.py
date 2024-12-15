import torch
import os
import librosa
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, samplesDir, labelsDir, samp_num=None):
        self.separator1 = '_'
        self.suffix = '.ogg'
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

        #         keep track of file paths
        for s in os.listdir(self.sampDir):
            #             e.g. 'Rooster'
            # label_class = s[:s.find(self.separator)]
            #             e.g. '1-26806-A.ogg'
            label_name = s[s.find(self.separator1) + 1:s.rfind(self.separator2)]
            #             e.g. 'denoising2/denoising2/Rooster_test/1-26806-A.ogg'
            label_path = os.path.join(self.labelsDir, label_name + self.suffix)

            sample_path = os.path.join(self.sampDir, s)
            self.samples.append(sample_path)
            self.labels.append(label_path)
        # take 100 random samples
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
            sum(len(files) for _, _, files in os.walk(self.sampDir))

    def _prepareSample(self, data):
        """
        convert to 12-channel STFT input
        """
        tens = torch.Tensor(data)
        stftData = torch.view_as_real(torch.stft(tens, n_fft=self.n_fft,
                                                 window=torch.hamming_window(self.n_fft), return_complex=True))
        # choose random frames from sample
        self.frames = torch.randint(stftData.size(dim=1), (self.frames_num,))
        # add frames for power of two number
        # toConcat = np.int_(2**(np.ceil(np.log2(stftData.size(dim=1)))) - stftData.size(dim=1))
        # extraFrames = torch.zeros(stftData.size(dim=0),toConcat,stftData.size(dim=2))
        stftData = stftData[:, self.frames, :]
        # stftData = torch.cat((stftData,extraFrames),1)
        freqVect = np.arange(self.n_fft / 2 + 1) * self.fs / self.n_fft
        numChannels = 12
        # newTens.shape = (channels,frames,freq)
        newTens = torch.zeros([numChannels, stftData.shape[1], stftData.shape[0]])
        newTens[:2, :, :] = stftData.permute(*torch.arange(stftData.ndim - 1, -1, -1))
        posEmbeds = torch.FloatTensor([[np.cos(2 ** i * np.pi * f / self.fs) for i in range(10)]
                                       for f in freqVect])
        fullSize = posEmbeds.T.unsqueeze(1).repeat(1, stftData.shape[1], 1)
        newTens[2:, :, :] = fullSize
        #         need 1024 bins
        return newTens[:, :, :-1]

    def _prepareLabel(self, data):
        """
        convert to 12-channel STFT input
        """
        toMono = np.sum(data, axis=1) if data.ndim > 1 else data
        tens = torch.Tensor(toMono)
        stftData = torch.view_as_real(torch.stft(tens, n_fft=self.n_fft,
                                                 window=torch.hamming_window(self.n_fft), return_complex=True))
        stftData = stftData.permute(*torch.arange(stftData.ndim - 1, -1, -1))
        return stftData[:, self.frames, :-1]

