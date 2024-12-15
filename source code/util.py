import torch
import torch.nn as nn
import numpy as np
import librosa

NUM_CHANNELS = 12

def compute_accuracy(net, testloader, two_outputs=False):
    """
    Computes accuracy for the network on a given test dataset.

    Args:
        net: The neural network model.
        testloader: DataLoader for the test dataset.
        two_outputs: Whether the model produces two outputs (default is False).

    Returns:
        The average loss over the entire dataset based only on the second output.
    """
    device = torch.device('cuda:0')
    loss_test_fn = nn.L1Loss()
    net.eval()  # Set the model to evaluation mode
    total = 0

    with torch.no_grad():  # No need to track gradients for validation
        for samples, labels in testloader:
            samples, labels = samples.to(device), labels.to(device)

            if two_outputs:  # Model returns two outputs
                _, outputs = net(samples)  # Ignore the first output
            else:  # Model returns one output
                outputs = net(samples)
            loss_test = loss_test_fn(outputs, labels)
            total += loss_test.item()

    # Return the average loss
    return total / len(testloader)


def prepareSample(data, n_fft, fs):
    """
    convert to 12-channel STFT input
    """
    toMono = np.sum(data, axis=1) if data.ndim > 1 else data
    tens = torch.Tensor(toMono)
    stftData = torch.view_as_real(torch.stft(tens, n_fft=n_fft,
                                             window=torch.hamming_window(n_fft), return_complex=True))
    # add frames for power of two number
    toConcat = np.int_(2 ** (np.ceil(np.log2(stftData.size(dim=1)))) - stftData.size(dim=1))
    extraFrames = torch.zeros(stftData.size(dim=0), toConcat, stftData.size(dim=2))
    stftData = torch.cat((stftData, extraFrames), 1)
    freqVect = np.arange(n_fft / 2 + 1) * fs / n_fft
    # newTens.shape = (channels,frames,freq)
    newTens = torch.zeros([NUM_CHANNELS, stftData.shape[1], stftData.shape[0]])
    newTens[:2, :, :] = stftData.permute(*torch.arange(stftData.ndim - 1, -1, -1))
    posEmbeds = torch.FloatTensor([[np.cos(2 ** i * np.pi * f / fs) for i in range(10)]
                                   for f in freqVect])
    fullSize = posEmbeds.T.unsqueeze(1).repeat(1, stftData.shape[1], 1)
    newTens[2:, :, :] = fullSize
    #         need 1024 bins
    return newTens[:, :, :-1]


# calculate SNR
def forSNR(path, n_fft):
    sample, _ = librosa.load(path)
    tens = torch.Tensor(sample)
    stftData = torch.stft(tens, n_fft=n_fft, window=torch.hamming_window(n_fft),
                          return_complex=True)
    return stftData


def compute_segSNR(stft1, stft2):
    clean = torch.sum(torch.square(torch.abs(stft1)), dim=0)
    error = torch.sum(torch.square(torch.abs(stft1 - stft2)), dim=0)
    divided = torch.div(clean, error)
    todB = torch.div(divided, 20e-6)
    return torch.mul(torch.log10(todB), 20)
