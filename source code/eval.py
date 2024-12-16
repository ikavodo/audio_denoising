import os
import argparse
import torch
import soundfile as sf
import librosa
from smallModel import smallModel  # Import your small model class
from Model import Model  # Import your large model class
from util import prepareSample # Assuming you have a function to prepare the sample


# Define the main function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate and denoise audio using a trained model')
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory containing audio files to denoise')
    parser.add_argument('--is_small_model', type=bool, default=True, help='Flag to choose between small or large model')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to the trained model file')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT size for the STFT')
    args = parser.parse_args()

    # Set the device (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model
    if args.is_small_model:
        model = smallModel()
        print("Using small model for evaluation.")
    else:
        model = Model()
        print("Using large model for evaluation.")

    # Load the trained weights
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.to(device)

    # Process each audio file in the directory
    for audio_file in os.listdir(args.audio_dir):
        audio_path = os.path.join(args.audio_dir, audio_file)

        # Only process audio files (skip non-audio files)
        if not audio_file.endswith(('.wav')):
            continue

        print(f"Processing {audio_file}...")

        # Load the audio file
        noisy, fs = librosa.load(audio_path, sr=None)

        # Prepare the sample (e.g., STFT)
        frames = prepareSample(noisy, args.n_fft, fs)

        # Denoise the audio
        with torch.no_grad():
            out = model(torch.unsqueeze(frames, 0).to(device))
            out = torch.squeeze(out, 0).cpu()
            out = out.permute(*torch.arange(out.ndim - 1, -1, -1))
            complexOut = torch.view_as_complex(out.contiguous())
            complexOut = torch.cat((complexOut, torch.zeros(1, complexOut.size(dim=1))), dim=0)

            # Inverse STFT to get denoised audio
            denoised = torch.istft(complexOut, n_fft=args.n_fft, window=torch.hamming_window(args.n_fft),
                                   return_complex=False)[:len(noisy)]

        # Save the denoised audio
        denoised_out_path = os.path.join(args.audio_dir, f"denoised_{audio_file}")
        sf.write(denoised_out_path, denoised.numpy(), fs)
        print(f"Saved denoised audio to {denoised_out_path}")

        # n_fft = 2048
        # stfts = forSNR(denoised_out_path, n_fft)
        # snr1 = compute_segSNR(stfts[0], stfts[1])
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.plot(snr1)
        # # Plot the noisy signal and the segmental SNRs from the 4 methods. Let the x-axis denote frames and y-axis denote the SNR in dB
        # # plt.legend('avg segmental SNR = %.2f' % torch.mean(snr1))
        # ax.set_title("model segmental SNR")
        # ax.set_xlabel("frame")
        # ax.set_ylabel("dB")
        # # ax.axhline(y=torch.mean(snr1),linestyle = '--', label='avg')
        # # ax.legend()
        #
        # fig.show()

# parser.add_argument('--audio_dir', type=str, required=True, help='Directory containing audio files to denoise')
if __name__ == "__main__":
    main()
