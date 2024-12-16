import librosa
import os
import soundfile as sf
import argparse
import torch

# Define the augmentation functions
def time_stretch(audio):
    """
    Apply time stretching to the audio by a fixed rate.
    """
    return librosa.effects.time_stretch(audio, rate=1.2)


def add_background_noise(audio, noise_dir):
    """
    Add background noise to the audio by overlaying a random segment of the noise.
    """
    noise_files = [f for f in os.listdir(noise_dir) if os.path.isfile(os.path.join(noise_dir, f))]

    # Check if the directory has any files
    if noise_files:
        # Choose a random file
        random_index = torch.randint(0, len(noise_files), (1,)).item()  # Generate a random index
        noise_path = os.path.join(noise_dir, noise_files[random_index])  # Full path to the chosen file
        noise, sr = librosa.load(noise_path, sr=None)
        noise_start = torch.randint(0, len(noise) - len(audio), (1,)).item()
        noise_clip = noise[noise_start:noise_start + len(audio)]
    else:
        noise_clip = 0
    noisy_audio = audio + noise_clip * 0.1  # Add 10% of the noise to the audio
    return noisy_audio


def augment_audio(audio, sr, noise_dir=None):
    """
    Augment the audio with time stretch, pitch shift, and optional background noise.
    """
    augmented_audio = audio.copy()

    # Apply pitch shift
    n_steps = torch.randint(-3, 3, (1,)).item()
  # Random pitch shift within range
    augmented_audio = librosa.effects.pitch_shift(augmented_audio, sr=sr, n_steps=n_steps)

    # Apply time stretching
    augmented_audio = time_stretch(augmented_audio)

    # Optionally, add background noise
    if noise_dir:
        augmented_audio = add_background_noise(augmented_audio, noise_dir)

    return augmented_audio


def process_directory(audio_dir, augmented_dir, noise_dir=None, N=5):
    """
    Process each audio file in the directory and create N augmented versions for each file.
    The augmented files will be saved in subdirectories named after the original files.
    """
    # Ensure the augmented directory exists
    if not os.path.exists(augmented_dir):
        os.makedirs(augmented_dir)

    # Loop through each audio file in the specified directory
    for audio_file in os.listdir(audio_dir):
        audio_path = os.path.join(audio_dir, audio_file)

        # Skip non-audio files
        if not audio_file.endswith(('.wav', '.mp3')):
            continue

        # Load the audio file
        audio, sr = librosa.load(audio_path, sr=None)

        # Create a subdirectory for the augmented versions
        folder_name = audio_file.split('.')[0]  # Use the file name without extension
        folder_path = os.path.join(augmented_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Create N augmented copies of each audio file
        for i in range(N):
            augmented_audio = augment_audio(audio, sr, noise_dir)

            # Define output path for the augmented file
            output_file = f"{folder_name}_{i}.wav"  # e.g., 1-26806-A_0.wav
            output_path = os.path.join(folder_path, output_file)

            # Save the augmented audio to the specified format
            sf.write(output_path, augmented_audio, sr, format='wav')  # Save as WAV


def main():
    """
    Main function to run the augmentation script.
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Apply audio augmentations to files in a directory.')
    parser.add_argument('audio_dir', type=str, help='Directory containing the audio files')
    parser.add_argument('augmented_dir', type=str, help='Directory to save augmented files')
    parser.add_argument('--noise_dir', type=str, default=None, help='Path to background noise files (optional)')
    parser.add_argument('--N', type=int, default=5, help='Number of augmented copies to create for each audio file')

    # Parse arguments
    args = parser.parse_args()

    # Process the audio directory and create augmented files
    process_directory(args.audio_dir, args.augmented_dir, args.noise_dir, args.N)
    print("augmented data successfully.")

if __name__ == "__main__":
    main()
