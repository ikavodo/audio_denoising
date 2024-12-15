import librosa
import numpy as np
import os
import soundfile as sf
import argparse

# Define the augmentation functions

def time_stretch(audio):
    """
    Apply time stretching to the audio by a fixed rate.
    """
    time_stretch = librosa.effects.time_stretch(audio, rate=1.2)
    return time_stretch


def add_background_noise(audio, noise_path):
    """
    Add background noise to the audio by overlaying a random segment of the noise.
    """
    noise, sr = librosa.load(noise_path, sr=None)
    noise_start = np.random.randint(0, len(noise) - len(audio))
    noise_clip = noise[noise_start:noise_start + len(audio)]
    noisy_audio = audio + noise_clip * 0.1  # Add 10% of the noise to the audio
    return noisy_audio


def augment_audio(audio, sr, noise_path=None):
    """
    Augment the audio with time stretch, pitch shift, and optional background noise.
    """
    # Create a copy of the original audio
    augmented_audio = audio.copy()

    # Apply pitch shift
    n_steps = np.random.randint(-3, 3)  # Random pitch shift within range
    augmented_audio = librosa.effects.pitch_shift(augmented_audio, sr=sr, n_steps=n_steps)

    # Apply time stretching
    augmented_audio = time_stretch(augmented_audio)

    # Optionally, add background noise
    if noise_path:
        augmented_audio = add_background_noise(augmented_audio, noise_path)

    return augmented_audio


def process_directory(audio_dir, noise_path, N):
    """
    Process each audio file in the directory and create N augmented versions for each file.
    """
    # Loop through each audio file in the specified directory
    for audio_file in os.listdir(audio_dir):
        audio_path = os.path.join(audio_dir, audio_file)

        # Check if the file is an audio file
        if not audio_file.endswith(('.wav', '.ogg', '.mp3')):  # Add other formats if needed
            continue

        # Load the audio file
        audio, sr = librosa.load(audio_path, sr=None)

        # Create N augmented copies of each audio file
        for i in range(N):
            augmented_audio = augment_audio(audio, sr, noise_path)

            # Define output path for the augmented file
            output_path = os.path.join(audio_dir, f"{audio_file}_{i}.ogg")

            # Save the augmented audio to the specified format
            sf.write(output_path, augmented_audio, sr, format='ogg')  # Use the desired format


def main():
    """
    Main function to run the augmentation script.
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Apply audio augmentations to files in a directory.')
    parser.add_argument('audio_dir', type=str, help='Directory containing the audio files')
    parser.add_argument('--noise_path', type=str, default=None, help='Path to background noise file (optional)')
    parser.add_argument('--N', type=int, default=5, help='Number of augmented copies to create for each audio file')

    # Parse arguments
    args = parser.parse_args()

    # Process the audio directory and create augmented files
    process_directory(args.audio_dir, args.noise_path, args.N)


if __name__ == "__main__":
    main()
