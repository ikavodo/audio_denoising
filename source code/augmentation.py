import librosa
import numpy as np
import os
import soundfile as sf

# Define the augmentation functions
# def pitch_shift(audio, sr):
#     n_steps = np.random.randint(-3, 3)
#     pitch_shift = librosa.effects.pitch_shift(audio, sr, n_steps)
#     return pitch_shift

def time_stretch(audio):
    time_stretch = librosa.effects.time_stretch(audio, rate=1.2)
    return time_stretch

def add_background_noise(audio, noise_path):
    noise, sr = librosa.load(noise_path, sr=None)
    noise_start = np.random.randint(0, len(noise)-len(audio))
    noise_clip = noise[noise_start:noise_start+len(audio)]
    noisy_audio = audio + noise_clip*0.1 # Add 10% of the noise to the audio
    return noisy_audio

# Set the paths to your audio files and background noise
audio_dir = 'planetest'
#noise_path = 'path/to/your/background/noise.wav'

# Loop through each audio file and apply the augmentation functions
for audio_file in os.listdir(audio_dir):
    audio_path = os.path.join(audio_dir, audio_file)
    audio, sr = librosa.load(audio_path, sr=None)
    for i in range(100): # Create 100 augmented files for each audio file
        augmented_audio = audio.copy()
        n_steps = np.random.randint(-3, 3)
        augmented_audio = librosa.effects.pitch_shift(augmented_audio, sr=sr, n_steps=n_steps)
        augmented_audio = time_stretch(augmented_audio)
        #augmented_audio = add_background_noise(augmented_audio, noise_path)
        output_path = os.path.join(audio_dir, f"{audio_file}_{i}.ogg")
        #librosa.output.write_wav(output_path, augmented_audio, sr)
        sf.write(output_path, augmented_audio, sr, format='wav')