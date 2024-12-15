# Denoise Network

This repository implements a denoising neural network for audio processing. The system includes functionality for augmenting audio data, training models, and evaluating the denoising performance.

---

## Setup

Ensure you have the required dependencies installed. Use the following command to install dependencies if a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

You need Python 3.x and PyTorch installed.

---

## Command Line Usage

### **1. Augment Audio Data**

To augment audio files located in a directory (e.g., `ESC-50/audio/`), use the following command:

```bash
python augmentation.py --audio_dir ./ESC-50/audio/ --output_dir ./ESC-50/augmented/ --N 5 --noise_path ./background_noise.wav
```

#### Arguments:
- `--audio_dir`: Directory containing the original audio files.
- `--output_dir`: Directory where augmented data will be saved.
- `--N`: Number of augmented versions to generate for each file.
- `--noise_path`: Optional path to a background noise file to be used in augmentation.

The augmentation script will generate directories for each file in `audio_dir` and populate them with augmented versions.

---

### **2. Train the Model**

To train the denoising network, use the `train.py` script. Specify the paths to the training and test datasets and indicate whether to train the small or large model:

```bash
python train.py --train_dir ./ESC-50/augmented/ --test_dir ./ESC-50/audio/ --train_labels ./ESC-50/train_labels/ --test_labels ./ESC-50/test_labels/ --model_type small
```

#### Arguments:
- `--train_dir`: Path to the directory containing the training data.
- `--test_dir`: Path to the directory containing the test data.
- `--train_labels`: Path to the labels for the training data.
- `--test_labels`: Path to the labels for the test data.
- `--model_type`: Specify which model to train (`small` or `large`).

The script will train the denoising network and save the model to `model.pth` by default.

---

### **3. Evaluate the Model**

To evaluate the denoising network on a directory of audio files, use the `eval.py` script:

```bash
python eval.py --audio_dir ./ESC-50/audio/ --output_dir ./ESC-50/denoised/ --model_type small
```

#### Arguments:
- `--audio_dir`: Directory containing the audio files to denoise.
- `--output_dir`: Directory where denoised audio files will be saved.
- `--model_type`: Specify which model to use (`small` or `large`).

The script will denoise all files in the specified directory and save them with the prefix `DENOISED_` in the output directory.

---

## Notes

1. **CustomDataset Integration**: The dataset is designed to work with both original and augmented data. Ensure that directories are structured correctly and labeled data is available.
2. **Visualization**: The `train.py` script also provides an option to visualize training and testing losses. Check the script for details.

---

