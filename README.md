# Denoise Network

This repository contains a PyTorch-based implementation of a denoising neural network for audio processing, based on the architecture proposed [here](https://arxiv.org/abs/2202.08702)[1]. The system includes functionality for augmenting audio data of choice (See the [ESC-50](https://github.com/karolpiczak/ESC-50) environmental noise dataset submodule as an example for environmental sound recording), training one of two UNet architectures, and then using the trained model to denoise another audio file of choice.

---

## Setup

You need Python 3.x, PyTorch and Librosa installed.

---

## Command Line Usage

### **1. Augment Audio Data**

To augment audio files located in a directory (e.g., `ESC-50/audio/`) for training, use the following command from inside the source code directory:

```bash
python augmentation.py --audio_dir ./ESC-50/audio/ --output_dir ./ESC-50/augmented/ --N 5 --noise_dir ./noise_dir
```
where noise_dir would be a directory with some noises you would want to get rid of in your target audio files (pink noise f.e.).

#### Arguments:
- `--audio_dir`: Directory containing the original audio files.
- `--output_dir`: Directory where augmented data will be saved.
- `--N`: Number of augmented versions to generate for each file.
- `--noise_dir`: Optional path to a directory with noise files to be used in augmentation.

The augmentation script will generate directories for each file in `audio_dir` and populate them with augmented versions of the original file.

---

### **2. Train the Model**

To train the denoising network, use the `train.py` script. You will need to divide your samples into training and test datasets (75% train 25% test is a good baseline). Specify the paths to these datasets and indicate whether to train the smaller or larger model:

```bash
python train.py --train_dir ./ESC-50/augmented/ --test_dir ./ESC-50/audio/ --train_labels ./ESC-50/train_labels/ --test_labels ./ESC-50/test_labels/ --model_type small
```

#### Arguments:
- `--train_dir`: Path to the directory containing the training data.
- `--test_dir`: Path to the directory containing the test data.
- `--train_labels`: Path to the labels for the training data.
- `--test_labels`: Path to the labels for the test data.
- `--model_type`: Specify which model to train (`small` or `large`).

The script will train the denoising network and save the model to `model.pth` by default. Notice it will also use a GPU if one such is available.

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

[1] Eloi Moliner, & Vesa Välimäki. (2022). A Two-Stage U-Net for High-Fidelity Denoising of Historical Recordings.

