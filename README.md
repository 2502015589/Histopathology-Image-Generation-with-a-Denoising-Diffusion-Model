# Histopathology Image Generation with DDPM

This project implements an unconditional denoising diffusion probabilistic model (DDPM) in PyTorch to generate 64 × 64 RGB histopathology patches from Gaussian noise.

## Setup and usage

### Environment

- Python 3.13
- PyTorch 2.6.0 with CUDA 12.4 for the recorded experiment

### Install dependencies

For GPU training, use matching CUDA-enabled builds of PyTorch 2.6.0 and torchvision 0.21.0.

```bash
python -m pip install -r requirements.txt
```

### Run

Open `as3.ipynb` in Jupyter or VS Code with the repository root as the working directory, then run the cells in order. To resume training, keep `denoising_diffusion_checkpoint.pt` in the working directory and set EPOCHS to the desired total.

The notebook saves the loss curve, 16 individual samples, and an overview grid to `outputs/`. Set `RUN_FULL_EVAL = False` to skip FID and Inception Score. The first evaluation may need internet access to download pretrained Inception weights.

## Dataset

The project uses the Histopathology subset of the [Scientific Image Classification Dataset](https://www.kaggle.com/datasets/rushilprajapati/data-final). Place the extracted images in `data/Histopathology/`. The notebook uses 4,143 images for training and 1,036 held-out images for evaluation.

Images are converted to RGB, resized with their aspect ratios preserved, and center-cropped to 64 × 64. Pixel values are scaled to [−1, 1] for training.

## Model and training

The denoiser is a time-conditioned U-Net with 1,455,843 trainable parameters. It uses two residual blocks at each encoder and decoder resolution and skip connections between matching resolutions. Sinusoidal timestep embeddings condition the residual blocks on the noise level.

During training, each image is corrupted with Gaussian noise at a randomly sampled timestep. The network predicts the v-prediction target, a timestep-dependent combination of the noise and the clean image, and is optimized with mean squared error. During sampling, the model starts from Gaussian noise and iteratively estimates the clean image, clips the estimate to [−1, 1], and samples the next reverse step.

| Parameter | Value |
|---|---|
| Image size | 64 × 64 RGB |
| Optimizer | AdamW |
| Learning rate | 0.0002 |
| Batch size | 32 |
| Training epochs | 50 |
| Diffusion steps | 1,000 |
| Linear beta schedule | 0.0001 to 0.02 |
| Training, split, and sample seed | 285 |

## Results

### Training loss

The training v-prediction MSE fell from 0.26957 at epoch 1 to 0.09977 at epoch 50.

![Training v-prediction loss over 50 epochs](examples/training_loss.png)

### Generated images

The grid shows held-out images on the left and 16 images generated from the final checkpoint on the right. The samples reproduce some of the pink-purple coloration and local texture. Cell nuclei and tissue boundaries remain less distinct than in the held-out images.

![Held-out and generated histopathology patches](examples/overview.png)

### Evaluation

The completed 50-epoch experiment used PyTorch-Ignite to score 1,036 generated images. FID compares them with 1,036 held-out images. Inception Score uses the generated images alone. Both groups were resized to 299 × 299 for Inception evaluation.

| Epoch | FID | Inception Score |
|---:|---:|---:|
| 50 | 0.0875 | 1.4496 |
