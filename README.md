# Handwritten Digits Generation Using GANs and CGANs

## Introduction

This project is designed for learning purposes, focusing on Generative Adversarial Networks (GANs) and Conditional 
Generative Adversarial Networks (CGANs). The primary goal is to understand these models and their applications in 
generating handwritten digits. Through this project, I aimed to enhance my knowledge of GANs and CGANs.


## Tabel of Contents
1. [Project Overview](#project-overview)
1. [Project Structure](#project-structure)
1. [Training](#training)
1. [Requirements](#requirements)
1. [Installation](#installation)
1. [Usage](#usage)

___

## Project Overview

The project includes the following key components:

1. **Training a GAN Network**: Develop and train a GAN network to generate handwritten digits from random noise.
The GAN consists of a generator and a discriminator where the generator creates images, and the discriminator evaluates their authenticity.

3. **Training a CGAN Network**: Implement and train a CGAN network that generates handwritten digits based on given labels.
The CGAN uses both noise and label information to produce digit images corresponding to specific classes.


___

## Project Structure

The project is organized as follows:

1. **Data Preparation**:
The data used for this project is sourced from the MNIST dataset, which contains grayscale images of handwritten digits (0-9).
Each image is 28x28 pixels in size. The dataset includes:
   - `data/external/mnist_x_train.npy`: Training images.
   - `data/external/mnist_y_train.npy`: Training labels.

3. **Scripts**:
   - `src/train_model_gan.py`: Script for training the GAN network.
   - `src/train_model_cgan.py`: Script for training the CGAN network.
   - `src/predict_model_gan.py`: Script for generating and plotting images using the trained GAN model.
   - `src/predict_model_cgan.py`: Script for generating and plotting images using the trained CGAN model.

___


## Training

In this project, the GAN model was trained for 85 epochs, while the CGAN model was trained for 100 epochs. 
The results from these trainings are saved in the `results` directory. 
For potentially better results, you may consider training the models for more epochs.


___


## Requirements
- TensorFlow
- NumPy
- Matplotlib


___


## Installation
1. Create a new environment with a 3.9 Python version.
1. Create a directory on your device and navigate to it.
1. Clone the repository:
   ```
   git clone https://github.com/amromeshref/Generating-Handwritten-Digits-Images-Using-GANs.git
   ```
1. Navigate to the `Generating-Handwritten-Digits-Images-Using-GANs` directory.
   ```
   cd Generating-Handwritten-Digits-Images-Using-GANs
   ```
1. Type the following command to install the requirements file using pip:
    ```
    pip install -r requirements.txt
    ```
1. To generate and visualize handwritten digits images using the trained GAN model, run this command:
   ```
   python3 src/predict_model_gan.py
   ```
1. To generate and visualize handwritten digits images based on labels using the trained CGAN model, run this command:
   ```
   python3 src/predict_model_cgan.py
   ```

___

## Usage

To use the pre-trained models to generate images:

#### Using the Pretrained GAN Model

```python
from src.predict_model_gan import GANModelPredictor

# Initialize the predictor
predictor = GANModelPredictor()

# Generate images
num_images = 25
images = predictor.generate_images(num_images)

# Plot generated images
predictor.plot_images(images)
```

#### Using the Pretrained CGAN Model

```python
from src.predict_model_cgan import CGANModelPredictor
import tensorflow as tf

# Initialize the predictor
predictor = CGANModelPredictor()

# Create random labels
num_images = 25
labels = tf.random.uniform([num_images], minval=0, maxval=10, dtype=tf.int32)

# Generate images
images = predictor.generate_images(num_images, labels)

# Plot the generated images with the corresponding labels
predictor.plot_images(images, labels)
```

___

## Other

A good [playlist](https://youtube.com/playlist?list=PLZsOBAyNTZwboR4_xj-n3K6XBTweC4YVD&si=vPdneLIHJJK2vd2Z) I watched to learn GANs.


