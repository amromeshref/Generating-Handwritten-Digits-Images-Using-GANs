import os
import sys
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.append(REPO_DIR_PATH)


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


NOISE_DIM = 100
NUM_CLASSES = 10


class CGANModelPredictor:
    def __init__(self):
        self.generator, _ = self.load_models()

    def load_models(self) -> tuple[tf.keras.models.Sequential, tf.keras.models.Sequential]:
        """
        Load generator and discriminator models
        Args:
            None
        Returns:
            generator: tensorflow.keras.models.Sequential, generator model
            discriminator: tensorflow.keras.models.Sequential, discriminator model
        """
        generator_model_path = os.path.join(
            REPO_DIR_PATH, "models", "best-models", "generator_cgan.h5")
        generator = tf.keras.models.load_model(generator_model_path)
        discriminator_model_path = os.path.join(
            REPO_DIR_PATH, "models", "best-models", "discriminator_cgan.h5")
        discriminator = tf.keras.models.load_model(discriminator_model_path)
        return generator, discriminator

    def generate_images(self, num_images: int) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Generate images using generator model
        Args:
            num_images: int, number of images to generate
        Returns:
            generated_images: tf.Tensor, generated images
            labels: tf.Tensor, labels of generated images
        """
        noise = tf.random.normal([num_images, NOISE_DIM])
        labels = tf.random.uniform(
            [num_images], minval=0, maxval=NUM_CLASSES, dtype=tf.int32)
        generated_images = self.generator([noise, labels], training=False)
        return generated_images, labels

    def plot_images(self, images: tf.Tensor, labels: tf.Tensor) -> None:
        """
        Plot images
        Args:
            images: tf.Tensor, images to plot
            labels: tf.Tensor, labels of images
        Returns:
            None
        """
        images = images.numpy()
        labels = labels.numpy()

        # Determine the number of images to plot
        num_images = images.shape[0]

        # Define the plot grid size (adjust this as needed)
        grid_size = int(np.ceil(np.sqrt(num_images)))

        # Create a figure with the defined grid size
        plt.figure(figsize=(grid_size * 2, grid_size * 2))

        for i in range(num_images):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(images[i], cmap='gray')
            plt.title(f'Label: {labels[i]}')
            plt.axis('off')  # Hide axes

        plt.tight_layout()
        plt.show()
