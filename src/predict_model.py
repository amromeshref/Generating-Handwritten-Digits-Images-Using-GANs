import os
import sys
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.append(REPO_DIR_PATH)


import tensorflow as tf
import matplotlib.pyplot as plt



class ModelPredictor:
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
            REPO_DIR_PATH, "models", "best-models", "generator.h5")
        generator = tf.keras.models.load_model(generator_model_path)
        discriminator_model_path = os.path.join(
            REPO_DIR_PATH, "models", "best-models", "discriminator.h5")
        discriminator = tf.keras.models.load_model(discriminator_model_path)
        return generator, discriminator

    def generate_images(self, num_images: int) -> tf.Tensor:
        """
        Generate images using generator model
        Args:
            num_images: int, number of images to generate
        Returns:
            generated_images: tf.Tensor, generated images
        """
        noise = tf.random.normal([num_images, 100])
        generated_images = self.generator(noise, training=False)
        return generated_images

    def plot_images(self, images: tf.Tensor) -> None:
        """
        Plot images
        Args:
            images: tf.Tensor, images to plot
        Returns:
            None
        """
        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(5, 5, i+1)
            plt.imshow(images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        plt.show()
