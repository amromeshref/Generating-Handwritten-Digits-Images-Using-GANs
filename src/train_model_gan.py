import os
import sys
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.append(REPO_DIR_PATH)

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Dropout, LeakyReLU, Conv2DTranspose, BatchNormalization, Reshape
from tensorflow.keras.models import Sequential



EPOCHS = 100
BATCH_SIZE = 256
NOISE_DIM = 100
BUFFER_SIZE = 60000


class ModelTrainer:
    def __init__(self):
        self.generator_optimizer = Adam(1e-4)
        self.discriminator_optimizer = Adam(1e-4)
        self.cross_entropy = BinaryCrossentropy(from_logits=True)

    def load_data(self) -> np.ndarray:
        """
        Load data
        Args:
            None
        Returns:
            images: np.ndarray, real images
        """
        images = np.load(os.path.join(REPO_DIR_PATH, "data",
                         "external", "mnist_x_train.npy"))
        images = self.preprocess_data(images)
        return images

    def preprocess_data(self, images: np.ndarray) -> np.ndarray:
        """
        Preprocess data
        Args:
            images: np.ndarray, input images
        Returns:
            images: np.ndarray, preprocessed images
        """
        images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
        # Normalize the images to [-1, 1]
        images = (images - 127.5) / 127.5
        return images

    def build_generator(self, input_shape: tuple = (100,)) -> Sequential:
        """
        Build generator model
        The output shape of the generator model is (28, 28, 1)
        Args:
            input_shape: tuple, input shape of the generator model
        Returns:
            model: tensorflow.keras.models.Sequential, generator model
        """
        model = Sequential()

        model.add(Dense(7*7*256, use_bias=False, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Reshape((7, 7, 256)))

        model.add(Conv2DTranspose(128, (5, 5), strides=(
            1, 1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(64, (5, 5), strides=(
            2, 2), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2),
                  padding='same', use_bias=False, activation='tanh'))

        return model

    def build_discriminator(self, input_shape: tuple = (28, 28, 1)) -> Sequential:
        """
        Build discriminator model
        Args:
            input_shape: tuple, input shape of the discriminator model
        Returns:
            model: tensorflow.keras.models.Sequential, discriminator model
        """
        model = Sequential()

        model.add(Conv2D(64, (5, 5), strides=(2, 2),
                  padding='same', input_shape=input_shape))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(1, activation='linear'))

        return model

    def calculate_generator_loss(self, yhat_fake: tf.Tensor) -> tf.Tensor:
        """
        Calculate generator loss
        Args:
            yhat_fake: tensor, output of the discriminator model when input is a generated(fake) image
        Returns:
            loss: tensor, generator loss
        """
        y_fake = tf.ones_like(yhat_fake)
        loss = self.cross_entropy(y_fake, yhat_fake)
        return loss

    def calculate_discriminator_loss(self, yhat_real: tf.Tensor, yhat_fake: tf.Tensor) -> tf.Tensor:
        """
        Calculate discriminator loss
        Args:
            yhat_real: tensor, output of the discriminator model when input is a real image
            yhat_fake: tensor, output of the discriminator model when input is a generated(fake) image
        Returns:
            loss: tensor, discriminator loss
        """
        y_real = tf.ones_like(yhat_real)
        y_fake = tf.zeros_like(yhat_fake)
        loss_real = self.cross_entropy(y_real, yhat_real)
        loss_fake = self.cross_entropy(y_fake, yhat_fake)
        loss = loss_real + loss_fake
        return loss

    def train_step(self, batch_size: int, noise_dim: int, generator: Sequential, discriminator: Sequential, real_images_batch: np.ndarray) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Perform one training step
        Args:
            batch_size: int, batch size
            noise_dim: int, dimension of the noise vector
            generator: tensorflow.keras.models.Sequential, generator model
            discriminator: tensorflow.keras.models.Sequential, discriminator model
            real_images_batch: np.ndarray, batch of real images
        Returns:
            discriminator_loss: tensor, discriminator loss
            generator_loss: tensor, generator loss
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate a batch of fake images
            noise = tf.random.normal([batch_size, noise_dim])
            fake_images = generator(noise, training=True)

            # Predict the labels of the real images using the discriminator
            yhat_real = discriminator(real_images_batch, training=True)

            # Predict the labels of the fake images using the discriminator
            yhat_fake = discriminator(fake_images, training=True)

            # Compute the discriminator loss
            discriminator_loss = self.calculate_discriminator_loss(
                yhat_real, yhat_fake)

            # Compute the generator loss
            generator_loss = self.calculate_generator_loss(yhat_fake)

        # Get the gradients of the generator and discriminator loss
        discriminator_gradients = disc_tape.gradient(
            discriminator_loss, discriminator.trainable_variables)
        generator_gradients = gen_tape.gradient(
            generator_loss, generator.trainable_variables)

        # Update the weights of the generator and discriminator
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables))
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables))

        return discriminator_loss, generator_loss

    def train(self, dataset: tf.data.Dataset, noise_dim: int, generator: Sequential, discriminator: Sequential, epochs: int, batch_size: int):
        """
        Train the GAN model
        Args:
            real_images: np.ndarray, real images
            noise_dim: int, dimension of the noise vector
            generator: tensorflow.keras.models.Sequential, generator model
            discriminator: tensorflow.keras.models.Sequential, discriminator model
            epochs: int, number of epochs
            batch_size: int, batch size
        Returns:
            None
        """
        date = datetime.now().strftime("%Y-%m-%d-%I-%M-%S")
        saved_dir = os.path.join(
            REPO_DIR_PATH, "results", "training_checkpoints_"+date)
        os.makedirs(saved_dir, exist_ok=True)
        os.makedirs(os.path.join(saved_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(saved_dir, "models"), exist_ok=True)
        for epoch in range(epochs):
            discriminator_loss = 0
            generator_loss = 0
            for image_batch in dataset:
                discriminator_loss, generator_loss = self.train_step(
                    batch_size, noise_dim, generator, discriminator, image_batch)
            print(
                f"Epoch {epoch+1}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}")
            if epoch % 5 == 0 or epoch == 0:
                fig = self.generate_images(generator, noise_dim)
                fig.savefig(os.path.join(saved_dir, "images",
                            f"image_at_epoch_{epoch+1}.png"))
                generator.save(os.path.join(saved_dir, "models",
                               f"generator_at_epoch_{epoch+1}.h5"))
                discriminator.save(os.path.join(
                    saved_dir, "models", f"discriminator_at_epoch_{epoch+1}.h5"))

    def generate_images(self, generator_model: Sequential, noise_dim: int) -> plt.Figure:
        """
        Generate images using generator model
        Args:
            generator_model: tensorflow.keras.models.Sequential, generator model
            noise_dim: int, dimension of the noise vector
        Returns:
            fig: matplotlib.pyplot.Figure, figure containing the generated images
        """
        noise = tf.random.normal([16, noise_dim])
        predictions = generator_model(noise, training=False)
        fig = plt.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        return fig


if __name__ == '__main__':
    trainer = ModelTrainer()
    real_images = trainer.load_data()
    generator = trainer.build_generator()
    discriminator = trainer.build_discriminator()
    dataset = tf.data.Dataset.from_tensor_slices(
        real_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    trainer.train(dataset, NOISE_DIM, generator,
                  discriminator, EPOCHS, BATCH_SIZE)
