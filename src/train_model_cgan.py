import os
import sys

REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.append(REPO_DIR_PATH)

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Conv2DTranspose, BatchNormalization, Reshape, LeakyReLU, Embedding, Concatenate, Input
from tensorflow.keras.models import Model
import numpy as np
from datetime import datetime


EPOCHS = 100
BATCH_SIZE = 256
NOISE_DIM = 100
BUFFER_SIZE = 60000
NUM_CLASSES = 10


class ModelTrainerCGAN:
    def __init__(self):
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.num_classes = 10
    
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load data
        Args:
            None
        Returns:
            images: np.ndarray, real images
            labels: np.ndarray, labels of real images
        """
        images = np.load(os.path.join(REPO_DIR_PATH, "data", "external", "mnist_x_train.npy"))
        images = self.preprocess_data(images)
        labels = np.load(os.path.join(REPO_DIR_PATH, "data", "external", "mnist_y_train.npy"))
        return images, labels
    
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
    
    def build_generator(self, num_classes: int, noise_dim: int) -> tf.keras.Model:
        """
        Build generator model
        The output shape of the generator model is (None, 28, 28, 1)
        Args:
            num_classes: int, number of classes
            noise_dim: int, dimension of the noise
        Returns:
            generator: tf.keras.Model, generator model
        """
        # Input layers for the noise and the label
        input_noise_shape = (noise_dim,)
        input_noise = Input(shape=input_noise_shape)
        input_label = Input(shape=(1,), dtype='int32')

        # Output from the embedding layer will be (None, 1, 100)
        embedding = Embedding(num_classes, noise_dim)(input_label)

        # Reshape the output to (None, 100)
        embedding = Flatten()(embedding)

        # Concatenate the noise and the label
        model_input = Concatenate()([input_noise, embedding])

        x = Dense(7*7*256, use_bias=False)(model_input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Reshape((7, 7, 256))(x)

        x = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        y = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)

        generator = Model([input_noise, input_label], y)

        return generator
    
    def build_discriminator(self, num_classes: int, input_image_shape: tuple = (28,28,1)) -> tf.keras.Model:
        """
        Build discriminator model
        Args:
            num_classes: int, number of classes
            input_image_shape: tuple, input shape of the discriminator model
        Returns:
            discriminator: tf.keras.Model, discriminator model
        """
        # Input layers for the image and the label
        input_image = Input(shape=input_image_shape)
        input_label = Input(shape=(1,), dtype='int32')

        # Output from the embedding layer will be (None, 1, 784)
        embedding = Embedding(num_classes, np.prod(input_image_shape))(input_label)

        # Reshape the output to (None, 784)
        embedding = Flatten()(embedding)

        # Reshape the image to (None, 28, 28, 1)
        embedding = Reshape(input_image_shape)(embedding)

        # Concatenate the image and the label
        # The shape of the model_input will be (None, 28, 28, 2)
        model_input = Concatenate()([input_image, embedding])

        x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(model_input)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        x = Flatten()(x)
        y = Dense(1, activation='linear')(x)

        discriminator = Model([input_image, input_label], y)

        return discriminator
    
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

    def train_step(self, batch_size: int, noise_dim: int, num_classes: int, generator: Model, discriminator: Model, real_images_batch: np.ndarray, real_labels_batch: np.ndarray):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate a batch of fake images
            noise = tf.random.normal([batch_size, noise_dim])
            fake_labels = tf.random.uniform([batch_size], minval=0, maxval=num_classes, dtype=tf.int32)
            fake_images = generator([noise, fake_labels], training=True)

            # Predict the labels of the real images using the discriminator
            yhat_real = discriminator([real_images_batch, real_labels_batch], training=True)

            # Predict the labels of the fake images using the discriminator
            yhat_fake = discriminator([fake_images, fake_labels], training=True)

            # Compute the discriminator loss
            discriminator_loss = self.calculate_discriminator_loss(yhat_real, yhat_fake)

            # Compute the generator loss
            generator_loss = self.calculate_generator_loss(yhat_fake)

        # Get the gradients of the generator and discriminator loss
        discriminator_gradients = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)
        generator_gradients = gen_tape.gradient(generator_loss, generator.trainable_variables)

        # Update the weights of the generator and discriminator
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

        return discriminator_loss, generator_loss

    def train(self, dataset: tf.data.Dataset, noise_dim: int, generator: Model, discriminator: Model, epochs: int, batch_size: int, num_classes: int):
        """
        Train the GAN model
        Args:
            real_images: np.ndarray, real images
            noise_dim: int, dimension of the noise vector
            generator: tensorflow.keras.models.Model, generator model
            discriminator: tensorflow.keras.models.Model, discriminator model
            epochs: int, number of epochs
            batch_size: int, batch size
        Returns:
            None
        """
        date = datetime.now().strftime("%Y-%m-%d-%I-%M-%S")
        saved_dir = os.path.join(REPO_DIR_PATH, "results", "training_checkpoints_" + date)
        os.makedirs(saved_dir, exist_ok=True)
        os.makedirs(os.path.join(saved_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(saved_dir, "models"), exist_ok=True)
        for epoch in range(epochs):
            discriminator_loss = 0
            generator_loss = 0
            for image_batch, label_batch in dataset:
                discriminator_loss, generator_loss = self.train_step(batch_size, noise_dim, num_classes, generator, discriminator, image_batch, label_batch)
            print(f"Epoch {epoch+1}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}")
            if epoch % 5 == 0 or epoch == 0:
                fig = self.generate_images(generator, noise_dim, num_classes)
                fig.savefig(os.path.join(saved_dir, "images",
                            f"image_at_epoch_{epoch+1}.png"))
                generator.save(os.path.join(saved_dir, "models",
                               f"generator_at_epoch_{epoch+1}.h5"))
                discriminator.save(os.path.join(
                    saved_dir, "models", f"discriminator_at_epoch_{epoch+1}.h5"))

    def generate_images(self, generator_model: Model, noise_dim: int, num_classes: int) -> plt.Figure:
        """
        Generate images using generator model.
        Args:
            generator_model: tensorflow.keras.models.Model, generator model
            noise_dim: int, dimension of the noise vector
            num_classes: int, number of classes for the labels
        Returns:
            fig: matplotlib.pyplot.Figure, figure containing the generated images
        """
        # Generate random noise and labels
        noise = tf.random.normal([16, noise_dim])
        labels = tf.random.uniform([16], minval=0, maxval=num_classes, dtype=tf.int32)
        
        # Generate images
        generated_images = generator_model([noise, labels], training=False)
        
        # Convert generated images from [-1, 1] to [0, 1]
        generated_images = (generated_images + 1) / 2.0
        
        # Plot images and their labels
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        axes = axes.flatten()
        
        for img, label, ax in zip(generated_images, labels, axes):
            ax.imshow(img.numpy().squeeze(), cmap='gray')
            ax.set_title(f'Label: {int(label)}')
            ax.axis('off')
        
        plt.tight_layout()
        return fig
        


if __name__ == "__main__":
    trainer = ModelTrainerCGAN()
    images, labels = trainer.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    generator = trainer.build_generator()
    discriminator = trainer.build_discriminator()
    trainer.train(dataset, NOISE_DIM, generator, discriminator, EPOCHS, BATCH_SIZE)
