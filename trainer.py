from utilities import ones_target, zeros_target
import torch
from utilities import images_to_vectors, vectors_to_images, noise







class Trainer(object):
    def __init__(self, discriminator, generator, loss, d_optimizer, g_optimizer, train_loader, batch_size, num_batches,
                 logger, num_test_samples, num_epochs, test_noise, device):

        self.discriminator = discriminator
        self.generator = generator
        self.loss = loss,
        self.d_optimizer = d_optimizer,
        self.g_optimizer = g_optimizer,
        self.train_loader = train_loader
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.logger = logger
        self.num_test_samples = num_test_samples
        self.num_epochs = num_epochs
        self.test_noise = test_noise
        self.device = device

    def train_discriminator(self, real_data, fake_data):
        N = real_data.size(0)
        # Reset gradients
        self.d_optimizer[0].zero_grad()

        # 1.1 Train on Real Data
        prediction_real = self.discriminator(real_data)
        # Calculate error and backpropagate
        error_real = self.loss[0](prediction_real, ones_target(N).to(self.device))
        error_real.backward()

        # 1.2 Train on Fake Data
        prediction_fake = self.discriminator(fake_data)
        # Calculate error and backpropagate
        error_fake = self.loss[0](prediction_fake, zeros_target(N).to(self.device))
        error_fake.backward()

        # 1.3 Update weights with gradients
        self.d_optimizer[0].step()

        # Return error and predictions for real and fake inputs
        return error_real + error_fake, prediction_real, prediction_fake

    def train_generator(self, fake_data):
        N = fake_data.size(0)
        # Reset gradients
        self.g_optimizer[0].zero_grad()
        # Sample noise and generate fake data
        prediction = self.discriminator(fake_data)
        # Calculate error and backpropagate
        error = self.loss[0](prediction, ones_target(N).to(self.device))
        error.backward()
        # Update weights with gradients
        self.g_optimizer[0].step()
        # Return error
        return error

    def train(self, epoch):

        for n_batch, (real_batch, foo) in enumerate(self.train_loader):
            N = real_batch.size(0)

            # 1. Train Discriminator
            real_data = torch.tensor(images_to_vectors(real_batch))
            real_data = real_data.to(self.device)

            # Generate fake data and detach
            # (so gradients are not calculated for generator)
            fake_data = self.generator(noise(N).to(self.device)).detach()
            fake_data = fake_data


            # Train D
            d_error, d_pred_real, d_pred_fake = self.train_discriminator(real_data, fake_data)

            # 2. Train Generator
            # Generate fake data
            fake_data = self.generator(noise(N).to(self.device))
            fake_data = fake_data.to(self.device)
            # Train G
            g_error = self.train_generator(fake_data)
            # Log batch error
            self.logger.log(d_error, g_error, epoch, n_batch, self.num_batches)
            # Display Progress every few batches
            if (n_batch) % 100 == 0:
                test_images = vectors_to_images(self.generator(self.test_noise.to(self.device)))
                test_images = test_images.data
                self.logger.log_images(
                    test_images, self.num_test_samples,
                    epoch, n_batch, self.num_batches
                )
                # Display status Logs
                self.logger.display_status(
                    epoch, self.num_epochs, n_batch, self.num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )


