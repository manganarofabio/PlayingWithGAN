import torch
from torch import nn, optim
from torchvision import transforms, datasets
import argparse, random, numpy as np
import models
from utilities import noise, Logger
from trainer import Trainer

def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])
        ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)


parser = argparse.ArgumentParser(description='PyTorch gan')
parser.add_argument('--seed', type=int, default=1994, metavar='S',
                    help='random seed (default: 1994)')

args = parser.parse_args()
def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # Load data
    data = mnist_data()
    # Create loader with data, so that we can iterate over it
    data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
    # Num batches
    num_batches = len(data_loader)

    discriminator = models.DiscriminatorNet().to(device)
    generator = models.GeneratorNet().to(device)

    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    loss_function = nn.BCELoss().to(device)

    num_test_samples = 16
    test_noise = noise(num_test_samples)

    num_epochs = 200

    # Create logger instance
    logger = Logger(model_name='VGAN', data_name='MNIST')
    # Total number of epochs to train
    trainer = Trainer(discriminator, generator, loss_function, d_optimizer, g_optimizer, data_loader, 100, num_batches,
                      logger, num_test_samples, num_epochs, test_noise, device)




    for ep in range(num_epochs):
        trainer.train(ep)


if __name__ == '__main__':
    main()

