import torch
import os
from torch import nn, optim
from VAE import VAE
import numpy as np
from torchvision.datasets import utils
import torch.utils.data as data_utils
import matplotlib.pyplot as plt


def get_data_loader(dataset_location, batch_size):
    URL = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
    # start processing

    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    splitdata = []
    for splitname in ["train", "valid", "test"]:
        filename = "binarized_mnist_%s.amat" % splitname
        filepath = os.path.join(dataset_location, filename)
        # utils.download_url(URL + filename, dataset_location, filename=filename, md5=None)
        with open(filepath) as f:
            lines = f.readlines()
        x = lines_to_np_array(lines).astype('float32')
        x = x.reshape(x.shape[0], 1, 28, 28)
        # pytorch data loader
        dataset = data_utils.TensorDataset(torch.from_numpy(x))
        dataset_loader = data_utils.DataLoader(x, batch_size=batch_size, shuffle=splitname == "train")
        splitdata.append(dataset_loader)
    return splitdata


def train(model, optimizer, epoch, train_loader, device):
    model.train()
    train_loss = 0.0
    for batch_idx, inputs in enumerate(train_loader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        recon_output, mu, logvar = model(inputs)
        elbo_loss = model.ELBO(recon_output, inputs, mu, logvar)
        elbo_loss.backward()
        train_loss += elbo_loss.item()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                elbo_loss.item() / len(inputs)))

    print('Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)


def test(model, epoch, test_loader, device):

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_loader):
            inputs = inputs.to(device)
            recon_output, mu, logvar = model(inputs)
            elbo_loss = model.ELBO(recon_output, inputs, mu, logvar)
            test_loss += elbo_loss.item()

            if batch_idx % 10 == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader),
                    elbo_loss.item() / len(inputs)))

    print('Epoch: {} Average loss: {:.4f}'.format(
          epoch, test_loss / len(test_loader.dataset)))
    return test_loss / len(test_loader.dataset)


def main(n_epochs, device, lr=3e-4):

    train_loader, valid_loader, test_loader = get_data_loader("binarized_mnist", 64)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    valid_losses = []

    # Training VAE
    for epoch in range(n_epochs):
        train_loss = train(model, optimizer, epoch, train_loader, device)
        valid_loss = test(model, epoch, valid_loader, device)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

    print("Saving the model")
    torch.save(model.state_dict(), "model.pth")
    plt.plot(train_losses, "train")
    plt.plot(valid_losses, "valid")
    plt.title("Learning curves")
    plt.savefig("Learning_curves.png")

if __name__ == "__main__":
    n_epochs = 20
    lr = 3e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(n_epochs, device, lr)
