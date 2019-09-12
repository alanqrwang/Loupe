import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.optim import Adam
import argparse
import pprint
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# User-defined modules
from model import Loupe
from utils import plot_figure, ReshapeTransform

parser = argparse.ArgumentParser(description='LOUPE in Pytorch')

parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--sparsity', default=0.1, type=float, metavar='N',
                    help='percentage of sparsity of undersampling')
parser.add_argument('--pmask_slope', default=5, type=float, metavar='N',
                    help='percentage of sparsity of undersampling')
parser.add_argument('--sample_slope', default=12, type=float, metavar='N',
                    help='percentage of sparsity of undersampling')

def main():
    args = parser.parse_args()
    args.device = None
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    pprint.pprint(vars(args))

    # Data loading
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.workers}
    transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])

    trainset = datasets.FashionMNIST('./F_MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, **params)
    testset = datasets.FashionMNIST('./F_MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, **params)

    image_dims = trainset[0][0][0].shape

    model = Loupe(image_dims, pmask_slope=args.pmask_slope, sample_slope=args.sample_slope, sparsity=args.sparsity, device=args.device)
    model = model.to(args.device)

    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    mask = None
    loss_list = []
    for epoch in range(args.epochs):
        for batch_idx, (x, _) in tqdm(enumerate(trainloader), total=int(len(trainset)/args.batch_size)):   
            x = x.view(-1, *image_dims, 1).to(args.device)
            output, mask = model(x, batch_idx)
            print('loss', output.shape, x.shape)
            loss = criterion(output, x)

            loss_list.append(loss)   
            if batch_idx % 50 == 0:
                plot_figure(mask[:,:,0].detach().cpu().numpy(), './intermediate/images/fig_%s_%s.png', epoch, batch_idx)

            print('loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    np.save('./mask.npy', mask.detach().cpu().numpy()) 
    fig = plt.figure()
    plt.plot(loss_list)
    plt.savefig('./loss.png') 
    plt.close()
    fig.clf()

if __name__ == "__main__":
    main()
