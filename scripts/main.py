import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.optim import Adam
import argparse
import pprint
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import copy
from collections import defaultdict
import pickle

# User-defined modules
import loupe_pytorch

parser = argparse.ArgumentParser(description='LOUPE in Pytorch')

parser.add_argument('--nb_epochs_train', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--gpu_id', default=0, type=int,
                    metavar='N', help='gpu id to train on')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--desired_sparsity', default=0.05, type=float, metavar='N',
                    help='percentage of sparsity of undersampling')
parser.add_argument('--pmask_slope', default=5, type=float, metavar='N',
                    help='percentage of sparsity of undersampling')
parser.add_argument('--sample_slope', default=12, type=float, metavar='N',
                    help='percentage of sparsity of undersampling')

parser.add_argument('--filename_prefix', default='loupe-pytorch-mri', type=str, help='filename prefix')
parser.add_argument('--models_dir', default='../models/', type=str, help='directory to save models')
parser.add_argument('--data_path', default='/nfs02/users/gid-dalcaav/projects/neuron/data/t1_mix/proc/resize256-crop_x32/LOUPE_sel_2D_MRI_SLICE/xdata.npy', type=str, help='path to data')
parser.add_argument('--loss', default='mae', type=str, help='loss to use')
parser.add_argument('--mode', default='relax', type=str, help='loss to use')
parser.add_argument('--load_checkpoint', default=0, type=int, help='loss to use')

def train_model(model, criterion, optimizer, dataloaders, num_epochs, batch_size, device, filename, mode, load_checkpoint):
    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_loss = 1e10
    # mask = None
    loss = 0
    loss_list = []
    val_loss_list = []
    for epoch in range(load_checkpoint, load_checkpoint+num_epochs):
        since = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                print('TRAIN EPOCH ' + str(epoch))
                model.train()
            else:
                print('VALIDATE EPOCH ' + str(epoch))
                model.eval()

            # metrics = defaultdict(float)
            # metrics['loss'] = 0
            # epoch_samples = 0

            for batch_idx, (x, _) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):   
                x = x.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(x, epoch, num_epochs, mode)
                    loss = criterion(output, x)

                    # metrics['loss'] += loss.data.cpu().numpy() * x.size(0)

                    if phase == 'train':
                        if batch_idx == 0:
                            loss_list.append(loss)
                        loss.backward()
                        optimizer.step()
                    else:
                        if batch_idx == 0:
                            val_loss_list.append(loss)

                # statistics
                # epoch_samples += x.size(0)

            # epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            # if phase == 'val' and epoch_loss < best_loss:
            #     print("saving best model")
            #     best_loss = epoch_loss
            #     best_model_wts = copy.deepcopy(model.state_dict())

        torch.save(model.state_dict(), filename.format(epoch=epoch))

    #     time_elapsed = time.time() - since
    #     print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # print('Best val loss: {:4f}'.format(best_loss))
    
    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model, loss_list, val_loss_list

def main():
    args = parser.parse_args()
    args.device = None
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpu_id))
    else:
        args.device = torch.device('cpu')

    pprint.pprint(vars(args))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # prepare save sub-folder
    local_name = '{prefix}_{mode}_{loss}_{pmask_slope}_{sample_slope}_{sparsity}_{lr}'.format(
        prefix=args.filename_prefix,
        mode=args.mode,
        loss=args.loss,
        pmask_slope=args.pmask_slope,
        sample_slope=args.sample_slope,
        sparsity=args.desired_sparsity,
        lr=args.lr)
    save_dir_loupe = os.path.join(args.models_dir, local_name)
    if not os.path.isdir(save_dir_loupe): os.makedirs(save_dir_loupe)
    filename = os.path.join(save_dir_loupe, 'model.{epoch:02d}.h5')
    print('model filenames: %s' % filename)

    # Data loading
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.workers}

    print('loading data...')
    xdata = np.load(args.data_path)
    trainset = loupe_pytorch.Dataset.Dataset(xdata[:int(len(xdata)*0.7)], xdata[:int(len(xdata)*0.7)])
    valset = loupe_pytorch.Dataset.Dataset(xdata[int(len(xdata)*0.7):], xdata[int(len(xdata)*0.7):])
    dataloaders = {
        'train': torch.utils.data.DataLoader(trainset, **params),
        'val': torch.utils.data.DataLoader(valset, **params)
    }
    print('done')

    if xdata.shape[-1] == 1:
        print('appending complex dimension into dataset')
        x = torch.cat((x, torch.zeros(x.shape).to(self.device)), dim=3)

    image_dims = xdata.shape[1:]

    model = loupe_pytorch.model.Loupe(image_dims, pmask_slope=args.pmask_slope, sample_slope=args.sample_slope, sparsity=args.desired_sparsity, device=args.device)
    model = model.to(args.device)
    # Load checkpoint
    if args.load_checkpoint != 0:
        print('LOADING FROM ' + filename)
        model.load_state_dict(torch.load(filename.format(epoch=args.load_checkpoint-1)))

    assert args.loss in ['mae', 'mse'], 'loss must be mae or mse'
    if args.loss == 'mae':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=args.lr)

    model, train_loss, val_loss = train_model(model, criterion, optimizer, dataloaders, args.nb_epochs_train, args.batch_size, args.device, filename, args.mode, args.load_checkpoint)

    # Save mask
    mask = model.state_dict()['pmask']
    mask_filename = os.path.join(save_dir_loupe, 'mask.npy')
    np.save(mask_filename, mask.detach().cpu().numpy()) 
    print('saved mask to', mask_filename)

    # Save training loss
    if args.load_checkpoint != 0:
        f = open(os.path.join(save_dir_loupe, 'losses.pkl'), 'rb') 
        old_losses = pickle.load(f)
        loss_dict = {'loss' : old_losses['loss'] + train_loss, 'val_loss' : old_losses['val_loss']}
    else:
        loss_dict = {'loss' : train_loss, 'val_loss' : val_loss}

    loss_filename = os.path.join(save_dir_loupe, 'losses.pkl')
    f = open(loss_filename,"wb")
    pickle.dump(loss_dict,f)
    f.close()
    print('saved loss to', loss_filename)

if __name__ == "__main__":
    main()
