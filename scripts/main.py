import torch
import torch.nn as nn
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
import glob
import sys

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
parser.add_argument('--data_path', default='/nfs02/data/processed_nyu/NYU_training_Biograph.npy', type=str, help='path to data')
parser.add_argument('--loss', default='mae', type=str, help='loss to use')
parser.add_argument('--load_checkpoint', default=0, type=int, help='loss to use')
parser.add_argument('--straight_through_mode', default='relax', type=str, help='loss to use')

parser.add_argument('--train_loupe', dest='train_loupe', action='store_true')
parser.add_argument('--train_unet', dest='train_loupe', action='store_false')
parser.set_defaults(train_loupe=True)

def save_checkpoint(epoch, model_state, optimizer_state, loss, val_loss, filename):
    state = {
            'epoch': epoch+1,
            'state_dict': model_state,
            'optimizer' : optimizer_state,
            'loss' : loss,
            'val_loss' : val_loss
        }
    torch.save(state, filename.format(epoch=epoch+1))

def train_model(model, criterion, optimizer, dataloaders, num_epochs, device, filename, straight_through_mode, load_checkpoint):
    loss_list = []
    val_loss_list = []
    for epoch in range(load_checkpoint, num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                print('TRAIN EPOCH ' + str(epoch))
                model.train()
            else:
                print('VALIDATE EPOCH ' + str(epoch))
                model.eval()

            epoch_loss = 0
            epoch_samples = 0

            for batch_idx, (x, _, condition) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):   
                x = x.to(device)
                condition = condition.float().to(device)
                condition = condition.unsqueeze(1)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(x, condition)
                    loss = criterion(output, x)

                    epoch_loss += loss.data.cpu().numpy() * x.size(0)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += x.size(0)

            epoch_loss /= epoch_samples
            if phase == 'train':
                train_epoch_loss = epoch_loss
                loss_list.append(epoch_loss)
            else:
                val_epoch_loss = epoch_loss
                val_loss_list.append(epoch_loss)

        if epoch % 10 == 0:
            save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(), train_epoch_loss, val_epoch_loss, filename)

    return model, loss_list, val_loss_list

def main():
    ###############################################
    # Argument parsing and gpu handling
    ###############################################
    args = parser.parse_args()
    args.device = None
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpu_id))
    else:
        args.device = torch.device('cpu')

    pprint.pprint(vars(args))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


    ###############################################
    # Data loading
    ###############################################
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.workers}


    print('loading data...')
    # xdata = np.load(args.data_path)
    xdata = np.load('./cond_dataset/binary_dataset.npy')
    conditions = np.load('./cond_dataset/binary_conditions.npy')
    print(conditions)
    trainset = loupe_pytorch.Dataset.CondDataset(xdata[:int(len(xdata)*0.7)], xdata[:int(len(xdata)*0.7)], conditions[:int(len(xdata)*0.7)])
    valset = loupe_pytorch.Dataset.CondDataset(xdata[int(len(xdata)*0.7):], xdata[int(len(xdata)*0.7):], conditions[int(len(xdata)*0.7):])
    dataloaders = {
        'train': torch.utils.data.DataLoader(trainset, **params),
        'val': torch.utils.data.DataLoader(valset, **params)
    }

    if xdata.shape[-1] == 1:
        print('appending complex dimension into dataset')
        xdata = np.concatenate((xdata, np.zeros(xdata.shape)), axis=3)

    image_dims = xdata.shape[1:]
    print('done')

    ###############################################
    # Define models
    ###############################################
    if args.train_loupe:
        print('I\'m training conditional Loupe!')
        model = loupe_pytorch.model.CondLoupe(image_dims, pmask_slope=args.pmask_slope, sample_slope=args.sample_slope, \
                                             sparsity=args.desired_sparsity, device=args.device)
    else:
        model = loupe_pytorch.model.UnetLoupe(image_dims, sample_slope=args.sample_slope, device=args.device, sample_mask=sample_mask)
    model = model.to(args.device)

    ###############################################
    # Define loss and optimizer
    ###############################################
    assert args.loss in ['mae', 'mse'], 'loss must be mae or mse'
    if args.loss == 'mae':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=args.lr)

    ###############################################
    # I/O user input and model saving
    ###############################################
    # prepare save sub-folder
    local_name = '{prefix}_{straight_through_mode}_{loss}_{pmask_slope}_{sample_slope}_{sparsity}_{lr}'.format(
        prefix=args.filename_prefix,
        straight_through_mode=args.straight_through_mode,
        loss=args.loss,
        pmask_slope=args.pmask_slope,
        sample_slope=args.sample_slope,
        sparsity=args.desired_sparsity,
        lr=args.lr)

    save_dir_loupe = os.path.join(args.models_dir, local_name)
    # Model with specified parameters has never been trained before
    if not os.path.isdir(save_dir_loupe):   
        # Can't load checkpoint if model doesn't exist
        if args.load_checkpoint != 0:
            sys.exit('No existing model was found.')
        # Load benchmark mask
        elif args.load_checkpoint == 0 and not args.train_loupe and args.masktype != 'default': 
            print('\nTraining unet only on benchmark mask %s with sparsity level %s\n' % (args.masktype, str(args.desired_sparsity)))
            assert args.masktype in ['EPI', 'Gaussian', 'Uniform'], 'invalid masktype'
            assert args.desired_sparsity in [0.125, 0.25], 'invalid sparsity level'
            filename = os.path.join(save_dir_loupe, 'model.{epoch:02d}.h5')
            confirm = input('Creating new model filename: %s. Continue? y/n ' % filename)
            if confirm == 'y':
                os.makedirs(save_dir_loupe)
            else:
                sys.exit()

            if args.desired_sparsity == 0.125:
                local_bench_mask = '/nfs02/data/processed_nyu/Masks/%s_12.npy' % (args.masktype)
                print('loading ' + local_bench_mask)
                sample_mask = np.load(local_bench_mask)
            elif args.desired_sparsity == 0.25:
                local_bench_mask = '/nfs02/data/processed_nyu/Masks/%s_25.npy' % (args.masktype)
                print('loading ' + local_bench_mask)
                sample_mask = np.load(local_bench_mask)
            else:
                sys.exit('Error in loading benchmark mask')

            sample_mask = np.fft.fftshift(sample_mask)
            plt.imshow(sample_mask, cmap='gray') 
            plt.colorbar()
            plt.savefig('%s_benchmark_mask.png' % (args.masktype))

            if len(sample_mask.shape) == 2:
                sample_mask = np.expand_dims(sample_mask, 0)
                sample_mask = np.expand_dims(sample_mask, -1)
            if sample_mask.shape[-1] == 1:
                sample_mask = np.concatenate((sample_mask, np.zeros(sample_mask.shape)), axis=3)
            print(sample_mask.shape)
            sample_mask = np.repeat(sample_mask, len(xdata), axis=0)
            print(sample_mask.shape)
            filename = os.path.join(save_dir_loupe, 'unet-model.{epoch:02d}.h5') 
        # Create new model filename
        else:
            filename = os.path.join(save_dir_loupe, 'model.{epoch:02d}.h5')
            confirm = input('Creating new model filename: %s. Continue? y/n ' % filename)
            if confirm == 'y':
                os.makedirs(save_dir_loupe)
            else:
                sys.exit()

    # Model with specified parameters HAS been trained before
    else: 
        # Not loading checkpoint and training loupe+unet
        if args.load_checkpoint == 0 and args.train_loupe:             
            confirm = input('This model already exists, but I\'m not loading the latest checkpoint. The existing model will be overwritten. Continue? y/n ')
            if confirm == 'y':
                filename = os.path.join(save_dir_loupe, 'model.{epoch:02d}.h5')
            else:
                sys.exit()

        # Not loading checkpoint and training unet only
        elif args.load_checkpoint == 0 and not args.train_loupe: 
            print('\nNot loading checkpoint and training unet only\n')
            all_models = glob.glob(os.path.join(save_dir_loupe, 'model.*.h5'))
            confirm = input('I found %s trained models. Load sample mask and train unet? y/n ' % (str(len(all_models))))
            if confirm == 'y':
                prob_mask = np.load(os.path.join(save_dir_loupe, 'mask.npy'))
                prob_mask = squash_mask(torch.tensor(prob_mask))
                prob_mask = sparsify(prob_mask)
                prob_mask = prob_mask[..., 0].detach().cpu().numpy()

                sample_mask = np.random.binomial(1, prob_mask)
                print(sample_mask.shape)
                mask_filename = os.path.join(save_dir_loupe, 'sample_mask.npy')
                print('Saving sampled mask to %s' % mask_filename)
                np.save(mask_filename, sample_mask)

                sample_mask = torch.tensor(sample_mask).float().to(args.device)
                filename = os.path.join(save_dir_loupe, 'unet-model.{epoch:02d}.h5') 
            else:
                sys.exit()

        # Loading checkpoint 
        else: 
            all_models = glob.glob(os.path.join(save_dir_loupe, 'model.*.h5'))
            confirm = input('I found %s trained models. Load %s? y/n ' % (str(len(all_models)), args.load_checkpoint))
            if confirm == 'y':
                filename = os.path.join(save_dir_loupe, 'model.{epoch:02d}.h5')
                checkpoint = torch.load(filename.format(epoch=args.load_checkpoint))
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                sys.exit()

    ###############################################
    # Train model
    ###############################################
    model, train_loss, val_loss = train_model(model, criterion, optimizer, dataloaders, args.nb_epochs_train, \
                                    args.device, filename, args.straight_through_mode, args.load_checkpoint)

    ###############################################
    # Save training loss
    ###############################################
    if args.load_checkpoint != 0:
        f = open(os.path.join(save_dir_loupe, 'losses.pkl'), 'rb') 
        old_losses = pickle.load(f)
        loss_dict = {'loss' : old_losses['loss'] + train_loss, 'val_loss' : old_losses['val_loss']}
    else:
        loss_dict = {'loss' : train_loss, 'val_loss' : val_loss}

    if args.train_loupe:
        loss_filename = os.path.join(save_dir_loupe, 'losses.pkl')
    else:
        loss_filename = os.path.join(save_dir_loupe, 'unet-losses.pkl')

    f = open(loss_filename,"wb")
    pickle.dump(loss_dict,f)
    f.close()
    print('saved loss to', loss_filename)

    ###############################################
    # Save learned mask
    ###############################################
    # if args.train_loupe and not args.is_conditional:
    #     mask = model.mask(args.straight_through_mode)
    #     if len(mask.shape) == 3:
    #         mask = mask[0].detach().cpu().numpy()
    #     else:
    #         mask = mask.detach().cpu().numpy()
    #     mask_filename = os.path.join(save_dir_loupe, 'mask.npy')
    #     np.save(mask_filename, mask) 
    #     print('saved mask to', mask_filename)

if __name__ == "__main__":
    main()
