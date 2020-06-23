import torch
import torch.nn as nn
from torch.optim import Adam
import argparse
import pprint
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import glob

# User-defined modules
import loupe_pytorch
from io_handler import io_handler

parser = argparse.ArgumentParser(description='LOUPE in Pytorch')

parser.add_argument('--nb_epochs_train', default=300, type=int, metavar='N',
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
parser.add_argument('--pmask_slope', default=0.25, type=float, metavar='N',
                    help='percentage of sparsity of undersampling')
parser.add_argument('--sample_slope', default=100, type=float, metavar='N',
                    help='percentage of sparsity of undersampling')
parser.add_argument('--num_modl_iters', default=5, type=int, metavar='N',
                    help='K value for MoDL')

parser.add_argument('--load_checkpoint', default=0, type=int, help='loss to use')
parser.add_argument('-fp', '--filename_prefix', type=str, help='filename prefix', required=True)
parser.add_argument('--models_dir', default='/nfs02/users/aw847/models/loupe_pytorch/', type=str, help='directory to save models')
parser.add_argument('--data_path', default='/nfs02/data/processed_nyu/NYU_training_Biograph.npy', type=str, help='path to data')
parser.add_argument('--val_data_path', default='/nfs02/data/processed_nyu/NYU_validation_Biograph.npy', type=str, help='path to data')

parser.add_argument('--loss', choices=['mse', 'mae'], type=str, help='loss to use', required=True)
parser.add_argument('--straight_through_mode', choices=['ste-identity', 'ste-sigmoid-fixed', 'ste-sigmoid-anneal', 'relax'], default='relax', type=str, help='straight-through type')
parser.add_argument('--recon_type', choices=['unet', 'cascade', 'modl'], default='unet', type=str, help='loss to use')

parser.add_argument('--train_type', choices=['loupe', 'cascade-mask', 'unet-mask', 'loupe-lines-mask',
                                             'spectrum-mask', 'coherence-mask', 'epi-vertical-mask',
                                             'gaussian-mask', 'uniform-mask', 'epi-horizontal-mask'], required=True, type=str, help='loss to use')

# loupe_pytorch.utils.add_bool_arg(parser, 'train_loupe')
loupe_pytorch.utils.add_bool_arg(parser, 'train_cond', default=False)
loupe_pytorch.utils.add_bool_arg(parser, 'is_epi', default=False)

def main():
    ###############################################
    # Argument parsing and gpu handling
    ###############################################
    args = parser.parse_args()
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


    print('Loading data...')

    if args.train_cond:
        xdata = np.load('./cond_dataset/binary_dataset.npy')
        conditions = np.load('./cond_dataset/binary_conditions.npy')
        trainset = loupe_pytorch.Dataset.CondDataset(xdata[:int(len(xdata)*0.7)], xdata[:int(len(xdata)*0.7)], conditions[:int(len(xdata)*0.7)])
        valset = loupe_pytorch.Dataset.CondDataset(xdata[int(len(xdata)*0.7):], xdata[int(len(xdata)*0.7):], conditions[int(len(xdata)*0.7):])
    else:
        print('loading', args.data_path)
        xdata = np.load(args.data_path)
        print('loading', args.val_data_path)
        xdata_val = np.load(args.val_data_path)
        # if xdata.shape[-1] == 1:
        #     print('Appending complex dimension into training set...')
        #     xdata = np.concatenate((xdata, np.zeros(xdata.shape)), axis=3)
        # if xdata_val.shape[-1] == 1:
        #     print('Appending complex dimension into validation set...')
        #     xdata_val = np.concatenate((xdata_val, np.zeros(xdata_val.shape)), axis=3)
        image_dims = xdata.shape[1:]
        trainset = loupe_pytorch.Dataset.Dataset(xdata, xdata)
        valset = loupe_pytorch.Dataset.Dataset(xdata_val, xdata_val)


    dataloaders = {
        'train': torch.utils.data.DataLoader(trainset, **params),
        'val': torch.utils.data.DataLoader(valset, **params)
    }

    print('done', xdata.shape, xdata_val.shape)

    ###############################################
    # Models
    ###############################################

    

    if args.train_type == 'loupe' and not args.train_cond:
        print('Training vanilla Loupe with %s recon, with epi = %s!' % (args.recon_type, str(args.is_epi)))
        model = loupe_pytorch.model.Loupe(image_dims=image_dims, 
                                          pmask_slope=args.pmask_slope, 
                                          sample_slope=args.sample_slope, 
                                          sparsity=args.desired_sparsity, 
                                          device=args.device, 
                                          is_epi=args.is_epi, 
                                          recon_type=args.recon_type,
                                          K=args.num_modl_iters)

    elif args.train_type == 'loupe' and args.train_cond:
        print('Training conditional Loupe with %s recon!' % (args.recon_type))

        model = loupe_pytorch.model.CondLoupe(image_dims=image_dims, 
                                              pmask_slope=args.pmask_slope,
                                              sample_slope=args.sample_slope,
                                              sparsity=args.desired_sparsity, 
                                              device=args.device, 
                                              recon_type=args.recon_type)

    else:
        print('Training %s recon with discrete mask, with epi = %s!' % (args.recon_type, str(args.is_epi)))
        model = loupe_pytorch.model.Loupe(image_dims=image_dims, 
                                          pmask_slope=args.pmask_slope, 
                                          sample_slope=args.sample_slope, 
                                          sparsity=args.desired_sparsity, 
                                          device=args.device, 
                                          is_epi=args.is_epi, 
                                          recon_type=args.recon_type,
                                          K=args.num_modl_iters)

    model = model.to(args.device)

    ###############################################
    # Loss and optimizer
    ###############################################
    if args.loss == 'mae':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=args.lr)

    ###############################################
    # I/O user input and model saving
    ###############################################
    # prepare save sub-folder
    local_name = '{prefix}_{recon_type}_{straight_through_mode}_{loss}_{pmask_slope}_{sample_slope}_{sparsity}_{lr}_{is_epi}'.format(
        prefix=args.filename_prefix,
        straight_through_mode=args.straight_through_mode,
        recon_type=args.recon_type,
        loss=args.loss,
        pmask_slope=args.pmask_slope,
        sample_slope=args.sample_slope,
        sparsity=args.desired_sparsity,
        lr=args.lr,
        is_epi=args.is_epi)

    save_dir_loupe = os.path.join(args.models_dir, local_name) 
    filename, model, optimizer, sample_mask = io_handler(save_dir_loupe, args, model, optimizer)
    
    if args.train_type != 'loupe':
        sample_mask = loupe_pytorch.utils.get_sample_mask(args.train_type, args.loss, args.desired_sparsity, args.device)
    else:
        sample_mask = None
    ###############################################
    # Train model
    ###############################################
    model, train_loss, val_loss = loupe_pytorch.utils.train_model(model, criterion, optimizer, dataloaders, args.nb_epochs_train,
                                    args.device, filename, args.straight_through_mode, args.load_checkpoint, args.train_cond, sample_mask)

    ###############################################
    # Save training loss
    ###############################################
    # if args.load_checkpoint != 0:
    #     f = open(os.path.join(save_dir_loupe, 'losses.pkl'), 'rb') 
    #     old_losses = pickle.load(f)
    #     loss_dict = {'loss' : old_losses['loss'] + train_loss, 'val_loss' : old_losses['val_loss'] + val_loss}
    # else:
    #     loss_dict = {'loss' : train_loss, 'val_loss' : val_loss}

    # if args.train_loupe:
    #     loss_filename = os.path.join(save_dir_loupe, 'losses.pkl')
    # else:
    #     loss_filename = os.path.join(save_dir_loupe, 'unet-losses.pkl')

    # f = open(loss_filename,"wb")
    # pickle.dump(loss_dict,f)
    # f.close()
    # print('saved loss to', loss_filename)

if __name__ == "__main__":
    main()
