import torch
import numpy as np
import matplotlib.pyplot as plt
import loupe_pytorch
from argparse import ArgumentParser
import pickle
import sys
import os
import glob
import parse
from pylab import subplot
from functools import reduce

parser = ArgumentParser(description='Mask Visualizer')

parser.add_argument('--h', default=8.5, type=float, help='figure height')
parser.add_argument('--w', default=10, type=float, help='figure width')
parser.add_argument('-fp', '--filename_prefix', default='loupe_v2_slope_experiments', type=str, help='filename prefix')
parser.add_argument('--models_dir', default='../models/', type=str, help='directory to save models')
parser.add_argument('-o', '--out_name', default='vis_mask_out', type=str, help='directory to save models')
parser.add_argument('--out_dir', default='../figs/', type=str, help='directory to save models')
parser.add_argument('--gpu_id', default=0, type=int, metavar='N', help='gpu id to train on')
parser.add_argument('--data_path', default='/nfs02/data/processed_nyu/NYU_training_Biograph.npy', type=str, help='path to data')


args = parser.parse_args()

args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda:'+str(args.gpu_id))
else:
    args.device = torch.device('cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

parameter_dict = {'mode':'*', 'loss':'*', 'pmask_slope': '*', 'sample_slope': '*', 'sparsity': '*', 'lr': '*'}

for param in parameter_dict.keys():
    inp = input('Value of %s: ' % (param))
    if inp == '':
        continue
    else:
        parameter_dict[param] = inp

to_visualize = input("loss, mask, or slice? l/m/s ") 
if to_visualize == 'l':
    format_string = '%s%s_{mode}_{loss}_{pmask_slope}_{sample_slope}_{sparsity}_{lr}/losses.pkl' % (args.models_dir, args.filename_prefix)
elif to_visualize == 'm':
    format_string = '%s%s_{mode}_{loss}_{pmask_slope}_{sample_slope}_{sparsity}_{lr}/model.141.h5' % (args.models_dir, args.filename_prefix)
elif to_visualize == 's':
    format_string = '%s%s_{mode}_{loss}_{pmask_slope}_{sample_slope}_{sparsity}_{lr}/model.249.h5' % (args.models_dir, args.filename_prefix)
else:
    sys.exit('Invalid input')

format_string_filled = format_string.format(
    mode=parameter_dict['mode'],
    loss=parameter_dict['loss'],
    pmask_slope=parameter_dict['pmask_slope'],
    sample_slope=parameter_dict['sample_slope'],
    sparsity=parameter_dict['sparsity'],
    lr=parameter_dict['lr'])

print(format_string_filled)
filepaths = glob.glob(format_string_filled)
if len(filepaths) == 0 and not args.is_conditional:
    sys.exit('No matching models found.')
else:
    print('I found %s models that matched your query:' % (str(len(filepaths))))
    for g in filepaths:
        print(g)

cont = input('\nContinue? y/n: ')
if cont == 'y': pass
else: sys.exit('User stopped')

fig = plt.figure(figsize=[args.w, args.h])
ax = plt.gca()
v = 0

subplot_nrow = 1
subplot_ncol = 3

# print('loading data...')
# xdata = np.load(args.data_path, mmap_mode='r')
# print('done')
for i, path in enumerate(filepaths):
    parsed = parse.parse(format_string, path)

    # model = loupe_pytorch.model.Loupe((xdata.shape[1], xdata.shape[2], xdata.shape[3]), \
    #     pmask_slope=float(parsed['pmask_slope']), sample_slope=float(parsed['sample_slope']),\
    #     sparsity=float(parsed['sparsity']), device='cuda', straight_through_mode='relax')

    # Loss
    if path.endswith('.pkl'):
        f = open(path, 'rb') 
        losses = pickle.load(f)
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(losses['loss'], label='Train loss, loss = %s, sparsity = %s' % (parsed['loss'], parsed['sparsity']), color=color)
        plt.plot(losses['val_loss'], label='Val loss, loss = %s, sparsity = %s' % (parsed['loss'], parsed['sparsity']), color=color, linestyle='dashed')
        plt.grid()
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        # plt.ylim([0, max(max(losses['loss']), max(losses['val_loss']))])
        plt.ylim([0.005, 0.0125])
        plt.grid(True)
        plt.title('EPI Loss')

    elif path.endswith('.h5'):
        checkpoint = torch.load(path)

        model = loupe_pytorch.model.CondLoupe((320, 320, 2), pmask_slope=5, sample_slope=100, \
                                                     sparsity=0.25, device='cuda')
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(args.device)

        for i in [0., 0.5, 1.]:
            inp = torch.tensor([i]).view(1, 1).to(args.device)
            mask = model.mask(inp)
            print(mask.shape)
            mask = np.fft.fftshift(mask[0].detach().cpu().numpy())
            v += 1
            ax = subplot(subplot_nrow, subplot_ncol, v)
            ax.set_title(str(round(i, 2)))
            im = ax.imshow(mask, cmap='gray')
            # fig.colorbar(im, ax=ax)
            fig.suptitle('Masks for Knee')
            # loss = checkpoint['loss']
            # loss_list.append(loss)
        # np.savetxt('./losses.txt', loss_list)
        # plt.plot(loss_list)
        plt.imshow(mask, cmap='gray')

    # Mask
    elif path.endswith('.npy') and not args.is_conditional:
        mask = np.load(path)
        mask = np.fft.fftshift(mask)
        v += 1
        ax = subplot(subplot_nrow, subplot_ncol, v)
        ax.set_title('loss = %s, sparsity = %s' % (parsed['loss'], parsed['sparsity']))
        im = ax.imshow(mask, cmap='gray')
        fig.colorbar(im, ax=ax)
        fig.suptitle('Masks for Knee')

    # Slice
    elif path.endswith('.h5'):
        model.load_state_dict(torch.load(path))
        model = model.to(args.device)
        pred = model(torch.tensor(xdata[3000]).float().unsqueeze(0).to(args.device), mode=parsed['mode'])
        v += 1
        ax = subplot(subplot_l, subplot_w, v)
        ax.set_title('loss = %s, sparsity = %s' % (parsed['loss'], parsed['sparsity']))
        im = ax.imshow(pred[0,:,:,0].cpu().detach().numpy(), cmap='gray')
        fig.colorbar(im, ax=ax)

# lossesses = np.loadtxt('./losses.txt')
# plt.plot(lossesses)
out_filename = '%s%s.png' % (args.out_dir, args.out_name)
plt.savefig(out_filename)
print('saved figure to ' + out_filename)
