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

parser = ArgumentParser(description='Mask Visualizer')

parser.add_argument('--h', default=8.5, type=float, help='figure height')
parser.add_argument('--w', default=10, type=float, help='figure width')
parser.add_argument('-fp', '--filename_prefix', nargs='+', help='<Required> filename prefix', required=True)
parser.add_argument('--models_dir', default='../models/', type=str, help='directory to save models')
parser.add_argument('--out_dir', default='../figs/', type=str, help='directory to save models')
parser.add_argument('--gpu_id', default=0, type=int, metavar='N', help='gpu id to train on')
parser.add_argument('--data_path', default='/nfs02/data/processed_nyu/NYU_training_Biograph.npy', type=str, help='path to data')
parser.add_argument('-t', '--type', choices=['l', 'm', 's'], type=str, required=True)

def add_bool_arg(parser, name, default=True):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no_' + name, dest=name, action='store_false')
    parser.set_defaults(**{name:default})

add_bool_arg(parser, 'train_cond')
add_bool_arg(parser, 'is_epi')

args = parser.parse_args()
if torch.cuda.is_available():
    args.device = torch.device('cuda:'+str(args.gpu_id))
else:
    args.device = torch.device('cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

parameter_dict = {'mode':'*', 'recon_type':'*', 'loss':'*', 'pmask_slope': '*', 'sample_slope': '*', 'sparsity': '*', 'lr': '*'}

for param in parameter_dict.keys():
    inp = input('Value of %s: ' % (param))
    if inp == '':
        continue
    else:
        parameter_dict[param] = inp

format_strings_by_dirs = {} # Contains all format strings in the query, organized by directory name 
paths_to_use = []
all_paths_by_dirs = {} # Contains all paths in the query, organized by directory name (a model has multiple .h5 files)
for prefix in args.filename_prefix:
    if args.type == 'l':
        format_string = '%s%s_{recon_type}_{mode}_{loss}_{pmask_slope}_{sample_slope}_{sparsity}_{lr}/losses.pkl' % (args.models_dir, prefix)
    elif args.type == 'm' or args.type == 's':
        format_string = '%s%s_{recon_type}_{mode}_{loss}_{pmask_slope}_{sample_slope}_{sparsity}_{lr}/model.*.h5' % (args.models_dir, prefix)

    format_string_filled = format_string.format(
        mode=parameter_dict['mode'],
        recon_type=parameter_dict['recon_type'],
        loss=parameter_dict['loss'],
        pmask_slope=parameter_dict['pmask_slope'],
        sample_slope=parameter_dict['sample_slope'],
        sparsity=parameter_dict['sparsity'],
        lr=parameter_dict['lr']
        )

    paths = glob.glob(format_string_filled)
    for path in paths:
        if os.path.dirname(path) not in all_paths_by_dirs: 
            all_paths_by_dirs[os.path.dirname(path)] = [path]
            format_strings_by_dirs[os.path.dirname(path)] = format_string
        else:
            all_paths_by_dirs[os.path.dirname(path)].append(path)

if args.type == 'l':
    for key in all_paths_by_dirs.keys():
        paths_to_use += all_paths_by_dirs[key]

elif args.type == 'm' or args.type == 's':
    latest_model_nums = []
    for key in all_paths_by_dirs.keys():
        latest_model = all_paths_by_dirs[key][-1]
        num = latest_model.split('.')[-2]
        format_strings_by_dirs[key] = format_strings_by_dirs[key].replace('*', num)
        paths_to_use.append(latest_model)

print(format_strings_by_dirs)
if len(paths_to_use) == 0:
    sys.exit('No matching models found.')
else:
    print('I found %s models that matched your query:' % (str(len(paths_to_use))))
    for g in paths_to_use:
        print(g)

cont = input('\nContinue? y/n: ')
if cont == 'y': pass
else: sys.exit('User stopped')

if args.type == 'l':
    fig, axes = plt.subplots(1, 1)
elif args.type == 'm':
    fig, axes = plt.subplots(1, len(paths_to_use))
    v = 0
elif args.type == 's':
    fig, axes = plt.subplots(2, len(paths_to_use)+1)
    v = 0

slice_to_use = 3000 # slice number for recons
if args.type == 's':
    print('loading data...')
    xdata = np.load(args.data_path, mmap_mode='r')
    print('done')

    im = axes[0, v].imshow(xdata[slice_to_use, ..., 0], cmap='gray')
    im = axes[1, v].imshow(xdata[slice_to_use, 100:175, 100:175 , 0], cmap='gray')
    axes[0, 0].set_title('ground truth')

for i, path in enumerate(paths_to_use):
    print(format_strings_by_dirs[os.path.dirname(path)])
    parsed = parse.parse(format_strings_by_dirs[os.path.dirname(path)], path)

    # Loss
    if args.type == 'l':
        f = open(path, 'rb') 
        losses = pickle.load(f)
        color = next(axes._get_lines.prop_cycler)['color']
        print(parsed['recon_type'])
        print(losses['loss'])
        plt.plot(losses['loss'], label='Train loss, recon type = %s' % (parsed['recon_type']), color=color)
        plt.plot(losses['val_loss'], label='Val loss, recon_type = %s' % (parsed['recon_type']), color=color, linestyle='dashed')
        plt.grid()
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        # plt.ylim([0.000, 0.0125])
        plt.grid(True)
        plt.title('EPI Loss')

    # Conditional Loupe Mask
    elif args.type == 'm' and args.train_cond:
        checkpoint = torch.load(path, map_location=torch.device(args.device))

        model = loupe_pytorch.model.CondLoupe((160, 224, 2), pmask_slope=5, sample_slope=100, \
                                                     sparsity=0.25, device=args.device)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(args.device)

        for i in [0., 0.5, 1.]:
            inp = torch.tensor([i]).view(1, 1).to(args.device)
            mask = model.mask(inp, get_prob_mask = False)
            mask = np.fft.fftshift(mask[0].detach().cpu().numpy())

            fig.suptitle('Masks for Knee')
            v += 1

    # Vanilla Loupe Mask
    elif args.type == 'm' and not args.train_cond:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = loupe_pytorch.model.Loupe((160, 224, 2), pmask_slope=float(parsed['pmask_slope']), sample_slope=float(parsed['sample_slope']),
                                                     sparsity=float(parsed['sparsity']), device=args.device, recon_type=parsed['recon_type'], 
                                                     is_epi=args.is_epi)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(args.device)

        mask = model.mask()
        mask = np.fft.fftshift(mask.detach().cpu().numpy())
        print(mask.shape)
        im = axes[v].imshow(mask, cmap='gray')
        axes[v].set_title('recon_type = %s' % (parsed['recon_type']))
        fig.colorbar(im, ax=axes[v])
        fig.suptitle('Masks for Knee')
        v += 1

    # Slice
    elif args.type == 's':
        print('loading data...')
        xdata = np.load(args.data_path, mmap_mode='r')
        print('done')
        v += 1
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = loupe_pytorch.model.Loupe((320, 320, 2), pmask_slope=float(parsed['pmask_slope']), sample_slope=float(parsed['sample_slope']),
                                                     sparsity=float(parsed['sparsity']), device=args.device, recon_type=parsed['recon_type'],
                                                     is_epi=args.is_epi)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(args.device)

        pred = model(torch.tensor(xdata[slice_to_use]).float().unsqueeze(0).to(args.device))
        pred = pred.cpu().detach().numpy()
        im = axes[0, v].imshow(pred[0,...,0], cmap='gray')
        axes[0, v].set_title('%s' % (parsed['recon_type']))
        im = axes[1, v].imshow(pred[0, 100:175, 100:175, 0], cmap='gray')
        axes[1, v].set_title(str(np.linalg.norm(xdata[slice_to_use] - pred[0])))
        fig.suptitle('Slices for Knee')


out_filename = '%s%s-%s.png' % (args.out_dir, args.filename_prefix, args.type)
plt.savefig(out_filename)
print('saved figure to ' + out_filename)
