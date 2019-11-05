import glob
import matplotlib.pyplot as plt
import torch
import numpy as np
import loupe_pytorch
from pylab import subplot

# models = glob.glob('/home/aw847/loupe_pytorch/models/cond-vs-no-cond_True_relax_mse_5.0_100.0_0.25_0.001/model.29.h5')
# models = sorted(models)
model_filename = '/home/aw847/loupe_pytorch/models/mask-arch-1lcs_True_relax_mse_5.0_100.0_0.25_0.001/model.9991.h5'
# loss_list = []
# for model in models:
checkpoint = torch.load(model_filename)

model = loupe_pytorch.model.CondLoupe((320, 320, 2), pmask_slope=5, sample_slope=100, \
                                             sparsity=0.25, device='cuda')
model.load_state_dict(checkpoint['state_dict'])
model = model.to('cuda')
fig = plt.figure(figsize=[8.5, 10])
ax = plt.gca()
v = 0

for i in [0., 0.5, 1.]:
    inp = torch.tensor([i]).view(1, 1).to('cuda')
    mask = model.mask(inp)
    print(mask.shape)
    mask = np.fft.fftshift(mask[0].detach().cpu().numpy())
    v += 1
    ax = subplot(3, 1, v)
    ax.set_title(str(round(i, 2)))
    im = ax.imshow(mask, cmap='gray')
    # fig.colorbar(im, ax=ax)
    fig.suptitle('Masks for Knee, 1 LC layer')
    # loss = checkpoint['loss']
    # loss_list.append(loss)
# np.savetxt('./losses.txt', loss_list)
# plt.plot(loss_list)
plt.imshow(mask, cmap='gray')
plt.savefig('../figs/binary-dataset-10000-1lcs-mask.png')