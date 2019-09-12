import numpy as np
import matplotlib.pyplot as plt

mask = np.load('./mask.npy')
fig = plt.figure()
plt.imshow(mask[:,:,0])
plt.colorbar()
plt.savefig('./fig.png')
plt.close()
fig.clf()
